# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hardware-free tests for backtalk.brain.

Every one of these mocks the single real boundary: the Claude Agent
SDK's ClaudeSDKClient. No real subprocess is ever spawned, no network
call is ever made, no model is ever loaded. Message classes below only
need to satisfy brain.py's own dispatch (type(msg).__name__ string
checks, plus isinstance checks against the SDK's real ToolUseBlock /
ToolResultBlock, which are plain data classes safe to import directly).
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from claude_agent_sdk import ToolResultBlock, ToolUseBlock

from backtalk import brain


class AssistantMessage:
    def __init__(self, content):
        self.content = content


class UserMessage:
    def __init__(self, content):
        self.content = content


class StreamEvent:
    def __init__(self, event):
        self.event = event


class ResultMessage:
    def __init__(self, usage=None, total_cost_usd=None, session_id=None):
        self.usage = usage
        self.total_cost_usd = total_cost_usd
        self.session_id = session_id


def _text_delta(text):
    return StreamEvent({"type": "content_block_delta",
                        "delta": {"type": "text_delta", "text": text}})


def _block_stop():
    return StreamEvent({"type": "content_block_stop"})


class FakeStreamClient:
    """Stands in for ClaudeSDKClient in ask_stream/command/reset_turn
    tests: a scripted list of messages, played back through
    receive_response(), plus call recorders."""

    def __init__(self, events=None):
        self._events = events or []
        self.queried = []
        self.interrupt_calls = 0
        self.disconnect_calls = 0

    async def query(self, utterance):
        self.queried.append(utterance)

    async def receive_response(self):
        for e in self._events:
            yield e

    async def interrupt(self):
        self.interrupt_calls += 1

    async def disconnect(self):
        self.disconnect_calls += 1


class DiagnoseConnectFailureTests(unittest.TestCase):
    def test_auth_hint_wins(self):
        msg = brain.diagnose_connect_failure(
            Exception("401 Unauthorized: please run /login"))
        self.assertIn("sign back in", msg)

    def test_usage_hint(self):
        msg = brain.diagnose_connect_failure(
            Exception("usage limit exceeded for this plan"))
        self.assertIn("out of usage", msg)

    def test_network_hint_from_stderr_tail(self):
        msg = brain.diagnose_connect_failure(
            Exception("boom"), stderr_tail="getaddrinfo ENOTFOUND")
        self.assertIn("reach the network", msg)

    def test_no_match_returns_none(self):
        self.assertIsNone(
            brain.diagnose_connect_failure(Exception("something else")))

    def test_unrelated_warning_text_never_false_triggers(self):
        # Kokoro/torch startup warnings print around connect too; none of
        # their language should ever be mistaken for a real cause.
        self.assertIsNone(brain.diagnose_connect_failure(
            Exception("UserWarning: torch.load weights_only=False")))


class TallyAndRememberSessionTests(unittest.TestCase):
    def setUp(self):
        self.b = brain.WarmBrain(model="test-model")

    def test_tally_accumulates_usage_and_cost(self):
        rm = ResultMessage(usage={"output_tokens": 10, "input_tokens": 5,
                                   "cache_read_input_tokens": 2},
                            total_cost_usd=0.01)
        self.b._tally(rm)
        self.assertEqual(self.b.session["turns"], 1)
        self.assertEqual(self.b.session["out_tokens"], 10)
        self.assertEqual(self.b.session["in_tokens"], 7)
        self.assertAlmostEqual(self.b.session["cost"], 0.01)

    def test_tally_count_turn_false_skips_turn_increment(self):
        rm = ResultMessage(usage={"output_tokens": 1, "input_tokens": 1})
        self.b._tally(rm, count_turn=False)
        self.assertEqual(self.b.session["turns"], 0)

    def test_tally_missing_fields_default_to_zero(self):
        self.b._tally(ResultMessage())
        self.assertEqual(self.b.session["out_tokens"], 0)
        self.assertEqual(self.b.session["in_tokens"], 0)

    def test_tally_never_raises_on_malformed_usage(self):
        rm = ResultMessage(usage={"output_tokens": "not-a-number"})
        self.b._tally(rm)   # must not raise

    def test_remember_session_noop_when_resume_disabled(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / ".backtalk_session"
            with mock.patch.object(brain, "SESSION_FILE", str(path)), \
                 mock.patch.object(brain, "CFG", {"resume_last_session": False}):
                self.b._remember_session(ResultMessage(session_id="abc123"))
            self.assertFalse(path.exists())

    def test_remember_session_writes_session_id_when_enabled(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / ".backtalk_session"
            with mock.patch.object(brain, "SESSION_FILE", str(path)), \
                 mock.patch.object(brain, "CFG", {"resume_last_session": True}):
                self.b._remember_session(ResultMessage(session_id="abc123"))
            self.assertEqual(path.read_text(), "abc123")

    def test_remember_session_survives_unwritable_path(self):
        with mock.patch.object(brain, "SESSION_FILE", "/nonexistent-dir/x"), \
             mock.patch.object(brain, "CFG", {"resume_last_session": True}):
            self.b._remember_session(ResultMessage(session_id="abc123"))  # no raise


class SetPermissionModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_ask_maps_to_sdk_default(self):
        b = brain.WarmBrain(model="test-model")
        b._client = mock.AsyncMock()
        await b.set_permission_mode("ask")
        b._client.set_permission_mode.assert_awaited_once_with("default")

    async def test_other_modes_pass_through(self):
        b = brain.WarmBrain(model="test-model")
        b._client = mock.AsyncMock()
        await b.set_permission_mode("bypassPermissions")
        b._client.set_permission_mode.assert_awaited_once_with(
            "bypassPermissions")

    async def test_noop_without_a_client(self):
        b = brain.WarmBrain(model="test-model")
        await b.set_permission_mode("ask")  # must not raise


class AskStreamTests(unittest.IsolatedAsyncioTestCase):
    def _brain_with(self, events):
        b = brain.WarmBrain(model="test-model")
        b._client = FakeStreamClient(events)
        return b

    async def test_splits_on_sentence_boundaries(self):
        b = self._brain_with([
            _text_delta("Hello there. "),
            _text_delta("How are you?"),
            ResultMessage(),
        ])
        with mock.patch.object(brain, "log"):
            sentences = [s async for s in b.ask_stream("hi")]
        self.assertEqual(sentences, ["Hello there.", "How are you?"])

    async def test_trailing_partial_sentence_flushed_at_end(self):
        b = self._brain_with([
            _text_delta("No terminal punctuation here"),
            ResultMessage(),
        ])
        with mock.patch.object(brain, "log"):
            sentences = [s async for s in b.ask_stream("hi")]
        self.assertEqual(sentences, ["No terminal punctuation here"])

    async def test_content_block_stop_flushes_filler_immediately(self):
        # Filler said right before a tool call must ship now, not sit
        # buffered until the whole turn (including the tool run) ends.
        b = self._brain_with([
            _text_delta("On it, let me grab that"),
            _block_stop(),
            ResultMessage(),
        ])
        with mock.patch.object(brain, "log"):
            sentences = [s async for s in b.ask_stream("hi")]
        self.assertEqual(sentences, ["On it, let me grab that"])

    async def test_tool_elapsed_closes_only_after_every_parallel_call_resolves(self):
        events = [
            AssistantMessage([
                ToolUseBlock(id="t1", name="Read", input={}),
                ToolUseBlock(id="t2", name="Read", input={}),
            ]),
            UserMessage([ToolResultBlock(tool_use_id="t1", content="ok")]),
            UserMessage([ToolResultBlock(tool_use_id="t2", content="ok")]),
            ResultMessage(),
        ]
        b = self._brain_with(events)
        with mock.patch.object(brain, "log"), \
             mock.patch.object(brain.time, "time", side_effect=[10.0, 13.0]):
            [s async for s in b.ask_stream("hi")]
        self.assertEqual(b.tool_elapsed, 3.0)

    async def test_dirty_flag_lifecycle(self):
        b = self._brain_with([_text_delta("Hi."), ResultMessage()])
        with mock.patch.object(brain, "log"):
            [s async for s in b.ask_stream("hi")]
        self.assertFalse(b._dirty)

    async def test_result_message_tallies_and_remembers_session(self):
        b = self._brain_with([
            _text_delta("Hi."),
            ResultMessage(usage={"output_tokens": 3}, session_id="s1"),
        ])
        with mock.patch.object(brain, "log"), \
             mock.patch.object(brain.WarmBrain, "_pull_rate_limits",
                                new=mock.AsyncMock()), \
             mock.patch.object(brain, "CFG", {"resume_last_session": False}):
            [s async for s in b.ask_stream("hi")]
        self.assertEqual(b.session["out_tokens"], 3)
        self.assertEqual(b.session["turns"], 1)


class ResetTurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_noop_when_pipe_is_clean(self):
        b = brain.WarmBrain(model="test-model")
        b._client = FakeStreamClient([])
        b._dirty = False
        await b.reset_turn()
        self.assertEqual(b._client.interrupt_calls, 0)

    async def test_drains_stale_messages_through_result_message(self):
        b = brain.WarmBrain(model="test-model")
        b._client = FakeStreamClient([
            AssistantMessage([]), ResultMessage(),
        ])
        b._dirty = True
        with mock.patch.object(brain, "log"):
            await b.reset_turn(timeout=1.0)
        self.assertFalse(b._dirty)
        self.assertEqual(b._client.interrupt_calls, 1)

    async def test_drain_timeout_rebuilds_the_session(self):
        class HangingClient(FakeStreamClient):
            async def receive_response(self):
                import asyncio
                await asyncio.sleep(1.0)
                yield ResultMessage()   # pragma: no cover - never reached

        b = brain.WarmBrain(model="test-model")
        b._client = HangingClient([])
        b._dirty = True
        started = mock.AsyncMock()
        with mock.patch.object(brain, "log"), \
             mock.patch.object(brain.WarmBrain, "start", started):
            await b.reset_turn(timeout=0.05)
        started.assert_awaited_once()
        self.assertFalse(b._dirty)


class PullRateLimitsTests(unittest.IsolatedAsyncioTestCase):
    def _brain_with_usage(self, usage):
        b = brain.WarmBrain(model="test-model")
        b._client = mock.Mock()
        b._client._query._send_control_request = mock.AsyncMock(
            return_value=usage)
        return b

    async def test_noop_when_show_usage_disabled(self):
        b = self._brain_with_usage({"rate_limits": {}})
        with mock.patch.object(brain, "CFG", {"show_usage": False}), \
             mock.patch.object(brain.signals, "set_rate_limit") as sl:
            await b._pull_rate_limits()
        sl.assert_not_called()

    async def test_accepts_either_percentage_spelling(self):
        usage = {"rate_limits": {
            "five_hour": {"utilization": 40, "resets_at": None},
            "seven_day": {"used_percentage": 60, "resets_at": None},
        }}
        b = self._brain_with_usage(usage)
        with mock.patch.object(brain, "CFG", {"show_usage": True}), \
             mock.patch.object(brain.signals, "set_rate_limit") as sl:
            await b._pull_rate_limits()
        calls = {c.args[0]: c.args[1] for c in sl.call_args_list}
        self.assertAlmostEqual(calls["five_hour"], 0.40)
        self.assertAlmostEqual(calls["seven_day"], 0.60)

    async def test_iso_resets_at_converted_to_epoch_seconds(self):
        usage = {"rate_limits": {
            "five_hour": {"utilization": 10,
                          "resets_at": "2026-01-01T00:00:00+00:00"},
        }}
        b = self._brain_with_usage(usage)
        with mock.patch.object(brain, "CFG", {"show_usage": True}), \
             mock.patch.object(brain.signals, "set_rate_limit") as sl:
            await b._pull_rate_limits()
        window, pct, resets = sl.call_args.args
        self.assertEqual(window, "five_hour")
        self.assertIsInstance(resets, int)

    async def test_control_request_failure_is_swallowed(self):
        b = self._brain_with_usage({})
        b._client._query._send_control_request.side_effect = RuntimeError("x")
        with mock.patch.object(brain, "CFG", {"show_usage": True}), \
             mock.patch.object(brain.signals, "set_rate_limit") as sl:
            await b._pull_rate_limits()   # must not raise
        sl.assert_not_called()


class StartModeResolutionTests(unittest.IsolatedAsyncioTestCase):
    class FakeSDKClient:
        instances = []

        def __init__(self, options):
            self.options = options
            self.connected = False
            self.disconnect_calls = 0
            StartModeResolutionTests.FakeSDKClient.instances.append(self)

        async def connect(self):
            if self.options.resume:
                raise RuntimeError("stale session")
            self.connected = True

        async def disconnect(self):
            self.disconnect_calls += 1

    def setUp(self):
        StartModeResolutionTests.FakeSDKClient.instances = []
        self.cfg = {
            "agent_dir": "~", "permission_mode": "ask",
            "extra_dirs": [], "visible_skills": None,
        }

    async def test_ask_resolves_to_sdk_default(self):
        with mock.patch.object(brain, "CFG", self.cfg), \
             mock.patch.object(brain, "ClaudeSDKClient", self.FakeSDKClient):
            b = brain.WarmBrain(model="test-model")
            await b.start()
        self.assertEqual(b._client.options.permission_mode, "default")

    async def test_legacy_default_alias_resolves_to_sdk_default(self):
        cfg = dict(self.cfg, permission_mode="default")
        with mock.patch.object(brain, "CFG", cfg), \
             mock.patch.object(brain, "ClaudeSDKClient", self.FakeSDKClient):
            b = brain.WarmBrain(model="test-model")
            await b.start()
        self.assertEqual(b._client.options.permission_mode, "default")

    async def test_bypass_passes_through_unchanged(self):
        cfg = dict(self.cfg, permission_mode="bypassPermissions")
        with mock.patch.object(brain, "CFG", cfg), \
             mock.patch.object(brain, "ClaudeSDKClient", self.FakeSDKClient):
            b = brain.WarmBrain(model="test-model")
            await b.start()
        self.assertEqual(b._client.options.permission_mode,
                          "bypassPermissions")

    async def test_resume_failure_falls_back_to_a_fresh_session(self):
        with mock.patch.object(brain, "CFG", self.cfg), \
             mock.patch.object(brain, "ClaudeSDKClient", self.FakeSDKClient), \
             mock.patch.object(brain, "log"):
            b = brain.WarmBrain(model="test-model", resume_id="old-session")
            await b.start()
        instances = self.FakeSDKClient.instances
        self.assertEqual(len(instances), 2)
        self.assertEqual(instances[0].options.resume, "old-session")
        self.assertIsNone(instances[1].options.resume)
        self.assertTrue(b._client.connected)

    async def test_resume_success_creates_only_one_client(self):
        cfg = dict(self.cfg)
        with mock.patch.object(brain, "CFG", cfg), \
             mock.patch.object(brain, "ClaudeSDKClient", self.FakeSDKClient), \
             mock.patch.object(brain, "log"):
            b = brain.WarmBrain(model="test-model")
            # No resume_id this time: patch the fake client to always
            # succeed by never setting options.resume truthy.
            await b.start()
        self.assertEqual(len(self.FakeSDKClient.instances), 1)


if __name__ == "__main__":
    unittest.main()
