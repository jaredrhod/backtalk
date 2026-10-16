"""speak_reply must always end with the bus parked and the listener told.

Four ways a turn can fail. brain.ask_stream speaks for three of them --
it yields an apology for a timeout, a session error and an empty reply --
so this file is about the shape of the code around them, plus the fourth
case where the PROMPT itself cannot be sent and ask_stream raises.

That fourth one used to be unhandled: speak_reply is started as a
fire-and-forget task whose try only catches CancelledError, so the
exception sat in the task until the listener's NEXT utterance happened
to collect it (main.py awaits it before starting a new turn). In between
there was silence, with the face still showing "thinking" -- which reads
as a broken microphone, not as a brain that could not be reached.

Run: uv run python tests/test_turn_errors.py
"""
from pathlib import Path
import asyncio
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtalk import main as btmain                    # noqa: E402


class FakeMouth:
    def __init__(self):
        self.chunks: list[str] = []
        self.said: list[str] = []

    def say_chunk(self, s, pending=None):
        self.chunks.append(s)

    def say(self, s):
        self.said.append(s)


class PromptFailsBrain:
    """Raises the way brain.ask_stream does when the prompt cannot be sent."""

    async def ask_stream(self, text):
        raise RuntimeError("prompt failed")
        yield                                       # noqa: W0101 (generator)


class EmptyBrain:
    """Says nothing at all and returns cleanly."""

    async def ask_stream(self, text):
        if False:
            yield ""


class MidStreamBrain:
    """One sentence out, then dies: audio is already queued and playing."""

    async def ask_stream(self, text):
        yield "First sentence, already queued."
        raise RuntimeError("connection reset")


def run(brain, mouth, text="do something"):
    """Drive speak_reply against a fake brain, returning the signals mock."""
    with mock.patch.object(btmain, "signals") as sig, \
         mock.patch.object(btmain, "log"):
        asyncio.run(btmain.speak_reply(brain, mouth, text))
    return sig


class TestFailedTurn(unittest.TestCase):

    def test_a_prompt_failure_does_not_escape(self):
        """The task used to die holding the exception. It must return."""
        run(PromptFailsBrain(), FakeMouth())         # would raise before

    def test_a_prompt_failure_parks_the_bus(self):
        """Nothing was queued, so nothing will dequeue: park it here."""
        sig = run(PromptFailsBrain(), FakeMouth())
        sig.set_state.assert_called_with("idle")
        sig.static_stop.assert_called()
        sig.caption_clear.assert_called()

    def test_a_prompt_failure_is_answered_out_loud(self):
        """Silence is the one reply that cannot explain itself."""
        mouth = FakeMouth()
        run(PromptFailsBrain(), mouth)
        self.assertEqual(len(mouth.said), 1)
        self.assertIn("could not reach", mouth.said[0])
        self.assertEqual(mouth.chunks, [])

    def test_an_empty_turn_parks_the_bus_too(self):
        """The pre-existing zero-sentence path, held to the same rule."""
        sig = run(EmptyBrain(), FakeMouth())
        sig.set_state.assert_called_with("idle")
        sig.static_stop.assert_called()

    def test_a_partial_reply_is_left_to_drain(self):
        """Audio already queued drains and parks on its own (reply_done).
        Parking here would blank the caption the listener is reading."""
        mouth = FakeMouth()
        sig = run(MidStreamBrain(), mouth)
        self.assertEqual(mouth.chunks, ["First sentence, already queued."])
        self.assertEqual(mouth.said, [])             # no apology over speech
        sig.set_state.assert_not_called()
        sig.caption_clear.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
