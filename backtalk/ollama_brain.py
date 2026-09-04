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
"""OllamaBrain — a local, offline drop-in for WarmBrain (brain.py),
talking to a locally running Ollama server instead of Claude.

CONVERSATION ONLY: unlike WarmBrain, this brain has no file or shell
tools wired in, and never will without a real permission-gated tool
loop to match main.py's spoken "ask" gate. Switching to it trades
Claude's tool access for a fully offline, no-cost brain. Implements
just the surface main.py actually calls on `brain`, so the voice loop,
the signal bus, and the face all keep working unmodified regardless of
which brain is live.
"""
import json
import os
import re

import httpx

from backtalk.config import CFG, DISCIPLINE
from backtalk.vlog import log

_SENTENCE_END = re.compile(r"(?<=[.!?])\s")

# Tool-call rounds offered before ask_stream forces a tools-withheld
# final round. Generous on purpose: without "think" mode, this model
# explores somewhat aimlessly (measured live: 4 rounds just to find the
# right file for a question a smarter search would answer in 1-2).
MAX_TOOL_ROUNDS = 6

# READ-ONLY on purpose: no write/edit/shell tool here. Adding those needs
# their own spoken confirmation UX and a lot more thought about what
# "undo" means for a model with no memory of Claude's safety rails.
_TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read the text contents of a file inside the "
                       "agent's allowed folders.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string",
                     "description": "Path to the file, absolute or "
                                    "relative to the agent's folder."}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "list_directory",
        "description": "List the files and folders inside a directory "
                       "in the agent's allowed folders.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string",
                     "description": "Path to the folder, absolute or "
                                    "relative to the agent's folder."}},
                       "required": ["path"]}}},
]


def _allowed_roots():
    roots = [CFG.get("agent_dir", "")] + list(CFG.get("extra_dirs") or [])
    return [os.path.realpath(r) for r in roots if r]


def _resolve_in_bounds(path: str) -> str | None:
    """Resolve `path` against the agent's own folder and confirm it
    lands inside agent_dir or one of extra_dirs (the same folders the
    person already granted Claude). None if it's outside every allowed
    root — this brain never gets to see more of the disk than Claude
    already can."""
    base = CFG.get("agent_dir", ".")
    candidate = path if os.path.isabs(path) else os.path.join(base, path)
    real = os.path.realpath(candidate)
    for root in _allowed_roots():
        if real == root or real.startswith(root + os.sep):
            return real
    return None


def _run_read_file(path: str) -> str:
    real = _resolve_in_bounds(path)
    if real is None:
        return f"Denied: {path!r} is outside the folders I'm allowed to read."
    try:
        with open(real, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError as e:
        return f"Couldn't read {path!r}: {e}"
    if len(content) > 20000:
        content = content[:20000] + "\n...[truncated, the file continues]"
    return content


def _run_list_directory(path: str) -> str:
    real = _resolve_in_bounds(path)
    if real is None:
        return f"Denied: {path!r} is outside the folders I'm allowed to look in."
    try:
        entries = sorted(os.listdir(real))
    except OSError as e:
        return f"Couldn't list {path!r}: {e}"
    return "\n".join(entries) or "(empty folder)"


def _read_identity() -> str:
    """Ollama has no "claude_code" preset system prompt to fall back
    on, so the agent's own CLAUDE.md IS the system prompt here — same
    identity file, same rules, minus the tool-use sections that don't
    apply. Best-effort: an unreadable file degrades to a blank
    identity rather than crashing the brain."""
    path = os.path.join(CFG["agent_dir"], "CLAUDE.md")
    try:
        with open(path, "r", encoding="utf-8") as f:
            identity = f.read()
    except OSError:
        log(f"[ollama] couldn't read {path}, using a bare identity")
        identity = f"You are {CFG['name']}, a helpful voice assistant."
    return identity + (
        "\n\nYou're currently running as a local, offline model instead "
        "of your usual Claude self. You have two read-only tools: "
        "read_file and list_directory, scoped to your own folder and "
        "your extra_dirs. Use them whenever a question needs you to "
        "actually look at something, rather than guessing. You cannot "
        "write, edit, or run commands right now — say so plainly if "
        "asked to.")


class OllamaBrain:
    def __init__(self, model: str | None = None, permission_gate=None):
        self.model = model or CFG.get("ollama_model", "qwen3.5:9b")
        self.url = CFG.get("ollama_url", "http://localhost:11434")
        # Shape-compatible with WarmBrain.session so _spoken_usage in
        # main.py works unmodified; cost stays 0, it's your own machine.
        self.session = {"turns": 0, "out_tokens": 0, "in_tokens": 0,
                        "cost": 0.0}
        self._messages = [{"role": "system",
                           "content": _read_identity() + "\n\n" + DISCIPLINE}]
        self._client: httpx.AsyncClient | None = None
        # async def gate(tool_name: str, tool_input: dict) -> bool — the
        # same spoken yes/no/details flow Claude's tools already use.
        # Safety is opt-out, not opt-in: with no gate wired, every tool
        # call is DENIED rather than silently allowed (see ask_stream).
        self._permission_gate = permission_gate

    async def start(self):
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=10.0))
        # Prove the server is actually reachable now, not on the first
        # real question — a clear failure here beats a silent hang with
        # the face stuck on "thinking".
        r = await self._client.get(f"{self.url}/api/tags")
        r.raise_for_status()

    async def stop(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def interrupt(self):
        # No-op: asyncio's own task cancellation (main.py's
        # speak_task.cancel()) already tears down the in-flight
        # request via the `async with` stream context manager below.
        pass

    async def reset_turn(self, timeout: float = 8.0):
        # WarmBrain needs this to re-align a shared message pipe after
        # an interrupt; Ollama is one request per turn with no shared
        # pipe to desync, so there's nothing to do.
        pass

    async def set_permission_mode(self, mode: str):
        pass  # no tools wired in, so nothing to gate

    async def context_usage(self):
        return None

    async def command(self, cmd: str) -> str:
        """A small subset of WarmBrain's slash commands. Local
        equivalents, not the real thing: /compact here is NOT
        summarization (that would need its own model call), just a
        truncation — said plainly so it never reads as feature parity
        it doesn't have."""
        cmd = cmd.strip()
        if cmd == "/clear":
            self._messages = self._messages[:1]
            return ""
        if cmd == "/compact":
            self._messages = self._messages[:1] + self._messages[-2:]
            return ""
        if cmd.startswith("/model "):
            self.model = cmd.split(" ", 1)[1].strip()
            return ""
        if cmd.startswith("/effort "):
            return ("local models here don't support adjustable "
                    "reasoning effort; ignoring")
        return ""

    async def _run_tool(self, name: str, args: dict) -> str:
        allowed = await self._permission_gate(name, args) \
            if self._permission_gate else False
        if not allowed:
            return "Permission denied by the user."
        if name == "read_file":
            return _run_read_file(str(args.get("path", "")))
        if name == "list_directory":
            return _run_list_directory(str(args.get("path", "")))
        return f"Unknown tool: {name}"

    async def ask_stream(self, utterance: str):
        """Yield complete sentences as they stream out of the model.
        Mirrors WarmBrain.ask_stream's shape so speak_reply() in
        main.py needs no changes to work with either brain.

        Handles read_file/list_directory tool calls in a loop: each
        round streams text (spoken as it lands) and may end with tool
        calls instead of/alongside a final answer; those get gated,
        executed, and fed back as a normal chat turn until the model
        stops asking for tools.

        Capped at MAX_TOOL_ROUNDS rounds where tools are OFFERED, plus
        one guaranteed extra round with tools withheld entirely, which
        forces a prose answer from whatever's been gathered so far
        instead of ending on a dangling tool call. Skipping that last
        round was the exact bug this comment is warning future-you off:
        measured live, the model found and read the right file on
        what would have been its LAST allowed round, then had nothing
        left to answer with — the correct data was in `_messages` and
        got thrown away because the loop just ended."""
        self._messages.append({"role": "user", "content": utterance})
        for round_n in range(MAX_TOOL_ROUNDS + 1):
            force_final = round_n == MAX_TOOL_ROUNDS
            buf = ""
            assistant_text = ""
            tool_calls = None
            # think=False: Qwen3's hidden reasoning mode burns 2000+
            # tokens on trivial questions (measured: 30s to first word
            # on "what's 2+2", vs under half a second with it off). A
            # voice loop lives or dies on latency, same philosophy as
            # backtalk keeping Claude on the fast tier.
            payload = {"model": self.model, "messages": self._messages,
                      "stream": True, "think": False}
            if not force_final:
                payload["tools"] = _TOOLS
            try:
                async with self._client.stream(
                        "POST", f"{self.url}/api/chat",
                        json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        msg = chunk.get("message") or {}
                        piece = msg.get("content", "")
                        if piece:
                            assistant_text += piece
                            buf += piece
                            while True:
                                m = _SENTENCE_END.search(buf)
                                if not m:
                                    break
                                sentence, buf = (buf[:m.end()].strip(),
                                                 buf[m.end():])
                                if sentence:
                                    yield sentence
                        if msg.get("tool_calls"):
                            tool_calls = msg["tool_calls"]
                        if chunk.get("done"):
                            self.session["turns"] += 1
                            self.session["in_tokens"] += int(
                                chunk.get("prompt_eval_count") or 0)
                            self.session["out_tokens"] += int(
                                chunk.get("eval_count") or 0)
                            break
            except Exception as e:
                log(f"[ollama] request failed: {e!r}")
                yield ("I lost the connection to the local model. Check "
                      "that Ollama is still running.")
                return
            tail = buf.strip()
            if tail:
                yield tail
            self._messages.append({
                "role": "assistant",
                "content": assistant_text,
                **({"tool_calls": tool_calls} if tool_calls else {}),
            })
            if not tool_calls or force_final:
                return   # a real answer, or the forced final round — done
            for call in tool_calls:
                fn = call.get("function") or {}
                name = fn.get("name", "")
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                result = await self._run_tool(name, args)
                self._messages.append({
                    "role": "tool", "content": result,
                    "tool_call_id": call.get("id", ""), "name": name,
                })
            # loop back: the model gets the tool result as the next turn
            # (unreachable on the forced-final round, which always returns
            # above before reaching here — tools are withheld that round)
