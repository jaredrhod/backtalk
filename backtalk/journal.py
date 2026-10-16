"""Session-end journaling: write down what a voice session actually was.

The agent's memory lives in files that a human curates. That is the right
arrangement -- but it only works if something remembers to write, and until
now nothing did, so memory rotted between sessions.

This module does the narrow, honest half of the job: it records what was
said and writes it down. It does NOT decide what is worth remembering.
Choosing a durable lesson out of a conversation is judgement, and
backtalk has none -- config.py is explicit that the agent's personality
lives in the AGENTS.md of whatever folder agent_dir points at. So this
writes the facts, including a plain transcript, and leaves the meaning to
whoever reads it next.

Summarising costs a model call, so it is optional and bounded: if the brain
is slow or unreachable the entry is still written, just without a summary.
A journal that sometimes skips the summary is useful. A hangup that hangs
on a model call is not.

This writes nothing unless `journal_dir` is configured. Off by default,
because it writes outside backtalk's own directory and that should be a
choice, not a surprise.
"""
from __future__ import annotations

import asyncio
import datetime
import os
import re
from pathlib import Path

# A marathon session must not produce a file nobody will read. Kept generous:
# the cap is a safety net, not a policy.
MAX_EVENTS = 400
MAX_CHARS = 4000

_UNSAFE = re.compile(r"[^0-9A-Za-z _-]+")

SUMMARY_PROMPT = """\
Below is a transcript of one voice session. Write a journal entry for it.

Write for someone reading this weeks from now. Include, where the \
transcript supports it: what was worked on, decisions that were made, \
things said about the user that would be worth remembering next time, \
and anything that looks like a mistake worth not repeating. Plain prose, \
no headings, no markdown, no bullet points. If the session was trivial, \
say so in one sentence rather than padding it out.

TRANSCRIPT:
{transcript}"""


def _safe(text: str) -> str:
    """A filename from a timestamp that is safe on every platform."""
    out = _UNSAFE.sub("-", text).strip().replace(" ", "_")
    return out or "session"


class Journal:
    """Accumulates one session, then writes it out."""

    def __init__(self, directory: str | os.PathLike | None,
                 agent_name: str = "Seyon"):
        self.directory = Path(directory).expanduser() if directory else None
        self.agent_name = agent_name
        self.events: list[tuple[str, str]] = []
        self.started = datetime.datetime.now()

    @property
    def active(self) -> bool:
        """Whether journaling is switched on at all."""
        return self.directory is not None

    @property
    def turns(self) -> int:
        return sum(1 for role, _ in self.events if role == "user")

    def say_user(self, text: str) -> None:
        self._add("user", text)

    def say_agent(self, text: str) -> None:
        self._add("agent", text)

    def say_activity(self, text: str) -> None:
        """A line about what was on screen, not something anyone said.

        Kept as its own role so a reader can tell the two apart: the
        transcript stays a conversation and the activity is clearly the
        machine's own note, not a turn the user took.
        """
        self._add("activity", text)

    def _add(self, role: str, text: str) -> None:
        if not self.active:
            return
        text = " ".join((text or "").split())
        if not text:
            return
        if len(self.events) >= MAX_EVENTS:
            return
        self.events.append((role, text[:MAX_CHARS]))

    def transcript(self) -> str:
        def who(role: str) -> str:
            if role == "user":
                return "you"
            if role == "activity":
                return "activity"
            return self.agent_name.lower()
        return "\n".join("%s: %s" % (who(r), t) for r, t in self.events)

    def render(self, summary: str = "") -> str:
        """The markdown body. Separate from writing so it can be tested."""
        started = self.started
        ended = datetime.datetime.now()
        mins = int((ended - started).total_seconds() // 60)
        lines = [
            "# Voice session, %s" % started.strftime("%Y-%m-%d %H:%M"),
            "",
            "- started: %s" % started.strftime("%H:%M"),
            "- ended: %s (%d min)" % (ended.strftime("%H:%M"), mins),
            "- turns: %d" % self.turns,
            "- written by: backtalk, automatically",
            "",
        ]
        if summary.strip():
            lines += ["## What happened", "", summary.strip(), ""]
        lines += ["## Transcript", ""]
        for role, text in self.events:
            who = {"user": "you",
                   "activity": "activity"}.get(role, self.agent_name)
            lines.append("**%s:** %s" % (who, text))
        lines += ["", "<!-- written by backtalk; edit freely, it is yours -->",
                  ""]
        return "\n".join(lines)

    def write(self, summary: str = "") -> Path | None:
        """Write the entry. Never raises: a failed journal must not stop a
        hangup. Returns the path written, or None if nothing was written."""
        if not self.active or not self.events:
            return None
        directory = self.directory
        if directory is None:  # pragma: no cover -- active implies a directory
            return None
        try:
            directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = directory / ("voice-session_%s.md" % _safe(stamp))
            # Seconds are not enough. Starting, stopping and restarting inside
            # one second is ordinary, and the second file would silently
            # overwrite the first: a lost session, with no error anywhere.
            # So if the name is taken, add a counter until it is not.
            n = 2
            while path.exists():
                path = directory / ("voice-session_%s_%d.md"
                                    % (_safe(stamp), n))
                n += 1
            path.write_text(self.render(summary), encoding="utf-8")
            return path
        except OSError:
            return None


async def summarize(ask, transcript: str, timeout: float = 25.0) -> str:
    """Ask the brain for a summary of one session.

    `ask` is an async generator function taking a prompt and yielding
    sentence strings -- the same shape as WarmBrain.ask_stream -- so this
    can be tested without a brain.

    Returns "" on any failure or timeout. An entry without a summary is
    still an entry; a hangup that waits on a model is a hangup that hangs.
    """
    if not transcript.strip():
        return ""

    async def run() -> str:
        parts = []
        async for sentence in ask(SUMMARY_PROMPT.format(
                transcript=transcript[-MAX_CHARS * 4:])):
            parts.append(sentence)
        # The stream yields whole sentences that already carry their own
        # trailing space, so a plain " ".join() doubles every gap.
        return " ".join("".join(parts).split()).strip()

    try:
        return await asyncio.wait_for(run(), timeout=timeout)
    except (asyncio.TimeoutError, Exception):
        return ""