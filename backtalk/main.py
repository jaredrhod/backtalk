# backtalk: talk to your opencode agent out loud.
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
"""backtalk — talk to your opencode agent out loud.

Flow: hold the key and speak -> local transcription -> your agent's warm
opencode session streams the reply -> sentences go to the mouth the moment
they complete (~1-2s to first audio on warm turns). The greeting plays
over a hidden warmup query so the first real turn is already hot.

Typing in this terminal is a first-class turn too: same conversation,
spoken reply, and typing while it talks interrupts it.

THE VOICE CONSOLE: exact phrases, spoken (or typed) alone, control the
session itself so you never go back to the keyboard: "clear the
session" / "compact the session" / "switch to the deep model" / "back
to the fast model" / "set effort to low" (or medium, high, max) /
"usage report" / "go hands free" and "push to talk mode" (the MIC) /
"stop asking for permission" and "start asking again" (permissions,
called auto-approve, a different axis than the microphone on purpose).
And with permission_mode "ask" (the default), gated tool calls ASK OUT
LOUD and your spoken yes or no decides them; any other answer is
passed back to the agent as the reason.

Flags:
  --open-mic   start in hands-free listening for this session (the
               config key mic_mode makes it the standing default, and
               the voice can switch live either way: "go hands free" /
               "push to talk mode"). Know the tradeoff: room audio (a
               video, music, another voice assistant) can trigger
               replies to speech never meant for the agent. The talk
               key keeps working: it interrupts, and holding it always
               gets you heard.
  --barge-in   with --open-mic: keep listening WHILE speaking.
               HEADPHONES REQUIRED — with open speakers the mic hears
               the reply and the agent interrupts itself.
  --wake-word  with --open-mic: ignore speech until a wake phrase
               (config key wake_word_phrases, default "hey seyon") is
               heard; say the command in the same breath or the next
               one. "hey seyon, what's the weather" and "hey seyon"
               then "what's the weather" both work. The talk key stays
               direct. Costs one extra transcription pass per turn.
  --model X    override the model for this session (full id).

Say "goodbye <name>" / "end voice mode" to hang up. Ctrl-C works.
"""
import asyncio
import json
import os
import queue
import random
import re
import socket
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from backtalk import board, care, recall, screen, signals, voiceprint
from backtalk.brain import WarmBrain
from backtalk.config import CFG
from backtalk.ears import (Ears, explain_audio_failure, record_held,
                           strip_wake, warm as warm_ears)
from backtalk.journal import Journal, summarize as journal_summarize
from backtalk.mouth import Mouth, set_turn_language
from backtalk.ptt import PTTListener
from backtalk.vlog import log

NAME = CFG["name"]
QUIT_PHRASES = CFG["quit_phrases"]

# Waits that never finish on their own -- ptt.wait_press, typed_q.get,
# the open-mic capture, the ears warm -- run HERE, not in asyncio's
# default executor. Runner.close() joins every default-executor worker,
# so one still inside wait_press made "end voice mode" log "[backtalk]
# hung up" and then hang FOREVER with the instance lock held (py-spy:
# Thread-4 _do_shutdown -> executor.shutdown -> join, MainThread parked
# in runners.close). This pool is invisible to the runner; the clean
# exit kills its threads with os._exit like every other one.
_BLOCKERS = ThreadPoolExecutor(max_workers=8, thread_name_prefix="blocker")

# ---- THE SPOKEN PERMISSION GATE (permission_mode "ask", the default).
# When the agent wants a gated tool, the SDK routes the decision here:
# the ask is spoken, the turn pauses (the SDK waits indefinitely; the
# timeout below is ours), and the NEXT utterance or typed line is the
# answer. "yes" approves; anything else denies, with the user's own
# words passed back as the reason. Silence means no.
PERM_TIMEOUT_S = 75
_PERM: dict[str, Any] = {"fut": None, "asked_at": 0.0,  # pending ask + when
                         "hinted": False}              # hint said yet?
_CONFIRM: dict[str, Any] = {"verb": None, "at": 0.0}   # pending "say confirm"
_INTERRUPT_ANSWER = "\x00interrupt"      # sentinel: turn is being killed
# Live AUTO-APPROVE is OUR flag, not an SDK mode flip: the CLI refuses
# a live switch INTO bypassPermissions unless it was launched with the
# danger flag, so instead the gate below auto-approves silently while
# this is on. Same behavior, no reconnect, conversation intact. A
# session that BOOTS in bypassPermissions never consults the gate at
# all; saying "start asking again" flips the SDK side live (that
# direction is allowed) and turns this off. ONLY the explicit
# bypassPermissions value arms this: any other mode (acceptEdits, plan)
# passes through to the SDK and keeps the spoken gate for whatever the
# SDK routes here. (Auto-approve is about PERMISSIONS; hands-free
# LISTENING is about the microphone: see _MIC below. Two different
# axes, deliberately never sharing a name.)
_AUTOAPPROVE: dict[str, Any] = {"on": False}
# The microphone mode, switchable live by voice. "ptt" = mic closed
# except while the key is held. "open" = hands-free listening (VAD).
# The key keeps working in open mode: it interrupts, and holding it
# always gets you heard. gen bumps on every switch so an in-flight
# open-mic capture from before the switch gets discarded, never
# processed.
_MIC: dict[str, Any] = {"mode": "ptt", "gen": 0, "btn": False}

# Approvals are EXACT matches after normalization, never prefixes:
# "yesterday", "yes or no", and "yes, but do not overwrite" must all
# fail. Anything that is not an exact yes DENIES, with the words passed
# back to the agent as the reason. Deny is always the default.
# Exact matches only, and the reason is in the comment on _norm_speech:
# prefix matching turns "yesterday" and "yes or no" into consent. So the
# set has to actually CONTAIN what people say -- and the phrase somebody
# reaches for is the one the prompt just put in their head. Asking for
# PERMISSION and then denying "permission granted" is the system tripping
# a user with its own vocabulary, and it quotes their words back as the
# reason for the refusal.
_YES = {"yes", "yeah", "yep", "yup", "sure", "approve", "approved",
        "go ahead", "do it", "yes please", "yes sir", "yes boss",
        "yes go ahead", "go for it", "green light", "okay", "ok", "y",
        "permission granted", "granted", "you have permission",
        "you may", "allowed", "allow it", "confirmed", "affirmative"}
_CHAIN_MARKS = ("&&", "||", ";", "|", "$(", "`", "\n")


def _norm_speech(text):
    """Lowercase, every non-letter but digits to space, collapse. Whisper
    loves interior commas ("yes, confirm"); end-stripping alone misses
    them. Digits survive so "quiet for 20 minutes" can still be read."""
    out = []
    for ch in text.lower():
        out.append(ch if "a" <= ch <= "z" or "0" <= ch <= "9" else " ")
    return " ".join("".join(out).split())


async def _run_enroll(ears, mouth):
    """Learn the owner's voice from a few short phrases. Returns True if
    the print was saved, False otherwise. The caller has already arranged
    for the line to exit after this returns."""
    loop = asyncio.get_event_loop()
    path = CFG.get("speaker_profile") or voiceprint.default_profile()
    seconds = float(CFG.get("speaker_enroll_seconds", 3.0))
    mouth.say("I'll learn your voice from a few short phrases. Speak normally.")
    mouth.wait_done(timeout=30)
    vectors = []
    for prompt in voiceprint.ENROLL_PROMPTS:
        mouth.say(prompt)
        mouth.wait_done(timeout=30)
        pcm = await loop.run_in_executor(_BLOCKERS, ears.record_clip,
                                         seconds)
        f = voiceprint.feature(pcm) if pcm is not None else None
        if f is not None:
            vectors.append(f)
    vec = voiceprint.combine(vectors)
    if vec is None:
        mouth.say("I did not catch enough of your voice. "
                  "Let's try again when it's quieter.")
        mouth.wait_done(timeout=30)
        return False
    thr = voiceprint.self_threshold(vectors)
    voiceprint.save_profile(path, vec, threshold=thr)
    mouth.say("Enrolled. I will answer only to you now.")
    mouth.wait_done(timeout=30)
    return True


async def _run_whoami(ears, mouth):
    """Say how well the current voice matches the enrolled print, without
    arming the gate. A person can hear the score and the floor before
    trusting the gate with a room. Returns after one clip."""
    loop = asyncio.get_event_loop()
    speaker = voiceprint.load_speaker(CFG)
    if speaker is None:
        mouth.say("I have not learned your voice yet. Run with --enroll "
                  "first, then ask me again.")
        mouth.wait_done(timeout=30)
        return
    seconds = float(CFG.get("speaker_enroll_seconds", 3.0))
    mouth.say("Say a sentence and I'll tell you how well I know this voice.")
    mouth.wait_done(timeout=30)
    pcm = await loop.run_in_executor(_BLOCKERS, ears.record_clip, seconds)
    score = speaker.score(pcm)
    if score is None:
        mouth.say("I did not catch enough of your voice to judge.")
    else:
        verdict = "that's you" if score >= speaker.threshold else "not quite you"
        mouth.say(f"Your voice scored {score:.2f} against a floor of "
                  f"{speaker.threshold:.2f}, so {verdict}.")
    mouth.wait_done(timeout=30)


def _deny_pending(reason=_INTERRUPT_ANSWER):
    """Resolve a pending spoken ask as a deny. Called whenever the turn
    that posed it is being interrupted, so the ask can never outlive its
    turn and hijack a later utterance (or stall the pipe drain)."""
    f = _PERM["fut"]
    if f is not None and not f.done():
        f.set_result(reason)


def _human_what(tool, tool_input, ctx):
    """The SHORT spoken form, built for a person who has never seen a
    terminal: plain words, no paths, no syntax. Built by code, never by
    the model, so it cannot understate; and every ask offers "details",
    which reads the full literal form below. (Field case: the gate read
    whole file paths and command syntax at a brand-new user.)"""
    d = tool_input or {}
    if tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        path = str(d.get("file_path") or d.get("notebook_path")
                   or "a file").replace("\\", "/")
        name = path.rsplit("/", 1)[-1]
        in_vault = any(h and path.startswith(str(h).rstrip("/") + "/")
                       for h in (CFG.get("extra_dirs") or []))
        verb = "edit" if "Edit" in tool else "create or change"
        if in_vault and name.endswith(".md"):
            return f"{verb} a note in your vault called {name[:-3]}"
        return f"{verb} a file called {name}"
    if tool == "Bash":
        cmd = " ".join(str(d.get("command", "")).split())
        first = (cmd.split() or ["a"])[0].rsplit("/", 1)[-1]
        chained = any(m in cmd for m in _CHAIN_MARKS)
        return (f"run a {first} command in the terminal"
                + (", with several chained parts" if chained else ""))
    if tool == "WebFetch":
        url = str(d.get("url", ""))
        host = url.split("//", 1)[-1].split("/", 1)[0] or "a site"
        return f"read a web page at {host}"
    name = getattr(ctx, "display_name", None) or tool
    return f"use the {name} tool"


_DETAILS = {"details", "the details", "give me details",
            "give me the details", "what command", "what is it",
            "say more", "more", "what exactly", "the exact command"}


def _full_detail(tool, tool_input, ctx):
    """The full literal form, spoken only when the person asks for
    "details". Never lets a long command hide its tail: truncation is
    DISCLOSED and shell chaining is called out (the agent composes
    tool_input itself, so this line must not be steerable into
    understatement)."""
    d = tool_input or {}
    if tool == "Bash":
        cmd = " ".join(str(d.get("command", "")).split())
        chained = any(m in cmd for m in _CHAIN_MARKS)
        line = ("a chained command: " if chained else
                "run a command: ") + cmd[:90]
        if len(cmd) > 90:
            line += (f", and {len(cmd) - 90} more characters. "
                     "Check the log before approving")
        return line
    if tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        path = str(d.get("file_path") or d.get("notebook_path")
                   or "a file").replace("\\", "/")
        bits = path.rsplit("/", 2)
        name = "/".join(bits[-2:]) if len(bits) >= 2 else path
        return f"{'edit' if 'Edit' in tool else 'write'} the file {name}"
    if tool == "WebFetch":
        return f"fetch a web page: {str(d.get('url', ''))[:70]}"
    desc = (getattr(ctx, "description", None) or "").strip()
    name = getattr(ctx, "display_name", None) or tool
    return f"use {name}" + (f", {desc[:70]}" if desc else "")


def make_permission_gate(mouth):
    from backtalk.permresult import (PermissionResultAllow,
                                     PermissionResultDeny)

    async def gate(tool, tool_input, ctx):
        if _AUTOAPPROVE["on"]:
            return PermissionResultAllow(behavior="allow")
        what = _human_what(tool, tool_input, ctx)
        detail = _full_detail(tool, tool_input, ctx)
        loop = asyncio.get_running_loop()
        signals.static_stop()
        log(f"[perm]   asking: {what}")
        log(f"[perm]   detail: {detail}")
        if tool == "Bash":   # the FULL command always reaches the log
            log(f"[perm]   full command: {str((tool_input or {}).get('command', ''))[:2000]}")
        ask = f"Permission check. I want to {what}. Yes, no, or details?"
        if not _PERM["hinted"]:
            # the escape hatch announces itself exactly once, at the
            # moment it becomes relevant (a field case: a new user
            # couldn't find the phrase to turn the checks off)
            _PERM["hinted"] = True
            ask += (" And any time you're done with these checks, say "
                    "stop asking for permission.")
        mouth.say(ask)
        answer = None
        try:
            deadline = loop.time() + PERM_TIMEOUT_S
            while answer is None:
                fut = loop.create_future()
                _PERM["fut"] = fut
                _PERM["asked_at"] = time.monotonic()
                while True:
                    try:
                        got = await asyncio.wait_for(
                            asyncio.shield(fut), 1.0)
                        break
                    except asyncio.TimeoutError:
                        if loop.time() >= deadline:
                            fut.cancel()
                            mouth.say("No answer, so I didn't do it.")
                            log("[perm]   timed out, denied")
                            return PermissionResultDeny(
                                behavior="deny",
                                message="No spoken answer within the "
                                        "timeout; the action was not "
                                        "approved.",
                                interrupt=False)
                        # keep the ring honest while we wait
                        if not mouth.speaking:
                            signals.set_state("listening")
                if (got != _INTERRUPT_ANSWER
                        and _norm_speech(got) in _DETAILS):
                    # read the full literal form, then ask again with a
                    # fresh clock: asking for details is engagement,
                    # not silence
                    log("[perm]   details requested")
                    mouth.say(f"The details: I want to {detail}. "
                              "Yes or no?")
                    deadline = loop.time() + PERM_TIMEOUT_S
                    continue
                answer = got
        finally:
            _PERM["fut"] = None
        if answer == _INTERRUPT_ANSWER:
            log("[perm]   turn interrupted, denied silently")
            return PermissionResultDeny(
                behavior="deny",
                message="Interrupted by the user; the turn is being "
                        "cancelled.",
                interrupt=False)
        approved = _norm_speech(answer) in _YES
        # the model keeps working either way: restore the working state
        signals.set_state("thinking")
        signals.static_start()
        if approved:
            log("[perm]   approved by voice")
            return PermissionResultAllow(behavior="allow")
        log(f"[perm]   denied: {answer!r}")
        return PermissionResultDeny(
            behavior="deny",
            message=f'Denied by voice. The user said: "{answer[:500]}"',
            interrupt=False)
    return gate


# ---- THE VOICE CONSOLE: session verbs, spoken. Exact phrases only,
# spoken alone, so ordinary sentences can never trigger them. (Grown
# from a community member's own build shared in the Discord.)
CONSOLE_VERBS = {
    "clear":     ("clear the session", "clear the context",
                  "clear context", "fresh slate", "slash clear"),
    "compact":   ("compact the session", "compact the context",
                  "compact context", "slash compact"),
    "deep":      ("switch to the deep model", "use the deep model",
                  "slash model deep"),
    "fast":      ("switch to the fast model", "use the fast model",
                  "back to the fast model", "slash model fast"),
    "usage":     ("usage report", "slash usage"),
    "look":      ("look at my screen", "look at the screen",
                  "look at my display", "look at the display",
                  "what is on my screen", "what's on my screen",
                  "what is on the screen", "read my screen",
                  "describe my screen", "look at this", "look at that"),
    "micopen":   ("go hands free", "hands free mode",
                  "hands free listening", "open mic", "open the mic"),
    "micptt":    ("push to talk", "push to talk mode",
                  "back to push to talk", "back to the button"),
    "noask":     ("stop asking for permission",
                  "stop asking permission",
                  "stop asking me for permission",
                  "turn off the permission prompt",
                  "turn off the permission prompts",
                  "turn off the permissions prompt",
                  "turn off the permissions prompts",
                  "turn off permissions", "turn off permission checks",
                  "disable the permission checks",
                  "disable permission checks", "auto approve",
                  "auto approve mode"),
    "ask":       ("start asking again", "ask before acting",
                  "ask for permission again"),
}
_EFFORTS = ("low", "medium", "high", "xhigh", "max")


def console_match(text):
    norm = " ".join(text.lower().replace("-", " ").split()).strip(" .,!?")
    for verb, phrases in CONSOLE_VERBS.items():
        if norm in phrases:
            return verb
    for lvl in _EFFORTS:
        if norm in (f"set effort to {lvl}", f"effort {lvl}",
                    f"slash effort {lvl}"):
            return f"effort:{lvl}"
    return None


def _write_config_key(key, value):
    """The agent rewrites the config; the person never hand-edits it.
    Returns True on a persisted write. A file that fails to PARSE is
    left untouched (rewriting from {} would wipe every other setting);
    the in-memory CFG updates either way so the session behaves."""
    from backtalk.config import CONFIG_PATH
    CFG[key] = value
    try:
        data = json.loads(CONFIG_PATH.read_text())
    except FileNotFoundError:
        data = {}
    except (OSError, ValueError) as e:
        log(f"[console] config not writable/parsable, session-only: {e}")
        return False
    data[key] = value
    try:
        CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n")
    except OSError as e:
        log(f"[console] config write failed, session-only: {e}")
        return False
    return True


def _fmt_tokens(n):
    if n >= 1_000_000:
        return f"about {round(n / 1_000_000, 1):g} million tokens"
    if n >= 1000:
        return f"about {round(n / 1000)} thousand tokens"
    return f"{n} tokens"


def _spoken_usage(sess, ctx_usage):
    """A short CFO brief of the session, written for the ear: plain
    numerals only (the TTS reads "40" fine; symbols come out garbled)."""
    turns = sess["turns"]
    parts = [f"{turns} turn{'s' if turns != 1 else ''} this session",
             _fmt_tokens(sess["out_tokens"]) + " spoken out"]
    cents = round(sess["cost"] * 100)
    if cents >= 1:
        parts.append(f"roughly {cents} cents" if cents < 100
                     else f"roughly {round(cents / 100)} dollars")
    try:
        cats = (getattr(ctx_usage, "categories", None)
                or (ctx_usage or {}).get("categories") or [])
        # the breakdown includes "Free space" and the autocompact
        # buffer; only OCCUPIED categories belong in the spoken number
        total = sum(int(c.get("tokens") or 0) for c in cats
                    if isinstance(c, dict)
                    and "free" not in str(c.get("name", "")).lower()
                    and "buffer" not in str(c.get("name", "")).lower())
        if total:
            parts.append(_fmt_tokens(total)
                         + " sitting in the context window")
    except Exception:
        pass
    return ". ".join(parts) + "."

_PASTE_ON = "\x1b[200~"    # bracketed-paste markers (we enable the mode below)
_PASTE_OFF = "\x1b[201~"


# <<anything>> is a stage direction: lifted out, never spoken, published on
# the bus when the audio carrying it starts. Bounded so a runaway model cannot
# swallow a paragraph into one "tag".
_DIRECTION_TAG = re.compile(r"<<([^<>]{1,80})>>")


def _clean_typed(line: str) -> str:
    """Scrub terminal-copy artifacts: blockquote gutter glyphs and stray
    whitespace (copying from a CLI chat render drags bars along)."""
    line = line.strip()
    while line[:1] in ("▎", "│", ">"):
        line = line[1:].lstrip()
    return line


def _join_paste(body: str) -> str:
    """Pasted blob -> one clean message (gutters scrubbed, lines joined)."""
    parts = [_clean_typed(l) for l in body.split("\n")]
    return " ".join(" ".join(p for p in parts if p).split())


def _typed_reader_pipe(q: "queue.Queue[str]", fd: int):
    """Non-tty stdin (pipes/tests): line assembly with paste markers."""
    import os
    pend = ""
    while True:
        try:
            b = os.read(fd, 65536)
        except OSError:
            return
        if not b:
            return
        pend += b.decode("utf-8", "replace")
        while True:
            if _PASTE_ON in pend:
                if _PASTE_OFF not in pend:
                    break
                head, rest = pend.split(_PASTE_ON, 1)
                body, pend = rest.split(_PASTE_OFF, 1)
                *hlines, hpart = head.split("\n")
                for l in hlines:
                    l = _clean_typed(l)
                    if l:
                        q.put(l)
                text = _join_paste(hpart + body)
                if text:
                    q.put(text)
                continue
            if "\n" in pend:
                line, pend = pend.split("\n", 1)
                line = _clean_typed(line)
                if line:
                    q.put(line)
                continue
            break


def _typed_reader_simple(q: "queue.Queue[str]"):
    """Windows (no termios): plain line input on a thread. Pastes work;
    they just echo normally instead of collapsing to a count."""
    while True:
        try:
            line = _clean_typed(input())
        except (EOFError, OSError):
            return
        if line:
            q.put(line)


def _typed_reader(q: "queue.Queue[str]"):
    """Terminal stdin -> typed messages (daemon thread). Typed lines are
    first-class turns: same pipeline as a spoken utterance, spoken reply.

    On a POSIX tty we OWN the input line (cbreak: no kernel echo, no
    canonical buffering — the little line editor below echoes keys,
    handles backspace, and assembles bracketed pastes invisibly). The
    kernel's canonical mode is unfixable for pastes: it echoes the
    markers as visible junk and holds unfinished marker lines hostage.
    Pastes show as `[pasted N chars]`; Enter sends everything as ONE
    message. Ctrl-C still works (ISIG stays on); termios restored at
    exit."""
    import atexit
    import os
    fd = sys.stdin.fileno()
    if not os.isatty(fd):
        _typed_reader_pipe(q, fd)
        return
    try:
        import termios
        import tty as _tty
    except ImportError:            # Windows: no termios — simple reader
        _typed_reader_simple(q)
        return
    old = termios.tcgetattr(fd)              # pyright: ignore[reportAttributeAccessIssue]
    _tty.setcbreak(fd)                       # pyright: ignore[reportAttributeAccessIssue]
    sys.stdout.write("\x1b[?2004h")         # bracket pastes, please
    sys.stdout.flush()

    def _restore():
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)  # pyright: ignore[reportAttributeAccessIssue]
        except Exception:
            pass
        sys.stdout.write("\x1b[?2004l")
        sys.stdout.flush()
    atexit.register(_restore)

    MARKS = (_PASTE_ON, _PASTE_OFF)

    def _partial_tail(s: str) -> int:
        """Length of a trailing partial paste-marker (hold it for the
        next read)."""
        for m in MARKS:
            for k in range(min(len(s), len(m) - 1), 0, -1):
                if m.startswith(s[-k:]):
                    return k
        return 0

    buf = ""          # the input line being composed
    paste = None      # accumulating paste body, or None
    pend = ""
    while True:
        try:
            b = os.read(fd, 4096)
        except OSError:
            _restore()
            return
        if not b:
            _restore()
            return
        pend += b.decode("utf-8", "replace")
        keep = _partial_tail(pend)
        proc = pend[:len(pend) - keep] if keep else pend
        pend = pend[len(pend) - keep:] if keep else ""
        i = 0
        while i < len(proc):
            if paste is not None:
                j = proc.find(_PASTE_OFF, i)
                if j < 0:
                    paste += proc[i:]
                    break
                paste += proc[i:j]
                i = j + len(_PASTE_OFF)
                text = _join_paste(paste)
                paste = None
                if text:
                    if buf and not buf.endswith(" "):
                        buf += " "
                    buf += text
                    sys.stdout.write(text if len(text) <= 60
                                     else f"[pasted {len(text)} chars]")
                    sys.stdout.flush()
                continue
            if proc.startswith(_PASTE_ON, i):
                paste = ""
                i += len(_PASTE_ON)
                continue
            ch = proc[i]
            i += 1
            if ch in ("\r", "\n"):
                sys.stdout.write("\n")
                sys.stdout.flush()
                line = buf.strip()
                buf = ""
                if line:
                    q.put(line)
            elif ch in ("\x7f", "\x08"):     # backspace
                if buf:
                    buf = buf[:-1]
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
            elif ch >= " " or ch == "\t":    # printable: echo + collect
                buf += ch
                sys.stdout.write(ch)
                sys.stdout.flush()


async def speak_reply(brain: WarmBrain, mouth: Mouth, text: str,
                      on_reply=None, image: bytes | None = None):
    """First sentence ships alone (fast start); the rest go in
    2-sentence breaths — fuller chunks get livelier prosody (single
    short sentences come out flat).

    on_reply, if given, is called with each sentence as it is actually
    spoken. That is what the journal records: the spoken words, in the
    order the ears got them, not the model's raw stream."""
    t0 = time.time()
    first = True
    batch: list[str] = []
    pending: list[str] = []          # directions waiting for their chunk

    def emit(raw: str):
        nonlocal first, batch, pending
        # STAGE DIRECTIONS: your agent may write <<anything>> inline. It is
        # lifted out here, never spoken, and published on the signal bus when
        # this chunk's audio starts (signals.direction). backtalk has no
        # opinion on what a direction means; something watching the bus does.
        #
        # This used to strip only the ANGLE BRACKETS, which left the tag body
        # in the sentence and the TTS read it aloud.
        found = _DIRECTION_TAG.findall(raw)
        if found:
            pending += [d.strip() for d in found if d.strip()]
        raw = _DIRECTION_TAG.sub(" ", raw)
        # TTS hygiene: backticks and markdown fences are never speakable.
        s = " ".join(raw.replace("`", "").split()).strip()
        if not s:
            return
        if first:
            if on_reply is not None:
                # Recorded where the words are actually queued, so an
                # interrupted reply still leaves behind the part the user
                # really heard -- and once per audible chunk, never per
                # sentence, or a 2-sentence breath appears twice.
                try:
                    on_reply(s)
                except Exception:
                    pass
            log(f"[{NAME}] ({time.time()-t0:.1f}s to first) {s}"
                + (f"  <directions: {pending}>" if pending else ""))
            mouth.say_chunk(s, pending)
            signals.caption(s)
            pending = []
            first = False
        else:
            log(f"[{NAME}] {s}" + (f"  <directions: {pending}>" if pending else ""))
            batch.append(s)
            if len(batch) >= 2:
                spoken = " ".join(batch)
                if on_reply is not None:
                    try:
                        on_reply(spoken)
                    except Exception:
                        pass
                mouth.say_chunk(spoken, pending)
                # Caption what is going audible now, so a display shows the
                # reply as it is spoken rather than after it finishes.
                signals.caption(spoken)
                pending = []
                batch = []

    try:
        # A screen image only rides the first (and only) turn; passing it
        # only when present keeps every other caller's brain untouched.
        stream = (brain.ask_stream(text, image=image) if image is not None
                  else brain.ask_stream(text))
        async for sentence in stream:
            emit(sentence)
        if batch:
            tail = " ".join(batch)
            if on_reply is not None:
                try:
                    on_reply(tail)
                except Exception:
                    pass
            mouth.say_chunk(tail, pending)
            pending = []
            batch = []
            # The last partial batch is still audio the listener will hear,
            # so it gets captioned like every other one.
            signals.caption(tail)
        if first:
            # Zero sentences yielded (brain error / empty turn): nothing
            # will ever dequeue, so nothing resets the bus — park it here.
            signals.static_stop()
            signals.set_state("idle")
            signals.caption_clear()
        # NOTE: the caption is deliberately NOT cleared here. Queueing a
        # sentence is not the same as playing it: the model can finish a
        # reply seconds before the audio reaches the speakers, so clearing
        # on queue emptied the caption just as it was finally becoming
        # readable. It is cleared by reply_done(), when the audio has
        # actually drained.
    except asyncio.CancelledError:
        try:
            await brain.interrupt()
        except Exception:
            pass
        raise
    except Exception as e:
        # ask_stream raises when the PROMPT itself cannot be sent; its
        # other three failure modes (no answer, session error, empty
        # reply) yield a spoken line instead, and this block is what
        # makes the fourth behave like the rest. Without it the task
        # dies holding the exception, nothing parks the bus -- no
        # done-callback reads the task until the NEXT utterance -- and
        # the listener is left in silence with the face still showing
        # "thinking", which reads as a broken microphone rather than a
        # brain that could not be reached.
        log(f"[turn] reply failed: {type(e).__name__}: {e}")
        if batch:
            # Complete sentences already accumulated but not yet queued
            # are still things the listener was about to hear.
            tail = " ".join(batch)
            if on_reply is not None:
                try:
                    on_reply(tail)
                except Exception:
                    pass
            mouth.say_chunk(tail, pending)
            signals.caption(tail)
            pending = []
            batch = []
        if first:
            # Nothing was ever queued, so nothing will ever dequeue and
            # reset the bus: park it here, and answer out loud. Silence
            # is the one reply that cannot explain itself.
            signals.static_stop()
            signals.set_state("idle")
            signals.caption_clear()
            mouth.say("I could not reach my side of that one. "
                      "Ask me again and I will try once more.")


LOOK_PROMPT = (
    "The attached image is a screenshot of my screen, taken just now. "
    "Tell me what is on it in a few sentences, and point out anything "
    "worth noticing. If I asked about something specific, answer that "
    "about the image. If the image is unreadable or blank, say so "
    "plainly rather than guessing.")


async def speak_look(brain: WarmBrain, mouth: Mouth, on_reply=None):
    """Answer a question about the screen: one screenshot, one spoken turn.

    A capture failure is spoken, never raised -- the voice line has to
    survive a machine with no display. The bus is parked on that path
    because handle() already lit it up for a turn that now is not coming.
    """
    try:
        png = screen.capture()
    except RuntimeError as e:
        log(f"[screen] look unavailable: {e}")
        signals.static_stop()
        signals.set_state("idle")
        mouth.say("I can't see your screen on this machine.")
        return
    await speak_reply(brain, mouth, LOOK_PROMPT, on_reply=on_reply, image=png)


CHECK_IN_PROMPT = (
    "The person has gone quiet for a while. Start the conversation "
    "yourself with one or two short, warm, natural sentences in your own "
    "voice -- a friendly hello, a small thought, or an offer to help with "
    "what they were doing. Do not ask them to confirm anything, and never "
    "say that you were waiting or counting the time.")


def check_in_due(enabled, interval, idle, speaking, replying, asking,
                 paused=False, quiet=False, held=False):
    """True when the assistant should start a conversation of its own.

    A check-in is the one thing here that talks FIRST, so it is fenced on
    every side: it must be switched on, the quiet must have lasted a whole
    interval, and nothing may be in flight -- not the mouth (the person is
    hearing an answer), not a reply task, and not a permission ask waiting
    on an answer. On top of that the person can pause it, the clock can be
    in their quiet hours, and the focused app can be one that holds it.
    Kept a pure function so each fence is a plain yes/no.
    """
    return (bool(enabled) and interval > 0 and idle >= interval
            and not paused and not quiet and not held
            and not speaking and not replying and not asking)


def in_quiet_hours(hour, start, end):
    """True when a 24-hour `hour` falls inside [start, end), wrapping past
    midnight (start 23, end 7 covers late night through early morning). An
    empty or equal window is never quiet."""
    if start == end:
        return False
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def held_by_app(label, holds):
    """True when a focused-window label contains any of the hold terms,
    case-insensitively. No terms means never held."""
    low = (label or "").lower()
    return bool(low) and any(h.lower() in low for h in holds or ())


# The spoken phrases that silence check-ins for a stretch, and the ones that
# bring them back. Kept as starts-with matches so "quiet for twenty minutes"
# and "let me work for a bit" both land, without a phrase for every wording.
_QUIET_ON = (
    "stay quiet", "be quiet", "quiet for", "quiet please",
    "give me some quiet", "let me work", "leave me be",
    "leave me alone", "don't check in", "dont check in",
    "stop checking in", "no check ins", "pause check ins",
)
_QUIET_OFF = (
    "you can check in again", "you can check in now", "check in again",
    "start checking in again", "start checking in", "resume check ins",
    "come check on me again", "check on me again",
)
# Small spoken numbers, so "quiet for twenty minutes" works without digits.
_NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
}


def _quiet_seconds(norm, default):
    """Seconds named in a "quiet" phrase, else the configured default.

    Understands digits and small number words, and the two idioms people
    actually say: "an hour" and "half an hour". Any unit it cannot read
    falls back to the default, so "stay quiet" is always safe to say."""
    if "half an hour" in norm or "half hour" in norm:
        return 1800.0
    m = re.search(r"(?:for\s+)?(\d+)\s*(second|minute|min|hour|hr)s?\b", norm)
    if not m:
        # A whole word before the unit, so "remind me in an hour" matches
        # "an hour" and not the "min" hiding inside "remind".
        m = re.search(
            r"(?:for\s+)?([a-z]+)\s+(second|minute|min|hour|hr)s?\b", norm)
    if not m:
        return float(default)
    raw, unit = m.group(1), m.group(2)
    n = int(raw) if raw.isdigit() else _NUM_WORDS.get(raw)
    if n is None:
        return float(default)
    scale = 1 if unit.startswith("sec") else \
        60 if unit.startswith(("min",)) else 3600
    return float(n * scale)


def checkin_control(text, default_quiet):
    """Read a spoken check-in command. Returns ("pause", seconds) or
    ("resume", 0) or None. Kept pure so the wording is easy to test."""
    norm = _norm_speech(text)
    for p in _QUIET_OFF:
        if norm.startswith(p):
            return ("resume", 0)
    for p in _QUIET_ON:
        if norm.startswith(p):
            return ("pause", _quiet_seconds(norm, default_quiet))
    return None


def _human_span(secs):
    """Ten seconds becomes "10 seconds", ninety minutes becomes "90
    minutes", not "1.5 hours" -- the unit a person would actually say."""
    secs = int(secs)
    if secs >= 3600 and secs % 3600 == 0:
        n, unit = secs // 3600, "hour"
    elif secs >= 60:
        n, unit = round(secs / 60), "minute"
    else:
        n, unit = max(1, secs), "second"
    return f"{n} {unit}" + ("" if n == 1 else "s")


def _spoken_quiet(secs):
    """The spoken acknowledgement of a spoken pause. A non-positive span
    reads as "no check-ins" -- the phrase was heard, the length was not."""
    if int(secs) <= 0:
        return "Okay, no check-ins for now."
    return (f"Quiet for {_human_span(secs)}. I'll leave you be until then. "
            "Say you can check in again to bring me back sooner.")


# The bare words that mean "stop talking to me right now". Kept to exact
# whole-utterance matches so a sentence containing "stop" still reaches the
# brain; only a lone "stop" (or "cancel", "never mind") cuts the line.
_STOP_WORDS = (
    "stop", "stop it", "stop please", "please stop", "cancel",
    "cancel that", "never mind", "nevermind", "forget it", "shut up",
    "hush", "enough", "that's enough", "thats enough",
)


def stop_command(text):
    """True when the whole utterance is a "stop talking" word. Pure, so the
    phrasing is easy to test and to extend."""
    return _norm_speech(text) in {_norm_speech(w) for w in _STOP_WORDS}


# The spoken openings that ask for a reminder or a timer. Matched as
# starts-with after normalization, so "remind me to call mom in an hour"
# and "set a timer for 5 minutes" both land.
_REMIND_ON = (
    "remind me", "set a reminder", "set reminder", "make a reminder",
    "set a timer", "set timer", "start a timer", "start timer",
    "timer for", "wake me",
)


def reminder_request(text):
    """Read a spoken reminder. Returns (seconds, label), with seconds 0
    when no length was named, or None when this is not a reminder at all.
    Kept pure: the length reuses _quiet_seconds, so "in 20 minutes", "an
    hour", and "half an hour" all work for free."""
    norm = _norm_speech(text)
    if not any(norm.startswith(p) for p in _REMIND_ON):
        return None
    secs = _quiet_seconds(norm, 0)
    label = ""
    low = text.lower()
    i = low.rfind(" to ")
    if i != -1:
        label = text[i + 4:].strip(" .!?,")
    return (secs, label)


def _spoken_reminder(secs, label):
    """The spoken acknowledgement of a set reminder."""
    tail = f" to {label}" if label else ""
    return f"Okay, I'll remind you in {_human_span(secs)}{tail}."


def _reminder_line(label):
    """The spoken reminder itself, when its moment arrives."""
    return f"Reminder: {label}." if label else "Your timer is up."


# The board's spoken surface. A phrase reaches the glass only when it names
# the board ("... on the board") or opens with "present" -- "show me how to
# X" is a question for the model, only "show X on the board" is a command.
# The bare verbs below are actions the board takes on its own contents.
_BOARD_VERB_ACTS = {"clear": "clear", "sweep": "clear", "wipe": "clear",
                    "clean": "clear", "reset": "reset", "explode": "explode",
                    "assemble": "assemble"}
_BOARD_TAIL_WORDS = frozenset(("the", "board", "glass", "screen", "it",
                               "that", "now", "please"))
_BOARD_TAILS = (" on the board", " on the glass", " on the screen",
                " on the air board")
_BOARD_SHOW = ("put ", "show ", "present ", "throw ", "bring up ",
               "pull up ", "stick ")
# Talk ABOUT the board ("what is on the board") names the board too, so the
# head is checked against the words that open a question and left to the
# brain instead of being read as a present.
_BOARD_ASK = frozenset(("what", "what's", "which", "who", "whose", "is",
                        "are", "was", "were", "do", "does", "did", "can",
                        "could", "how", "why", "when", "where", "anything",
                        "everything", "nothing"))


def board_command(text):
    """Read a spoken board command. Returns a barehands /cmd dict or None.

    Kept pure so the wording is easy to test, the same as the other spoken
    commands. Only phrases that name the board (or open with "present") are
    claimed here, so ordinary conversation still reaches the brain."""
    low = text.lower().strip()
    words = low.split()
    if words and words[0] in _BOARD_VERB_ACTS \
            and all(w in _BOARD_TAIL_WORDS for w in words[1:]):
        return {"a": _BOARD_VERB_ACTS[words[0]]}
    tail = next((t for t in _BOARD_TAILS if low.endswith(t)), None)
    if tail:
        head = low[:-len(tail)].strip()
    elif low.startswith("present "):
        head = low[len("present "):].strip()
    else:
        return None
    head = head.strip(" .,!?")
    if not head or head.split()[0] in _BOARD_ASK:
        return None
    for verb in _BOARD_SHOW:
        if head.startswith(verb):
            rest = head[len(verb):].strip()
            if rest.startswith("me "):
                rest = rest[3:].strip()
            rest = rest or head
            return {"a": "present", "title": rest}
    return {"a": "present", "title": head}


def _board_ack(cmd, ok):
    """The spoken answer to a board command. Pure, so the wording is easy to
    test."""
    if not ok:
        return "I couldn't reach the board."
    return {
        "clear": "Board's clear.",
        "reset": "Board reset.",
        "explode": "Blown apart.",
        "assemble": "Back together.",
    }.get(cmd.get("a"), "It's up on the board.")


# The spoken openings that ask the agent to search its own journal. Matched
# as starts-with after normalization, so the topic is whatever follows.
_RECALL_ON = (
    "what do you remember about", "what do you know about",
    "when did we talk about", "when did we last talk about",
    "when did i talk about", "what did we say about",
    "what did i say about", "do you remember", "recall",
    "search your journal for", "search the journal for",
)


def recall_request(text):
    """Read a spoken journal-recall request. Returns the topic to search for,
    or None when this is not a recall question. Pure, so the wording is easy
    to test."""
    norm = _norm_speech(text)
    for p in _RECALL_ON:
        if norm.startswith(p):
            return norm[len(p):].strip() or None
    return None


# The spoken openings that tell the line to KEEP a line. Matched as
# starts-with after normalization. There is deliberately no bare "remember"
# -- "remember the time we..." is talk, not a save -- so the opener must be
# one that means "write this down". The words AFTER the opener are kept
# verbatim (case and punctuation intact) because they land in a note the
# person reads back.
_REMEMBER_ON = (
    "remember that", "remember this", "make a note of", "make a note",
    "note down", "note this", "note that", "take a note", "take this down",
    "write this down", "write that down", "jot this down", "jot that down",
    "add to my notes", "add a note", "save this", "save that",
)


def remember_request(text):
    """Read a spoken "remember this" command. Returns the words to keep (the
    original text after the opener, trimmed of the separator), or None when
    this is not one. Pure, so the wording is easy to test."""
    body = text.strip()
    low = body.lower()
    for p in _REMEMBER_ON:
        if low.startswith(p):
            return body[len(p):].strip(" .,:;!?-\t") or None
    return None


def remember_note(directory, text):
    """Append one kept line to a per-day markdown file in `directory`,
    returning the path written. Raises OSError if the folder cannot be
    written; the caller decides what to say. Side-effect only: backtalk
    never reads these back, the person's vault does."""
    path = Path(directory) / f"Remembered {time.strftime('%Y-%m-%d')}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"- {time.strftime('%H:%M')} — {text}\n")
    return str(path)


# The spoken questions that ask to HEAR BACK what was kept -- the counterpart
# to "remember this". Read from the same Remembered notes, newest first.
_REMEMBERED_ON = (
    "what did i ask you to remember", "what did i tell you to remember",
    "what have i asked you to remember", "what did i ask you to note",
    "read me what i asked you to remember", "read back what i told you",
    "read me my notes", "read me the notes", "read me my remembered notes",
    "what do you have in my notes", "what s in my notes",
    "whats in my notes", "what s in the inbox",
)


def remembered_request(text):
    """True when the person is asking to hear back what was kept. Pure."""
    norm = _norm_speech(text)
    return any(norm == p or norm.startswith(p + " ") for p in _REMEMBERED_ON)


def read_remembered(directory, limit=3):
    """The last few kept lines, newest first, from the "Remembered" notes in
    `directory`. [] when there is nothing to read. Never raises: a folder
    that is missing, or a note that will not open, is simply skipped."""
    if not directory:
        return []
    lines = []
    for path in sorted(Path(directory).glob("Remembered *.md"), reverse=True):
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
        except OSError:
            continue
        for raw in reversed(content.splitlines()):
            body = raw.strip().lstrip("-").strip()
            if not body:
                continue
            if " — " in body:
                body = body.split(" — ", 1)[1].strip()
            lines.append(body)
            if len(lines) >= limit:
                return lines
    return lines


def spoken_remembered(items):
    """One spoken line for what the person asked to be remembered. Pure."""
    if not items:
        return "You haven't asked me to remember anything yet."
    if len(items) == 1:
        return "You asked me to remember this: " + items[0] + "."
    joined = "; ".join(items[:-1]) + ", and " + items[-1]
    return "Here's what you asked me to remember: " + joined + "."


# The spoken openings that start a focus stretch. The length reuses
# _quiet_seconds, so "focus for twenty five minutes" and "focus for half an
# hour" both work; naming no length uses the configured default. A bare
# "focus" is left out on purpose: "focus the window" is a screen command.
_FOCUS_ON = (
    "focus for", "focus mode", "start focus", "start a focus", "start a "
    "pomodoro", "focus time", "deep work for", "deep work mode",
    "pomodoro", "do not disturb for", "do not disturb",
    "dont disturb for", "dont disturb", "quiet mode for", "quiet mode",
)


def focus_request(text, default_minutes):
    """Read a spoken focus request. Returns whole minutes (at least one) for
    the stretch, or None when this is not one. Pure."""
    norm = _norm_speech(text)
    if not any(norm.startswith(p) for p in _FOCUS_ON):
        return None
    secs = _quiet_seconds(norm, (default_minutes or 25) * 60)
    return max(1, int(round(secs / 60)))


# Fed to the brain when the person asks for a briefing, and at launch by an
# automatic one. The brain already holds the notes; this only frames the
# shape of the answer -- short, spoken, drawn from what it can actually see.
_BRIEFING_PROMPT = (
    "Give the person a short spoken briefing drawn only from what you can "
    "actually see in their notes and this conversation: what is on their "
    "plate today, and what they last worked on. Two or three short, warm "
    "sentences in your own voice. Never read out a file path or markdown, "
    "and never invent anything you cannot see.")
_BRIEFING_ON = (
    "brief me", "give me a briefing", "give me my briefing",
    "what s on my plate", "whats on my plate", "what is on my plate",
    "my daily briefing", "daily briefing", "morning briefing",
    "catch me up",
)


def briefing_request(text):
    """True when the person is asking for a daily briefing. Pure."""
    norm = _norm_speech(text)
    return any(norm == p or norm.startswith(p + " ")
               for p in _BRIEFING_ON)


def briefing_due(last_day, today):
    """True when a briefing has not yet been given today. Pure."""
    return last_day != today


def briefing_state_path(directory):
    """(file, today) for the once-a-day briefing marker. Pure."""
    return str(Path(directory) / ".briefing_day"), time.strftime("%Y-%m-%d")


def read_briefing_day(path):
    """The last day a briefing was given, or "" when none or unreadable."""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def write_briefing_day(path, day):
    """Remember the day of the last briefing. Best-effort: a box where the
    marker cannot be written still gets its briefing."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(day)
    except OSError:
        log(f"[briefing] could not remember the day in {path}")


async def journal_flush(journal: Journal, brain: WarmBrain):
    """Write the session down, then get out of the way.

    Two rules, both learned the hard way by everything else in this file:
    a hangup must never hang, and a hangup must never crash. The summary is
    asked for on its OWN brain session so it cannot pollute the
    conversation it is summarising -- the real brain has this whole exchange
    in it, and adding bookkeeping to it would be visible to the next turn.
    """
    if not journal.active or not journal.events:
        return
    try:
        summary = ""
        if CFG.get("journal_summary"):
            # stop() on the scratch brain releases its session handle; it
            # never touches the shared server, which the real brain still
            # owns until a few lines below.
            scratch = WarmBrain(model=None, can_use_tool=lambda *a, **k: None)
            try:
                await scratch.start()
                summary = await journal_summarize(
                    scratch.ask_stream, journal.transcript())
            finally:
                try:
                    await scratch.stop()
                except Exception:
                    pass
        path = journal.write(summary)
        if path:
            log(f"[journal] wrote {path.name}")
        elif summary:
            log("[journal] could not write the entry (see above)")
    except Exception as e:
        # Journalling is a courtesy to the future reader. It is never worth
        # failing a hangup over.
        log(f"[journal] skipped: {type(e).__name__}: {e}")


async def amain():
    open_mic = "--open-mic" in sys.argv
    barge_in = "--barge-in" in sys.argv or CFG.get("barge_in")
    wake_word = "--wake-word" in sys.argv or CFG.get("wake_word")
    wake_phrases = [str(p).lower()
                    for p in (CFG.get("wake_word_phrases") or [])]
    if wake_word and not wake_phrases:
        # wake word on but no phrases configured: fall back to a plain
        # open mic rather than a listener that can never wake.
        log("[backtalk] wake_word is on but wake_word_phrases is empty; "
            "ignoring wake word")
        wake_word = False
    # Partial transcription: while you talk, growing transcripts go to the
    # bus so a face can caption the words live. None unless asked for, so
    # nothing pays the extra whisper work by default.
    emit_partial = signals.heard if CFG.get("partial_stt") else None
    model = None
    if "--model" in sys.argv:
        try:
            model = sys.argv[sys.argv.index("--model") + 1]
        except IndexError:
            pass

    CFG_BOOT_MODE = CFG["permission_mode"]
    _AUTOAPPROVE["on"] = CFG_BOOT_MODE == "bypassPermissions"
    # A wake word is only reachable in open mode; asking for one implies
    # hands-free, or the setting would silently do nothing in the default
    # PTT mode.
    _MIC["mode"] = "open" if (open_mic
                              or CFG.get("mic_mode") == "open"
                              or wake_word) else "ptt"
    # resume_last_session: reattach to the saved conversation, if any
    resume_id = None
    if CFG.get("resume_last_session"):
        try:
            from backtalk.brain import SESSION_FILE
            with open(SESSION_FILE) as f:
                resume_id = f.read().strip() or None
        except OSError:
            resume_id = None

    mouth = Mouth()
    ears = Ears()
    # OWNER GATE: a small voiceprint to stop the room waking the agent.
    speaker = voiceprint.load_speaker(CFG)
    owner_check = (speaker.check
                   if speaker is not None and CFG.get("speaker_id") else None)
    if CFG.get("speaker_id") and speaker is None:
        log("[backtalk] speaker_id is on but no voiceprint is enrolled; "
            "run with --enroll first. Not gating.")

    if "--enroll" in sys.argv:
        # Records a few short phrases to learn your voice, then exits.
        await _run_enroll(ears, mouth)
        return
    if "--whoami" in sys.argv:
        # Scores one clip against the print without arming the gate, then exits.
        await _run_whoami(ears, mouth)
        return
    brain = WarmBrain(model=model,
                       can_use_tool=make_permission_gate(mouth),
                       resume_id=resume_id)
    journal = Journal(CFG.get("journal_dir") or None, agent_name=NAME)
    mode = ("hands-free listening (the talk key still works)"
            if _MIC["mode"] == "open"
            else f"push-to-talk ({CFG['ptt_key']})")
    if wake_word and _MIC["mode"] == "open":
        mode += f", wake word {wake_phrases[0]!r}"
    log(f"[backtalk] up - agent={NAME} dir={CFG['agent_dir']} "
        f"model={brain.model} mic={mode} "
        f"(say 'goodbye {NAME.lower()}' to hang up)")
    mouth.say(CFG["greeting"])

    loop = asyncio.get_event_loop()
    # Warm the engines while the greeting plays: the STT model load and
    # the brain's prompt-cache toll both hide behind the spoken line.
    loop.run_in_executor(_BLOCKERS, warm_ears)
    # THE BRAIN CONNECT, guarded. This is the one startup step that
    # needs a working opencode install, a configured provider, the
    # internet, and available quota.
    # When it fails or hangs, the mouth still works, so SAY SO instead
    # of dying silently with the face stuck on idle (a real field
    # case: the greeting played, then nothing, and on Windows the
    # window closed before anyone could read the error).
    log("[backtalk] connecting the brain...")
    try:
        await asyncio.wait_for(brain.start(), 120)

        async def _warmup():
            async for _ in brain.ask_stream(
                    "Warmup ping - reply with the single word: ready"):
                pass
        await asyncio.wait_for(_warmup(), 180)
    except (Exception, asyncio.TimeoutError) as e:
        kind = ("timed out" if isinstance(e, asyncio.TimeoutError)
                else f"failed: {e!r}"[:220])
        log(f"[backtalk] BRAIN CONNECT {kind}")
        mouth.say("Bad news. The voice and the face are fine, but I "
                  "couldn't reach my brain, the opencode session. "
                  "Check this window for the error. The usual causes: "
                  "opencode isn't on the path, no provider is signed in, "
                  "the internet is down, or the provider is out of quota.")
        mouth.wait_done(timeout=30)
        raise SystemExit(1)
    log("[backtalk] brain warm")
    # the hidden warmup ping is plumbing, not conversation
    brain.session.update(turns=0, out_tokens=0, in_tokens=0, cost=0.0)
    # a configured effort level applies at launch (saved by the spoken
    # "set effort to X", or written by the person's agent on request)
    boot_effort = str(CFG.get("effort") or "").strip().lower()
    if boot_effort in _EFFORTS:
        await brain.command(f"/effort {boot_effort}")
        log(f"[backtalk] effort set to {boot_effort} (from config)")
    elif boot_effort:
        log(f"[backtalk] ignoring unknown effort {boot_effort!r} in config")

    speak_task: asyncio.Task | None = None
    typed_q: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=_typed_reader, args=(typed_q,), daemon=True).start()
    typed_fut: asyncio.Future | None = None

    # The activity tracker: a quiet, local note of which window is
    # focused, sampled every screen_track_interval and recorded in the
    # session journal. Off unless screen_enabled; an interval of 0 (or
    # less) means "no periodic sampling", so the spoken look still works
    # while nothing is watched in the background.
    track_interval = float(CFG.get("screen_track_interval") or 0)
    _LAST_ACTIVITY = {"label": ""}
    tracker: asyncio.Task | None = None
    if CFG.get("screen_enabled") and track_interval > 0:
        def _note_activity(label):
            _LAST_ACTIVITY["label"] = label
            journal.say_activity(label)
            signals.active_window(label)

        async def _track_activity():
            await screen.watch(track_interval, _note_activity)

        tracker = asyncio.create_task(_track_activity())

    # Check-in cadence. The assistant may start the conversation itself
    # after a full quiet interval, but only when switched on; 0 (or less)
    # disables it even then, and off is the shipped default. _ACTIVITY is
    # the shared quiet-clock: every real turn moves it, and the main wait
    # below measures from it. _CHECKIN carries the live state a spoken
    # "stay quiet" can move, and _LAST_ACTIVITY remembers what the person
    # was last seen doing, so a check-in can be specific instead of generic.
    check_interval = float(CFG.get("check_in_interval", 0) or 0)
    if not CFG.get("check_in_enabled"):
        check_interval = 0.0
    check_jitter = max(0.0, float(CFG.get("check_in_jitter", 0) or 0))
    quiet_hours = CFG.get("check_in_quiet_hours") or []
    hold_apps = list(CFG.get("check_in_hold_apps") or [])
    quiet_default = float(CFG.get("check_in_default_quiet", 3600) or 3600)
    followup_window = max(0.0, float(CFG.get("followup_window", 0) or 0))
    care_enabled = bool(CFG.get("care_enabled"))
    _ACTIVITY = {"at": time.monotonic()}
    _CHECKIN = {"paused_until": 0.0}
    _FOLLOWUP = {"until": 0.0}
    # When the last self-care nudge was spoken (monotonic), so a hello never
    # turns into a health report.
    _CARE = {"at": 0.0}
    # Pending spoken reminders, each {"at": monotonic-deadline, "label": str}.
    # The main wait caps itself at the nearest one so the loop wakes to
    # speak it; nothing else has to poll.
    _REMINDERS: list[dict[str, Any]] = []

    def _check_in_prompt():
        """The check-in prompt, with the last thing they were doing folded
        in when the activity tracker has seen something, and one self-care
        nudge when care is on and one is due. Kept short: the brain has the
        conversation either way, this is just a nudge."""
        prompt = CHECK_IN_PROMPT
        label = _LAST_ACTIVITY["label"]
        if label:
            prompt += (" For context, the last window you noticed them in "
                       "was " + label + ".")
        if care_enabled:
            line, _CARE["at"] = care.current_nudge(last=_CARE["at"])
            if line:
                prompt += (" If it fits naturally, you might also mention "
                           "that " + line + ".")
        return prompt

    def _reply_task(coro):
        """Start the reply task and, once it is done, open the follow-up
        window so a reply can be answered without the wake word."""
        nonlocal speak_task
        speak_task = asyncio.create_task(coro)
        if followup_window:
            speak_task.add_done_callback(_open_followup)

    def _open_followup(_task):
        # Fires whether the reply finished or was interrupted; either way
        # the person is likely mid-thought and the next thing they say
        # should not need the wake word.
        _FOLLOWUP["until"] = time.monotonic() + followup_window

    def _checkin_quiet():
        """True during the configured quiet hours, by the wall clock."""
        if len(quiet_hours) != 2:
            return False
        return in_quiet_hours(time.localtime().tm_hour,
                              int(quiet_hours[0]), int(quiet_hours[1]))

    def _held_now():
        """True when the focused window matches a hold term, so the line
        stays quiet through a meeting or a game."""
        if not hold_apps:
            return False
        app, title = screen.active_window()
        return held_by_app(" ".join(p for p in (app, title) if p), hold_apps)

    async def run_console(verb):
        """One voice-console verb. The current reply was already
        cancelled and awaited by handle(); the pipe gets drained here
        before the command goes out. A verb that blows up must never
        take the whole voice session down with it."""
        try:
            await _run_console_inner(verb)
        except Exception as e:
            log(f"[console] {verb} failed: {e}")
            mouth.say("That command hit an error. Check the log.")
            signals.set_state("idle")

    async def _run_console_inner(verb):
        _deny_pending()
        await brain.reset_turn()
        say_after = None
        if verb == "clear":
            resp = await brain.command("/clear")
            say_after = "Cleared. Fresh slate."
        elif verb == "compact":
            mouth.say("Compacting. One moment.")
            resp = await brain.command("/compact")
            say_after = "Compacted. Same conversation, smaller footprint."
        elif verb == "deep":
            mouth.say("Switching to the deep model. Heads up, replies "
                      "get slower. Say back to the fast model when "
                      "you're done.")
            resp = await brain.command(f"/model {CFG['deep_model']}")
            say_after = "Deep model online, for this session only."
        elif verb == "fast":
            resp = await brain.command(f"/model {CFG['model']}")
            say_after = "Back on the fast model."
        elif verb.startswith("effort:"):
            lvl = verb.split(":", 1)[1]
            resp = await brain.command(f"/effort {lvl}")
            saved = _write_config_key("effort", lvl)
            say_after = (f"Effort set to {lvl}, and saved as your "
                         "default." if saved else
                         f"Effort set to {lvl} for this session. The "
                         "config file couldn't be written, so it won't "
                         "stick past a restart.")
        elif verb == "usage":
            resp = ""
            mouth.say(_spoken_usage(brain.session,
                                    await brain.context_usage()))
        elif verb == "micopen":
            resp = ""
            if _MIC["mode"] == "open":
                mouth.say("Already in hands-free listening.")
            else:
                _MIC["mode"] = "open"
                _MIC["gen"] += 1
                _write_config_key("mic_mode", "open")
                log("[console] mic_mode -> open (hands-free listening)")
                mouth.say("Hands-free listening on. I'm always "
                          "listening now, so anything said in the room "
                          "can reach me. The talk key still works, and "
                          "holding it always gets you heard. Say push "
                          "to talk mode to bring the button back.")
        elif verb == "micptt":
            resp = ""
            if _MIC["mode"] == "ptt":
                mouth.say("Already on push to talk.")
            else:
                _MIC["mode"] = "ptt"
                _MIC["gen"] += 1
                _write_config_key("mic_mode", "ptt")
                log("[console] mic_mode -> ptt")
                key = str(CFG.get("ptt_key", "home")).replace("_", " ")
                mouth.say(f"Push to talk. Hold the {key} key and "
                          "talk; the mic stays closed otherwise.")
        elif verb == "noask":
            resp = ""
            _CONFIRM["verb"] = "noask"
            _CONFIRM["at"] = time.monotonic()
            mouth.say("Auto-approve means I act without asking "
                      "permission, and it becomes your saved default. "
                      "Say confirm to switch.")
        elif verb == "noask:confirmed":
            resp = ""
            saved = _write_config_key("permission_mode",
                                      "bypassPermissions")
            _AUTOAPPROVE["on"] = True
            log("[console] permission_mode -> bypassPermissions"
                + (" (saved)" if saved else " (session only)"))
            mouth.say(("Auto-approve on, and saved as your default. "
                       if saved else
                       "Auto-approve on for this session. The config "
                       "file couldn't be written, so it won't stick "
                       "past a restart. ")
                      + "Say start asking again any time to flip it "
                        "back.")
        elif verb == "ask":
            resp = ""
            saved = _write_config_key("permission_mode", "ask")
            _AUTOAPPROVE["on"] = False
            flipped = True
            if CFG_BOOT_MODE == "bypassPermissions":
                # a bypass-booted session never consults the gate, so
                # the SDK itself must flip (the safe direction is
                # allowed live). If that fails, saying "done" would be
                # a lie: the agent would keep acting silently.
                try:
                    await brain.set_permission_mode("ask")
                except Exception as e:
                    flipped = False
                    log(f"[console] live flip to ask FAILED: {e}")
            log("[console] permission_mode -> ask"
                + (" (saved)" if saved else " (session only)"))
            if flipped:
                mouth.say("Done. I'll ask out loud before real "
                          "actions"
                          + (", and that's saved as your default."
                             if saved else
                             ". The config file couldn't be written, "
                             "so tell me again after a restart."))
            else:
                mouth.say("I saved asking as your default, but this "
                          "session couldn't switch over. Restart the "
                          "voice line to get asking back.")
        else:  # pragma: no cover -- every console verb is handled above
            resp = ""
        if say_after:
            # the CLI answers slash commands with its own text
            # (confirmations, API errors); an error outranks our line
            low = (resp or "").lower()
            if resp and ("error" in low or "invalid" in low):
                mouth.say(resp[:160])
                log(f"[console] {verb} answered: {resp[:120]}")
            else:
                mouth.say(say_after)
        signals.set_state("idle")

    async def handle(text: str, spoke_from: float | None = None) -> bool:
        """Process one utterance; returns False on quit. spoke_from is
        when the utterance STARTED (the PTT press), so an answer can be
        told apart from speech that began before the ask even existed."""
        nonlocal speak_task
        _ACTIVITY["at"] = time.monotonic()
        log(f"[you]    {text}")
        # The turn is taken over: drop the partial-transcript line that was
        # building during the capture, so the next utterance starts a fresh
        # hearing line instead of leaving a stale one on the bus.
        if emit_partial is not None:
            signals.heard_clear()
        # ENGLISH ONLY. The ear keeps hearing whatever the room says --
        # Tamil, Hindi, anything -- but this assistant answers in
        # English every time: one language, one voice. The prompt says
        # the same; this is the belt to its braces, because a reply in
        # any other language reaching the speaker is a failure even
        # when the content is right.
        spoken = "en"
        set_turn_language(spoken)
        signals.language(spoken)
        # A pending spoken permission ask owns the next utterance IF
        # that utterance started after the ask was posed. Speech that
        # began earlier is the user interrupting the turn, not
        # answering a question they never heard: the ask resolves as a
        # silent deny and the utterance falls through as a normal
        # interrupt. Quit wins either way, but only as an EXACT phrase
        # here ("No! Don't hang up, skip it" must stay a deny reason,
        # not kill the session).
        if _PERM["fut"] is not None and not _PERM["fut"].done():
            started_after = (spoke_from is None
                             or spoke_from >= _PERM["asked_at"])
            if _norm_speech(text) in {_norm_speech(q)
                                      for q in QUIT_PHRASES}:
                _PERM["fut"].set_result("no")
                # falls through to the quit body below
            elif started_after:
                _PERM["fut"].set_result(text)
                return True
            else:
                _deny_pending()  # pragma: no cover -- a held key never starts before the ask
        # A pending auto-approve confirm owns it too, for two minutes;
        # after that it expires and speech flows normally again.
        verb = None
        if _CONFIRM["verb"]:
            pend, _CONFIRM["verb"] = _CONFIRM["verb"], None
            expired = time.monotonic() - _CONFIRM["at"] > 120
            if not expired and _norm_speech(text) in (
                    "confirm", "confirmed", "yes confirm",
                    "yes confirmed"):
                verb = pend + ":confirmed"
            elif not expired and not any(q in text.lower()
                                         for q in QUIT_PHRASES):
                mouth.say("Staying as we are.")
                return True
        if _norm_speech(text) in {_norm_speech(q) for q in QUIT_PHRASES}:
            if speak_task and not speak_task.done():
                speak_task.cancel()
            mouth.shut_up()
            mouth.say(CFG["signoff"])
            mouth.wait_done(timeout=15)
            return False
        # A bare "stop" / "cancel" / "never mind" cuts any reply and goes
        # no further -- it is an instruction to the LINE, not a question
        # for the brain. Only a lone stop word matches; a sentence that
        # happens to contain one still reaches the model.
        stopping = stop_command(text)
        was_speaking = bool(speak_task and not speak_task.done()) \
            or mouth.speaking
        if speak_task and not speak_task.done():
            log("[turn] interrupted mid-reply by new input")
            _deny_pending()          # an ask never outlives its turn
            speak_task.cancel()
            mouth.shut_up()
        if speak_task:
            # Let the cancellation fully land (its brain.interrupt()
            # included) BEFORE anything else touches the brain —
            # otherwise the dead turn's stop signal can race in after
            # the new query and kill the new answer (half of the
            # off-by-one bug; see brain.reset_turn for the other half).
            try:
                await speak_task
            except asyncio.CancelledError:
                pass
            except Exception:  # pragma: no cover -- speak_reply swallows its own
                pass
            speak_task = None
        if stopping:
            if was_speaking:
                mouth.say("Okay.")
            signals.set_state("idle")
            return True
        # A spoken reminder / timer is scheduled here, off the conversation.
        rem = reminder_request(text)
        if rem is not None:
            secs, label = rem
            if secs <= 0:
                mouth.say("Sure. How long from now should I remind you?")
            else:
                _REMINDERS.append({"at": time.monotonic() + secs,
                                   "label": label})
                mouth.say(_spoken_reminder(secs, label))
            log("[reminder] set")
            signals.set_state("idle")
            return True
        # A spoken "stay quiet" / "you can check in again" is handled here,
        # not by the brain: it is a setting for the voice LINE, and routing
        # it through the model would risk it being "answered" instead of
        # obeyed. It also mutes any reply still playing, like an interrupt.
        ctl = checkin_control(text, quiet_default)
        if ctl is not None:
            kind, secs = ctl
            if kind == "pause":
                _CHECKIN["paused_until"] = time.monotonic() + secs
                mouth.say(_spoken_quiet(secs))
            else:
                _CHECKIN["paused_until"] = 0.0
                mouth.say("Check-ins are back on. I'll look in again "
                          "when it's been quiet.")
            log(f"[check-in] spoken {kind}")
            signals.set_state("idle")
            return True
        # A spoken board command ("put the plan on the board", "clear the
        # board") is sent straight to the barehands server and answered
        # here. It never reaches the brain: it is an instruction to the
        # glass, not a question. Only claimed when a board is configured,
        # so a box with no board treats these as ordinary talk.
        bc = board_command(text) if board.enabled() else None
        if bc is not None:
            ok = await loop.run_in_executor(_BLOCKERS, board.send, bc)
            log(f"[board] {bc.get('a')} -> "
                f"{'ok' if ok else 'unreachable'}")
            mouth.say(_board_ack(bc, ok))
            signals.set_state("idle")
            return True
        # A spoken recall question ("what do you remember about X") searches
        # the session journal here and answers aloud. Only claimed when a
        # journal is configured; with none, it is an ordinary question.
        rq = recall_request(text) if CFG["journal_dir"] else None
        if rq:
            line = await loop.run_in_executor(
                _BLOCKERS, recall.spoken, rq, CFG["journal_dir"])
            log("[recall] answered from the journal")
            mouth.say(line)
            signals.set_state("idle")
            return True
        # A spoken "remember this ..." is kept in the notes folder and
        # acknowledged here; it never reaches the brain, which would only
        # spend a round trip saying "noted". Only claimed when a notes
        # folder is configured, so a box without one treats it as talk.
        rmb = remember_request(text) if CFG.get("remember_dir") else None
        if rmb:
            try:
                await loop.run_in_executor(
                    _BLOCKERS, remember_note, CFG["remember_dir"], rmb)
                log("[remember] kept a line")
                mouth.say("Got it. I've written that down.")
            except OSError:
                log("[remember] could not write the note")
                mouth.say("I couldn't write that down. Check the notes "
                          "folder.")
            signals.set_state("idle")
            return True
        # A spoken "what did I ask you to remember?" reads the kept lines
        # back, newest first. Only claimed when a notes folder is set, and
        # never sent to the brain: the notes are right here on disk.
        if CFG.get("remember_dir") and remembered_request(text):
            items = await loop.run_in_executor(
                _BLOCKERS, read_remembered, CFG["remember_dir"])
            log(f"[remember] read back {len(items)}")
            mouth.say(spoken_remembered(items))
            signals.set_state("idle")
            return True
        # A spoken focus stretch ("focus for twenty minutes", "start a
        # pomodoro") quiets the line and leaves one reminder for the break,
        # reusing the check-in pause and the reminder queue rather than a
        # clock of its own. "you can check in again" or a bare "stop" ends
        # it early.
        fq = focus_request(text, CFG.get("focus_minutes", 25))
        if fq:
            until = time.monotonic() + fq * 60
            _CHECKIN["paused_until"] = until
            _REMINDERS.append({"at": until, "label": "break time"})
            log(f"[focus] quiet for {fq} min")
            mouth.say("Okay, focusing for " + _human_span(fq * 60) +
                      ". I'll keep quiet until it's time for a break.")
            signals.set_state("idle")
            return True
        # A spoken "brief me" goes to the brain framed as a briefing, so the
        # answer is drawn from the notes the brain already holds rather than
        # from a template here.
        if briefing_request(text):
            log("[briefing] asked")
            signals.set_state("thinking")
            signals.static_start()
            _deny_pending()
            await brain.reset_turn()
            _reply_task(
                speak_reply(brain, mouth, _BRIEFING_PROMPT,
                            on_reply=journal.say_agent))
            return True
        verb = verb or console_match(text)
        if verb and verb != "look":
            await run_console(verb)
            return True
        signals.set_state("thinking")
        signals.static_start()
        # Clean the pipe: drain the interrupted turn's leftovers so the
        # new question can't pair with a stale ResultMessage. A gate
        # that fired in the meantime resolves first, or the drain would
        # wait on a ResultMessage the CLI is withholding for an answer.
        _deny_pending()
        await brain.reset_turn()
        # Only turns that actually reach the brain are recorded: a permission
        # answer and a console command are machinery, not conversation, and a
        # journal full of "/status" is a journal nobody reads.
        journal.say_user(text)
        # "look" is a real turn, not a console verb: it has to reach the
        # brain, with a screenshot attached, and come back as an answer.
        if verb == "look":
            _reply_task(
                speak_look(brain, mouth, on_reply=journal.say_agent))
        else:
            _reply_task(
                speak_reply(brain, mouth, text, on_reply=journal.say_agent))
        return True

    async def _do_check_in():
        """Speak one line the assistant starts itself. The bus is lit and
        the brain's turn reset exactly as a normal turn's would be, but no
        user utterance is recorded -- nothing was said to it."""
        log("[check-in] speaking first after a quiet spell")
        signals.set_state("thinking")
        signals.static_start()
        _deny_pending()
        await brain.reset_turn()
        _reply_task(
            speak_reply(brain, mouth, _check_in_prompt(),
                        on_reply=journal.say_agent))

    def _do_reminder(due):
        """Speak any reminders whose moment has passed. A plain spoken line,
        not a brain turn: a reminder must land instantly and must not depend
        on the model being reachable."""
        for r in due:
            mouth.say(_reminder_line(r["label"]))
        signals.set_state("idle")
        log(f"[reminder] spoke {len(due)}")

    # An automatic briefing, once per day, on the first launch of the day,
    # when the person has switched it on. Off by default: a session that
    # launches straight into a speech is a surprise. The spoken "brief me"
    # works either way.
    if CFG.get("daily_briefing"):
        bpath, today = briefing_state_path(CFG.get("signals_dir") or ".")
        if briefing_due(read_briefing_day(bpath), today):
            write_briefing_day(bpath, today)
            log("[briefing] first launch today")
            _deny_pending()
            await brain.reset_turn()
            _reply_task(
                speak_reply(brain, mouth, _BRIEFING_PROMPT,
                            on_reply=journal.say_agent))

    try:
        # ONE loop, two mic modes, switchable live (_MIC). The talk key
        # is constructed and honored in BOTH modes: in hands-free
        # listening it is the interrupt and the guaranteed way to be
        # heard over room noise. The open mic joins the wait-set only
        # in "open" mode; a mode switch bumps _MIC["gen"], the abort
        # callable closes the in-flight open mic promptly, and any
        # capture born under an old gen is discarded unprocessed.
        try:
            ptt = PTTListener(CFG["ptt_key"])
        except RuntimeError as e:
            # No global key hook (headless box, or CI). Hold-to-talk is
            # out; fall back to the open mic so the voice line still runs.
            log(f"[ptt] {e} — listening with the open mic instead")
            ptt = None
            _MIC["mode"] = "open"
        press_fut: asyncio.Future | None = None
        mic_fut: asyncio.Future | None = None
        mic_gen_seen = _MIC["gen"]
        # The open mic yields while the BUTTON records (or the double
        # capture would turn one held utterance into two turns), and,
        # without barge-in, while the mouth speaks.
        mic_gate = (lambda: _MIC["btn"]
                    or (not barge_in and mouth.speaking))

        def capture_open(g):
            """One open-mic turn. With wake_word on, ignore everything
            until a wake phrase is heard; if the phrase carried the
            command use it, otherwise listen once more for the command.
            wake_word only applies in open mode; PTT is always direct.
            The follow-up window is the exception: for a few seconds after
            a reply, speech is taken as the next thing said, no wake word,
            so answering is a conversation rather than a summoning."""
            ab = lambda: _MIC["gen"] != g
            if not (wake_word and _MIC["mode"] == "open"):
                return (g, ears.listen_once(gate=mic_gate, abort=ab,
                                            on_partial=emit_partial))
            if time.monotonic() < _FOLLOWUP["until"]:
                return (g, ears.listen_once(gate=mic_gate, abort=ab,
                                            on_partial=emit_partial))
            woke = ears.wait_for_wake(wake_phrases, gate=mic_gate, abort=ab,
                                      on_partial=emit_partial,
                                      owner_check=owner_check)
            if woke is None:
                return (g, None)
            cmd = strip_wake(woke, wake_phrases)
            if cmd:
                return (g, cmd)          # "hey seyon, what's the weather"
            return (g, ears.listen_once(gate=mic_gate, abort=ab,
                                        on_partial=emit_partial))

        mic_fails = 0
        while True:
            if _MIC["gen"] != mic_gen_seen:
                mic_gen_seen = _MIC["gen"]
                # consume futures that completed under the old mode so
                # a stale press or capture can't fire after a switch.
                # .exception() observes a failed capture without raising:
                # a mic that errored as its mode was pulled out from under
                # it is expected, not a reason to crash the turn loop.
                if press_fut is not None and press_fut.done():
                    press_fut.exception(); press_fut = None
                if mic_fut is not None and mic_fut.done():
                    mic_fut.exception(); mic_fut = None
            if typed_fut is None:
                typed_fut = loop.run_in_executor(_BLOCKERS, typed_q.get)
            if press_fut is None and ptt is not None:
                press_fut = loop.run_in_executor(_BLOCKERS, ptt.wait_press)
            waiters = {typed_fut}
            if press_fut is not None:
                waiters.add(press_fut)
            if _MIC["mode"] == "open":
                if mic_fut is None:
                    g = _MIC["gen"]
                    mic_fut = loop.run_in_executor(
                        _BLOCKERS, lambda g=g: capture_open(g))
                waiters.add(mic_fut)
            # Check-ins ride this same wait: when the feature is on, cap
            # the wait at the quiet time remaining so the loop wakes to
            # consider starting a conversation. The floor keeps that
            # wake-up sane once the window has already elapsed. A pending
            # reminder caps it too, so a spoken timer fires on time.
            timeout = None
            if check_interval:
                timeout = max(0.1, check_interval
                              - (time.monotonic() - _ACTIVITY["at"])
                              + random.uniform(
                                  0, check_jitter * check_interval))
            if _REMINDERS:
                soonest = max(0.0, min(r["at"] for r in _REMINDERS)
                              - time.monotonic())
                timeout = soonest if timeout is None else min(timeout, soonest)
            done, _ = await asyncio.wait(
                waiters, return_when=asyncio.FIRST_COMPLETED,
                timeout=timeout)
            if not done:
                # The wait timed out with nothing said. Reset the clock
                # either way. A due reminder speaks first and takes the
                # slot; otherwise, speak a check-in only if nothing is in
                # flight and the person has not asked for quiet.
                now = time.monotonic()
                due = [r for r in _REMINDERS if r["at"] <= now]
                for r in due:
                    _REMINDERS.remove(r)
                idle = now - _ACTIVITY["at"]
                _ACTIVITY["at"] = now
                if due:
                    _do_reminder(due)
                elif check_in_due(
                        CFG.get("check_in_enabled"), check_interval, idle,
                        mouth.speaking,
                        bool(speak_task and not speak_task.done()),
                        _PERM["fut"] is not None and not _PERM["fut"].done(),
                        paused=time.monotonic() < _CHECKIN["paused_until"],
                        quiet=_checkin_quiet(), held=_held_now()):
                    await _do_check_in()
                continue
            if typed_fut in done:
                text = typed_fut.result(); typed_fut = None
                if text and not await handle(text):
                    return
                continue
            if mic_fut is not None and mic_fut in done:
                try:
                    g, text = mic_fut.result()
                except Exception as e:
                    mic_fut = None
                    mic_fails += 1
                    if not explain_audio_failure(e):
                        log(f"[ears] open mic failed ({mic_fails}): {e!r}")
                    if mic_fails >= 3:
                        _MIC["mode"] = "ptt"
                        _MIC["gen"] += 1
                        mic_fails = 0
                        mouth.say("The open microphone keeps failing, "
                                  "so I'm switching to push to talk. "
                                  "Hold the key to reach me, and "
                                  "check this window for the error.")
                    continue
                mic_fut = None
                if g != _MIC["gen"]:
                    continue             # captured before a switch
                if text and not await handle(text):
                    return
                continue
            if press_fut is not None and press_fut in done:  # pragma: no branch -- a done set always holds typed or press
                press_fut.result(); press_fut = None
                if ptt is None:  # pragma: no cover -- reaching a press implies ptt
                    continue
                press_t = time.monotonic()
                perm_wait = (_PERM["fut"] is not None
                             and not _PERM["fut"].done())
                if speak_task and not speak_task.done() and not perm_wait:
                    log("[turn] interrupted mid-reply — key pressed")
                    speak_task.cancel()          # the button = interrupt
                # During a permission ask the TURN stays alive; the
                # press only silences playback and records the answer.
                mouth.shut_up()
                signals.static_stop()            # button kills the static too
                signals.set_state("listening")
                mouth.ducker.speech_start()      # duck NOW, while you talk
                print("[ptt] recording (release to send)...", flush=True)
                _MIC["btn"] = True               # open mic yields to the button
                try:
                    text = await loop.run_in_executor(
                        _BLOCKERS, lambda: record_held(
                            ptt.is_held,
                            on_release=lambda: signals.set_state("thinking"),
                            on_partial=emit_partial))
                except Exception as e:
                    # A device-level failure gets plain words instead of a
                    # raw exception. The pre-flight at startup cannot catch
                    # a microphone unplugged mid-session, and that is the
                    # case where the old message was worst: jargon, on
                    # every press, with the key hook still working so it
                    # looked like it was listening.
                    if explain_audio_failure(e):
                        mouth.say("I can't hear you. There's no working "
                                  "microphone I can use.")
                    else:
                        log(f"[ears] record/transcribe failed: {e!r}")
                        mouth.say("My ears hit an error. Check this "
                                  "window for the details.")
                    text = None
                finally:
                    _MIC["btn"] = False
                mouth.ducker.speech_end(0.2)     # snap back fast on release
                if not text:
                    log("[ptt] (tap or empty — ignored)")
                    signals.set_state("idle")
                    continue
                if not await handle(text, spoke_from=press_t):
                    return
    except KeyboardInterrupt:
        pass
    finally:
        _MIC["gen"] += 1     # abort any live open-mic capture right away
        if tracker and not tracker.done():
            tracker.cancel()
        if speak_task and not speak_task.done():
            speak_task.cancel()
        mouth.shutdown()  # restores the music on Ctrl-C / crash paths too
        signals.static_stop()
        signals.set_state("idle")
        await journal_flush(journal, brain)
        await brain.stop()
        log("[backtalk] hung up")


# Loopback port used purely as a mutex. Nothing is ever served on it.
_INSTANCE_PORT = 8791
_instance_lock = None


def _claim_single_instance() -> bool:
    """Refuse to be the second voice line on this machine, out loud.

    Two instances both hold the keyboard hook and both open the
    microphone, and the result looks EXACTLY like a broken talk key:
    presses register, the audio goes to whichever process won the
    device, and the loser reports an ignored tap. Nothing warned about
    it, so a user who double-clicks the Talk icon twice concludes the
    product is broken. The tell, when it was finally caught, was the
    same sentence transcribed twice at an identical timestamp.

    A bound socket is the mutex rather than a pid file, because the
    operating system releases it when this process dies HOWEVER it dies.
    A pid file outlives a crash or a force-kill and then lies about a
    process that is long gone, which is the failure it would exist to
    prevent.
    """
    global _instance_lock
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # No SO_REUSEADDR here on purpose: reuse is exactly what would let a
    # second instance bind alongside the first and defeat the whole point.
    try:
        s.bind(("127.0.0.1", _INSTANCE_PORT))
        s.listen(1)
    except OSError:
        s.close()
        return False
    _instance_lock = s
    return True


def _take_the_bus():
    """Take the signal bus over from whatever voice line came before.

    Holding the instance lock is the proof that nothing else can be
    speaking, so `idle` is the honest thing to write — and it repairs
    the one death no handler inside the old process can cover: a voice
    killed mid-speech by `taskkill /F` (the supervisor's stuck-turn
    restart) never reaches its own `finally`, so the bus would go on
    saying `speaking` about a process that is already gone, and a face
    would show a speaker that is no longer there.

    Registers the same park for our own exit, so the bus is never left
    lying about us either — see signals.register_exit_park for what
    that does and does not cover.
    """
    signals.register_exit_park()
    signals.park()


def main():
    if not _claim_single_instance():
        print("[backtalk] ANOTHER VOICE LINE IS ALREADY RUNNING on this "
              "machine, so this one is stopping.", flush=True)
        print("[backtalk] Two of them fight over the microphone and the "
              "talk key, which looks exactly like the talk key being "
              "broken. Use the window that is already open, or close it "
              "and start again.", flush=True)
        sys.exit(1)
    _take_the_bus()
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("\n[backtalk] interrupted — hanging up", flush=True)
    except BaseException:
        # The traceback normally goes to stderr, which is detached when
        # the supervisor spawns us, so a crash used to leave the bus
        # parked at "hung up" and NOTHING in the log to explain it. Land
        # it in backtalk.log, then die FOR REAL: a dead loop inside a
        # live process reads as a healthy voice to the supervisor, and it
        # will never restart what it believes is running. (os._exit, not
        # sys.exit, because a lingering non-daemon thread otherwise keeps
        # the corpse alive past shutdown.)
        try:
            log("[backtalk] CRASH — the voice line died:\n"
                + traceback.format_exc())
        except BaseException:
            pass
        os._exit(1)
    # The CLEAN path needs os._exit for the exact reason the crash path
    # above does: the whisper/ctranslate2 threadpool threads are
    # non-daemon and never finish, so interpreter shutdown hangs forever
    # AFTER "[backtalk] hung up" is already logged — a corpse that still
    # holds the instance lock and the signal bus, which the supervisor
    # reads as a healthy voice line that can never be restarted. (Seen
    # live: 31 threads, 74s CPU, :8791 held minutes after hangup.)
    # park() first because atexit does not run under os._exit.
    try:
        signals.park()
    except BaseException:
        pass
    os._exit(0)


if __name__ == "__main__":
    main()
