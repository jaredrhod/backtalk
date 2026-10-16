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
"""The signal bus — tiny files any other program can watch.

The voice line leaves notes; faces read the notes. That one dumb trick
is the whole integration surface:

  .voice_state        idle | listening | thinking | speaking
  .voice_waveform     JSON {ts, samples: [64 floats]} while audio plays
  .voice_loading_pid  exists while the thinking sound is playing
  .voice_rate_limits  JSON {window: {utilization, resets_at}} — only
                      written when show_usage is on

Written to signals_dir (default: the repo root). Visualizers built on
this contract just work.

THE BAREHANDS SEAM: set barehands_state_dir in backtalk.json to a
barehands checkout's state/ folder and the same signals are mirrored in
its format (state/state as a bare word, state/wave.json normalized
0..1) — the on-screen ring becomes your agent's face with zero glue.

Every write is wrapped: the bus must never crash the voice line.
"""
import atexit
import json
import os
import subprocess
import sys
import time

import numpy as np

from backtalk.config import CFG

_DIR = CFG["signals_dir"]
_STATE_FILE = os.path.join(_DIR, ".voice_state")
_WAVEFORM_FILE = os.path.join(_DIR, ".voice_waveform")
_LOADING_PID_FILE = os.path.join(_DIR, ".voice_loading_pid")
_DIRECTION_FILE = os.path.join(_DIR, ".voice_direction")
_REPLY_DONE_FILE = os.path.join(_DIR, ".voice_reply_done")
_RATE_LIMIT_FILE = os.path.join(_DIR, ".voice_rate_limits")

_BH = CFG.get("barehands_state_dir") or ""
_BH_STATE = os.path.join(_BH, "state") if _BH else ""
_BH_WAVE = os.path.join(_BH, "wave.json") if _BH else ""

_THINKING_SOUND = CFG.get("thinking_sound") or ""

_WAVEFORM_MIN_INTERVAL = 1.0 / 15   # ~15 writes/sec is plenty for 60fps reads
_last_waveform_write = 0.0
_static_proc: subprocess.Popen | None = None


def set_state(name: str):
    """Write the state. Never raises — the show must go on."""
    try:
        with open(_STATE_FILE, "w") as f:
            f.write(name)
    except OSError:
        pass
    if _BH_STATE:
        try:
            with open(_BH_STATE, "w") as f:
                f.write(name)
        except OSError:
            pass


def park():
    """Leave the bus reading `idle`, with the thinking sound stopped.

    The state that is true whenever nothing is running, so it is what a
    departing voice line should leave behind. Never raises."""
    set_state("idle")
    static_stop()


def register_exit_park():
    """Park the bus when this process leaves, however it leaves.

    A voice that dies mid-speech otherwise strands the bus at
    `speaking`: the finally in amain() is what normally writes `idle`,
    and a dying process that never reaches it leaves a face showing a
    speaker that is long gone. atexit covers the deaths a Python
    process can clean up after — an uncaught exception, sys.exit, a
    crash that unwinds the interpreter.

    It cannot cover `taskkill /F`, which is how the supervisor restarts
    a wedged voice, and no handler inside the dying process ever could.
    That case is answered from the other side instead: the successor
    parks the bus the moment it claims the instance, because holding
    that lock is proof that nothing else can be speaking.
    """
    atexit.register(park)


def feed_waveform(pcm: np.ndarray):
    """Feed one PCM block (int16) — throttled, downsampled to 64 points.

    Also re-asserts state="speaking" on the same throttle: this only runs
    while the mouth is audibly playing, so the bus self-heals within
    ~70ms if a stray writer stomps the state mid-speech. (That self-heal
    rule once closed a bug that took a whole evening to find.)"""
    global _last_waveform_write
    if pcm.size == 0:
        return
    now = time.time()
    if now - _last_waveform_write < _WAVEFORM_MIN_INTERVAL:
        return
    _last_waveform_write = now
    try:
        idx = np.linspace(0, pcm.size - 1, 64).astype(int)
        raw = pcm[idx].astype(float)
        with open(_WAVEFORM_FILE, "w") as f:
            f.write(json.dumps({"ts": now, "samples": raw.tolist()}))
        if _BH_WAVE:
            norm = np.clip(np.abs(raw) / 32768.0, 0.0, 1.0)
            with open(_BH_WAVE, "w") as f:
                f.write(json.dumps({"ts": now, "samples": norm.tolist()}))
    except (OSError, ValueError):
        pass
    set_state("speaking")


def direction(items):
    """Stage directions the agent wrote into its reply, published at the
    moment the audio carrying them starts playing.

    Your agent can emit `<<anything>>` inline and backtalk will never speak
    it. What the tag MEANS is deliberately not backtalk's business: it
    publishes the raw strings and something else decides. That is the whole
    reason this is a file and not a plugin API.

    The timing is the point, and it is the one part a watcher cannot do for
    itself: these fire when the sentence becomes AUDIBLE, not when the model
    generated it. A screen cue lands on the spoken word instead of seconds
    early. Never raises."""
    if not items:
        return
    try:
        with open(_DIRECTION_FILE, "w") as f:
            f.write(json.dumps({"ts": time.time(), "directions": list(items)}))
    except OSError:
        pass


_LANG_FILE = os.path.join(_DIR, ".voice_language")


def language(code: str | None):
    """The language the current turn is being held in, so a face can show
    which tongue Jarvis picked. None when it could not be told apart."""
    try:
        with open(_LANG_FILE, "w") as f:
            json.dump({"ts": time.time(), "language": code or None}, f)
    except (OSError, TypeError, ValueError):
        pass


def reply_done():
    """One reply has finished speaking and its audio has fully drained.

    Distinct from the state going idle, which also happens in the gaps
    BETWEEN sentences of the same reply. Anything waiting for the agent to
    genuinely stop talking wants this rather than a state flicker. Never
    raises."""
    try:
        with open(_REPLY_DONE_FILE, "w") as f:
            f.write(json.dumps({"ts": time.time()}))
    except OSError:
        pass


_rate_limits: dict = {}


def set_rate_limit(window: str, utilization, resets_at):
    """One usage window's reading — how much of the plan is spent.

    Merged rather than replaced, because the reading arrives one window
    at a time and a face wants to draw both at once. `utilization` is a
    0..1 fraction (or None when the window has not reported a number
    yet, which is a real state and not an error); `resets_at` is a unix
    epoch.

    NOTHING CALLS THIS UNLESS show_usage IS ON. That is a privacy
    default, not a performance one: this is the account holder's own
    spend, and it renders on a face that may well be pointed at a
    camera. It never appears without being asked for. (Community fix,
    ai-visualizer issue #1.)

    Never raises."""
    if not window:
        return
    _rate_limits[window] = {"utilization": utilization,
                            "resets_at": resets_at}
    try:
        with open(_RATE_LIMIT_FILE, "w") as f:
            f.write(json.dumps(_rate_limits))
    except OSError:
        pass


def _player_cmd(path: str) -> list[str] | None:
    if sys.platform == "darwin":
        return ["afplay", "-v", "0.35", path]
    for cand in ("ffplay", "aplay", "paplay"):
        from shutil import which
        if which(cand):
            if cand == "ffplay":
                return ["ffplay", "-nodisp", "-autoexit", "-loglevel",
                        "quiet", "-volume", "35", path]
            return [cand, path]
    return None


def static_start():
    """Optional thinking sound — plays while the brain works."""
    global _static_proc
    if not _THINKING_SOUND or not os.path.exists(_THINKING_SOUND):
        return
    static_stop()
    cmd = _player_cmd(_THINKING_SOUND)
    if not cmd:
        return
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        _static_proc = None
        return
    # The player is running now, so remember it even if the pid file cannot
    # be written: otherwise a successful Popen whose pid-file write failed
    # would orphan a looping player for the rest of the session.
    _static_proc = proc
    try:
        with open(_LOADING_PID_FILE, "w") as f:
            f.write(str(proc.pid))
    except OSError:
        pass


def static_stop():
    global _static_proc
    if _static_proc is not None:
        try:
            _static_proc.terminate()
        except OSError:
            pass
        _static_proc = None
    try:
        os.remove(_LOADING_PID_FILE)
    except OSError:
        pass


_CAPTION_FILE = os.path.join(_DIR, ".voice_caption")


def caption(text: str):
    """The words Jarvis is speaking right now, published as they stream.

    The face already knows it is thinking or speaking; this is the
    sentence itself, so a display can caption the reply as it happens
    instead of showing only an expression. Written per sentence, so a
    reader polling the bus sees the reply build up. Never raises.
    """
    try:
        with open(_CAPTION_FILE, "w") as f:
            f.write(json.dumps({"ts": time.time(), "text": text or ""}))
    except OSError:
        pass


def caption_clear():
    """Drop the caption once the reply is over."""
    try:
        os.remove(_CAPTION_FILE)
    except OSError:
        pass


_HEARD_FILE = os.path.join(_DIR, ".voice_heard")


def heard(text: str, final: bool = False):
    """The words being heard as they arrive, for a face to caption live.

    caption() is what the agent is SAYING; this is what it is HEARING. A
    partial transcript lands here while the person is still talking (see
    ears.PartialStream), so a display can show the words build up instead
    of waiting for the whole turn. `final` marks the finished transcript,
    which may differ from the last partial. Never raises."""
    try:
        with open(_HEARD_FILE, "w") as f:
            f.write(json.dumps({"ts": time.time(), "text": text or "",
                                "final": bool(final)}))
    except OSError:
        pass


def heard_clear():
    """Drop the heard line once the turn is taken over."""
    try:
        os.remove(_HEARD_FILE)
    except OSError:
        pass


_ACTIVE_FILE = os.path.join(_DIR, ".voice_active_window")


def active_window(text: str):
    """What the activity tracker last saw focused, for a face to show.

    Local only and never raises: a display is a courtesy, not something
    the voice line should ever fail over. The reader treats a stale
    timestamp as "unknown", so there is nothing to clear.
    """
    try:
        with open(_ACTIVE_FILE, "w") as f:
            f.write(json.dumps({"ts": time.time(), "text": text or ""}))
    except OSError:
        pass
