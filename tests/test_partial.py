"""Partial STT streaming, isolated.

PartialStream decides, purely, when an in-progress utterance is worth a
partial transcribe and emits the growing text; PartialTicker runs that
decision off the capture thread so the audio loop never waits on whisper.
listen_once/record_held publish partials through them. Everything here is
faked -- no mic, no model -- so it is the control flow that is checked: the
throttle, the reset on noise, and the promise that the ticker is stopped
before the final transcript.
"""
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                       # noqa: E402
from backtalk import ears                               # noqa: E402
from backtalk.config import CFG                         # noqa: E402
from backtalk.ears import (Ears, FRAME_LEN,             # noqa: E402
                           PartialStream, PartialTicker)

FAILURES = []


def check(name, cond, detail=""):
    print("    %s %s%s" % ("ok  " if cond else "FAIL", name,
                           ("  " + str(detail)) if detail else ""))
    if not cond:
        FAILURES.append(name)
    return cond


def frame(amp):
    return np.full(FRAME_LEN, amp, np.int16)


class FakeVad:
    def __init__(self, pattern):
        self.pattern = pattern
        self.i = 0

    def is_speech(self, buf, rate):
        v = self.pattern[self.i % len(self.pattern)]
        self.i += 1
        return v


class FakeStream:
    """A mic that yields scripted frames, slowly enough that a background
    ticker gets to run -- the timing a real capture always has."""

    def __init__(self, frames, delay=0.015):
        self.frames = frames
        self.i = 0
        self.delay = delay

    def read(self, n):
        f = self.frames[self.i % len(self.frames)]
        self.i += 1
        if self.delay:
            time.sleep(self.delay)
        return f.reshape(-1, 1), None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_interval():
    print("\n--- _partial_interval: config, with a floor ---")
    saved = CFG.get("partial_stt_ms")
    CFG["partial_stt_ms"] = 700
    check("milliseconds become seconds", ears._partial_interval() == 0.7)
    CFG["partial_stt_ms"] = 0
    check("a zero interval keeps the floor", ears._partial_interval() == 0.2)
    CFG["partial_stt_ms"] = saved


def test_stream_ranks():
    print("\n--- PartialStream: the throttle ---")
    now = {"t": 0.0}
    calls = []
    emitted = []

    def tr(pcm):
        calls.append(len(pcm))
        return f"words {len(calls)}"

    s = PartialStream(tr, emitted.append, interval=1.0, min_frames=2,
                      clock=lambda: now["t"])
    s.push(frame(1))
    check("a single frame is not worth a partial", s.tick() is None)
    s.push(frame(1))
    check("the first ripe tick emits", s.tick() == "words 1", emitted)
    s.push(frame(1))
    check("a tick inside the interval waits", s.tick() is None)
    now["t"] = 1.5
    s.push(frame(1))
    check("a tick after the interval emits the new text",
          s.tick() == "words 2", emitted)
    check("only the changed readings were emitted",
          emitted == ["words 1", "words 2"], emitted)


def test_stream_quiet_and_repeat():
    print("\n--- PartialStream: silence and repeats ---")
    emitted = []
    s = PartialStream(lambda p: "", emitted.append, interval=0.0,
                      min_frames=2)
    s.push(frame(1))
    s.push(frame(1))
    check("an empty partial emits nothing",
          s.tick() is None and emitted == [], emitted)

    s2 = PartialStream(lambda p: "same", emitted.append, interval=0.0,
                       min_frames=2)
    s2.push(frame(1))
    s2.push(frame(1))
    check("the first text emits", s2.tick() == "same")
    s2.push(frame(1))
    check("an unchanged reading is not re-emitted",
          s2.tick() is None and emitted == ["same"], emitted)


def test_stream_reset():
    print("\n--- PartialStream: reset forgets the audio ---")
    emitted = []
    s = PartialStream(lambda p: "x", emitted.append, interval=0.0,
                      min_frames=2)
    s.push(frame(1))
    s.push(frame(1))
    s.tick()
    s.reset()
    check("a reset forgets the audio", s.tick() is None)


def test_stream_tick_survives_a_racing_reset():
    print("\n--- PartialStream: a reset mid-decode discards the words ---")
    emitted = []
    s = PartialStream(lambda p: "words", emitted.append, interval=0.0,
                      min_frames=2)
    for _ in range(3):
        s.push(frame(1))

    def racing(pcm):
        s.reset()            # the capture thread clears the buffer mid-tick
        return "words"

    s._transcribe = racing
    out = s.tick()
    check("a reset during the transcribe drops the stale caption",
          out is None and emitted == [], (out, emitted))
    check("the race leaves the stream empty and re-usable",
          s.tick() is None)


def test_ticker():
    print("\n--- PartialTicker: drives the stream, stops clean ---")
    emitted = []
    hit = threading.Event()

    def emit(text):
        emitted.append(text)
        hit.set()

    s = PartialStream(lambda p: "live", emit, interval=0.0, min_frames=2)
    tick = PartialTicker(s, poll=0.01)
    tick.start()
    s.push(frame(1))
    s.push(frame(1))
    check("the ticker emits a partial", hit.wait(3.0), emitted)
    tick.stop()
    check("the emitted text is what the stream produced",
          emitted == ["live"], emitted)
    check("stopping an unstarted ticker is safe", PartialTicker(s).stop() is None)


def test_listen_once_partial():
    print("\n--- listen_once publishes partials, then transcribes ---")
    script = ([True] * 24 + [False] * 2)
    stream = FakeStream([frame(3000) if s else frame(0) for s in script])
    saved_mic, saved_tr = ears._open_mic, ears.transcribe
    calls = []
    ears._open_mic = lambda: stream
    ears.transcribe = lambda pcm, gate_no_speech=False: (
        calls.append(len(pcm)) or "a sentence")
    emitted = []
    got = threading.Event()

    def emit(text):
        emitted.append(text)
        got.set()

    try:
        e = Ears(silence_ms=60)
        e.vad = FakeVad(script)
        text = e.listen_once(on_partial=emit)
    finally:
        ears._open_mic, ears.transcribe = saved_mic, saved_tr
    check("the final transcript still comes back", text == "a sentence", text)
    check("a partial was published while speaking", got.wait(3.0), emitted)
    check("the final transcription is the longest, and the last call",
          len(calls) >= 2 and calls[-1] == max(calls), calls)


def test_gate_resets_partial():
    print("\n--- a gate that rises mid-capture resets the partial ---")
    script = [True] * 24 + [False] * 2
    stream = FakeStream([frame(3000) if s else frame(0) for s in script])
    saved_mic, saved_tr = ears._open_mic, ears.transcribe
    ears._open_mic = lambda: stream
    ears.transcribe = lambda pcm, gate_no_speech=False: "a sentence"
    flips = [True]

    def gate():
        first = flips[0]
        flips[0] = False
        return first

    try:
        e = Ears(silence_ms=60)
        e.vad = FakeVad(script)
        text = e.listen_once(gate=gate, on_partial=lambda t: None)
    finally:
        ears._open_mic, ears.transcribe = saved_mic, saved_tr
    check("a gated frame does not derail the next utterance",
          text == "a sentence", text)


def test_noise_resets_partial():
    print("\n--- a dropped blip resets the partial ---")
    script = ([True] * 5 + [False] * 3 + [True] * 24 + [False] * 2)
    stream = FakeStream([frame(3000) if s else frame(0) for s in script])
    saved_mic, saved_tr = ears._open_mic, ears.transcribe
    ears._open_mic = lambda: stream
    ears.transcribe = lambda pcm, gate_no_speech=False: "a sentence"
    emitted = []
    got = threading.Event()
    try:
        e = Ears(silence_ms=60)
        e.vad = FakeVad(script)
        text = e.listen_once(on_partial=lambda t: (emitted.append(t),
                                                   got.set()))
    finally:
        ears._open_mic, ears.transcribe = saved_mic, saved_tr
    check("the sentence after the blip is transcribed",
          text == "a sentence", text)
    check("a partial still fired for the real sentence", got.wait(3.0),
          emitted)


def test_record_held_partial():
    print("\n--- record_held shows partials during the hold ---")
    stream = FakeStream([frame(3000)])
    saved_mic, saved_tr = ears._open_mic, ears.transcribe
    ears._open_mic = lambda: stream
    ears.transcribe = lambda pcm, gate_no_speech=False: "held words"
    got = threading.Event()
    held = {"n": 0}

    def is_held():
        held["n"] += 1
        return held["n"] <= 20

    try:
        text = ears.record_held(is_held, on_partial=lambda t: got.set())
    finally:
        ears._open_mic, ears.transcribe = saved_mic, saved_tr
    check("hold-to-talk still transcribes", text == "held words", text)
    check("the hold published a partial", got.wait(3.0), held)


def test_record_clip():
    print("\n--- record_clip: a fixed run of frames ---")
    stream = FakeStream([frame(3000)])
    saved = ears._open_mic
    ears._open_mic = lambda: stream
    try:
        got = ears.Ears().record_clip(0.06)
    finally:
        ears._open_mic = saved
    check("record_clip returns int16 audio", got.dtype == np.int16, got.dtype)
    check("record_clip returns the requested length",
          got.shape == (int(0.06 * ears.RATE / ears.FRAME_LEN) * ears.FRAME_LEN,),
          got.shape)


def test_owner_gate():
    print("\n--- owner_check gates the utterance ---")
    # Two spoken turns. The owner check rejects the first, accepts the
    # second: the rejected clip must never reach whisper and the second
    # must come back as the transcript.
    script = ([True] * 20 + [False] * 2 + [True] * 20 + [False] * 2)
    stream = FakeStream([frame(3000) if s else frame(0) for s in script],
                        delay=0.0)
    saved_mic, saved_tr = ears._open_mic, ears.transcribe
    calls = []
    ears._open_mic = lambda: stream
    ears.transcribe = lambda pcm, gate_no_speech=False: (
        calls.append(len(pcm)) or "the owner's words")
    seen = []

    def owner(pcm):
        seen.append(len(pcm))
        return len(seen) == 2

    try:
        e = ears.Ears(silence_ms=60)
        e.vad = FakeVad(script)
        text = e.listen_once(owner_check=owner)
    finally:
        ears._open_mic, ears.transcribe = saved_mic, saved_tr
    check("the rejected clip is dropped, the owned one transcribed",
          text == "the owner's words", text)
    check("the owner check ran per utterance", len(seen) == 2, seen)
    check("only the owned clip reached whisper", len(calls) == 1, calls)


def test_owner_gate_accepts():
    print("\n--- owner_check that accepts lets the turn through ---")
    script = [True] * 20 + [False] * 2
    stream = FakeStream([frame(3000) if s else frame(0) for s in script],
                        delay=0.0)
    saved_mic, saved_tr = ears._open_mic, ears.transcribe
    ears._open_mic = lambda: stream
    ears.transcribe = lambda pcm, gate_no_speech=False: "my words"
    try:
        e = ears.Ears(silence_ms=60)
        e.vad = FakeVad(script)
        text = e.listen_once(owner_check=lambda pcm: True)
    finally:
        ears._open_mic, ears.transcribe = saved_mic, saved_tr
    check("an accepted utterance is transcribed", text == "my words", text)


def test_owner_gate_via_wake():
    print("\n--- wait_for_wake forwards the owner check ---")
    seen = []

    def listen_once(self, gate=None, abort=None, timeout_s=None,
                    on_partial=None, owner_check=None):
        seen.append(owner_check)
        return "hey seyon"

    saved = ears.Ears.listen_once
    ears.Ears.listen_once = listen_once
    try:
        got = ears.Ears().wait_for_wake(["hey seyon"],
                                        owner_check=lambda p: True)
    finally:
        ears.Ears.listen_once = saved
    check("the wake matched", got == "hey seyon", got)
    check("the owner check was handed down", seen and seen[0] is not None)


print("=" * 66)
print("PARTIAL STT STREAMING")
print("=" * 66)
test_interval()
test_stream_ranks()
test_stream_quiet_and_repeat()
test_stream_reset()
test_stream_tick_survives_a_racing_reset()
test_ticker()
test_listen_once_partial()
test_gate_resets_partial()
test_noise_resets_partial()
test_record_held_partial()
test_record_clip()
test_owner_gate()
test_owner_gate_accepts()
test_owner_gate_via_wake()

print("\n" + "=" * 66)
print("PARTIAL OK" if not FAILURES else "PARTIAL FAILURES: %s" % FAILURES)
sys.exit(1 if FAILURES else 0)
