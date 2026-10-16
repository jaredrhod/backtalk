"""Open-mic endpointing and the hands-free loudness gate.

listen_once() is the hands-free path: VAD opens an utterance after ~120ms
of speech, closes it after a trailing pause, and refuses to send a capture
to whisper unless the speech in it is loud enough to be a person (the
gate that stops whisper inventing "Thanks for watching!" over room hum).

The mic and the transcriber are faked -- the same trick test_ptt.py uses
-- so this is pure control-flow with no hardware and no model.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                       # noqa: E402
from backtalk import ears                               # noqa: E402
from backtalk.config import CFG                         # noqa: E402
from backtalk.ears import (Ears, FRAME_LEN,             # noqa: E402
                           speech_is_audible)

FAILURES = []


def check(name, cond, detail=""):
    print("    %s %s%s" % ("ok  " if cond else "FAIL", name,
                           ("  " + str(detail)) if detail else ""))
    if not cond:
        FAILURES.append(name)
    return cond


class FakeVad:
    """is_speech() driven by a scripted pattern, cycling if overrun."""

    def __init__(self, pattern):
        self.pattern = pattern
        self.i = 0

    def is_speech(self, buf, rate):
        v = self.pattern[self.i % len(self.pattern)]
        self.i += 1
        return v


class FakeStream:
    """A mic that yields scripted frames, repeating them if overrun."""

    def __init__(self, frames, abort_after=None):
        self.frames = frames
        self.i = 0
        self.abort_after = abort_after

    def read(self, n):
        f = self.frames[self.i % len(self.frames)]
        self.i += 1
        return f.reshape(-1, 1), None

    def abort(self):
        return self.abort_after is not None and self.i > self.abort_after

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def frame(amp):
    return np.full(FRAME_LEN, amp, np.int16)


def capture(script, silence_ms=60, abort_after=None):
    """Run listen_once against a scripted mic; return (text, transcribe_calls)."""
    stream = FakeStream([f for _, f in script], abort_after=abort_after)
    ears._open_mic = lambda: stream
    calls = []
    ears.transcribe = lambda pcm, gate_no_speech=False: (
        calls.append(len(pcm)) or "a sentence")
    ears_mod = Ears(silence_ms=silence_ms)
    ears_mod.vad = FakeVad([s for s, _ in script])
    text = ears_mod.listen_once(
        abort=(stream.abort if abort_after else None))
    return text, calls


def test_loudness_gate():
    print("\n--- the loudness gate itself ---")
    check("room hum is not audible", not speech_is_audible(8.0))
    check("just below the floor", not speech_is_audible(199.0))
    check("at the floor", speech_is_audible(200.0))
    check("real speech", speech_is_audible(3000.0))
    saved = CFG.get("stt_min_rms")
    CFG["stt_min_rms"] = 0
    check("floor 0 disables the gate", speech_is_audible(0.0))
    CFG["stt_min_rms"] = saved


def test_captures_a_sentence():
    print("\n--- sustained speech opens and closes one utterance ---")
    script = [(True, frame(3000))] * 12 + [(False, frame(0))] * 2
    text, calls = capture(script)
    check("the sentence is transcribed", text == "a sentence", text)
    check("exactly one transcription", len(calls) == 1, calls)


def test_ignores_a_leading_blip():
    print("\n--- a short blip is ignored, the sentence after it is not ---")
    script = ([(True, frame(3000))] * 3 + [(False, frame(0))] * 5
              + [(True, frame(3000))] * 12 + [(False, frame(0))] * 2)
    text, calls = capture(script)
    check("the later sentence is transcribed", text == "a sentence", text)
    check("the blip alone produced no turn", len(calls) == 1, calls)


def test_drops_quiet_noise():
    print("\n--- speech-shaped but quiet noise never reaches whisper ---")
    script = [(True, frame(8))] * 12 + [(False, frame(0))] * 2
    text, calls = capture(script, abort_after=40)
    check("no transcript", text is None, text)
    check("whisper was never called", calls == [], calls)


def test_no_speech_gate():
    print("\n--- the model's own no-speech score ---")
    check("clear speech is kept", ears.speech_is_confident(0.05))
    check("a music-bed transcript is not",
          not ears.speech_is_confident(0.75))
    saved = CFG.get("stt_no_speech_prob")
    CFG["stt_no_speech_prob"] = 0
    check("0 disables it", ears.speech_is_confident(0.99))
    CFG["stt_no_speech_prob"] = saved


def test_duration_gate():
    print("\n--- a blip shorter than a phrase is dropped ---")
    short = [(True, frame(3000))] * 10 + [(False, frame(0))] * 2
    text, calls = capture(short, abort_after=40)
    check("a 300ms blip never reaches whisper",
          text is None and calls == [], (text, calls))
    long_ = [(True, frame(3000))] * 30 + [(False, frame(0))] * 2
    text, calls = capture(long_)
    check("a phrase-long utterance gets through",
          text == "a sentence" and len(calls) == 1, (text, calls))


def test_timeout_without_speech():
    print("\n--- a timeout with no speech returns None ---")
    stream = FakeStream([frame(0)])
    ears._open_mic = lambda: stream
    ears.transcribe = lambda pcm, gate_no_speech=False: "never"
    e = Ears(silence_ms=60)
    e.vad = FakeVad([False])
    got = e.listen_once(timeout_s=0.01)
    check("no speech within the window -> None", got is None, got)


def test_gate_suppresses_then_opens():
    print("\n--- the barge-in gate mutes the open mic ---")
    frames = [frame(0)] * 3 + [frame(3000)] * 12 + [frame(0)] * 2
    stream = FakeStream(frames)
    ears._open_mic = lambda: stream
    ears.transcribe = lambda pcm, gate_no_speech=False: "a sentence"
    gate_calls = {"n": 0}

    def gate():
        gate_calls["n"] += 1
        return gate_calls["n"] <= 3     # speakers talking for the first frames

    e = Ears(silence_ms=60)
    e.vad = FakeVad([True] * 12 + [False] * 2)  # gated frames skip the VAD
    text = e.listen_once(gate=gate, timeout_s=5)
    check("the gate suppresses the mic, then it hears",
          text == "a sentence", text)
    check("the gated frames were dropped", gate_calls["n"] >= 3)


print("=" * 66)
print("OPEN-MIC ENDPOINTING")
print("=" * 66)

saved_mic, saved_transcribe = ears._open_mic, ears.transcribe
saved_min = CFG.get("stt_min_speech_ms")
try:
    CFG["stt_min_speech_ms"] = 0     # old 240ms floor for the core tests
    test_loudness_gate()
    test_captures_a_sentence()
    test_ignores_a_leading_blip()
    test_drops_quiet_noise()
    test_no_speech_gate()
    CFG["stt_min_speech_ms"] = 600   # and the phrase-length floor on
    test_duration_gate()
    CFG["stt_min_speech_ms"] = 0
    test_timeout_without_speech()
    test_gate_suppresses_then_opens()
finally:
    ears._open_mic, ears.transcribe = saved_mic, saved_transcribe
    CFG["stt_min_speech_ms"] = saved_min

print("\n" + "=" * 66)
print("ENDPOINTING OK" if not FAILURES else "ENDPOINTING FAILURES: %s" % FAILURES)
sys.exit(1 if FAILURES else 0)
