"""Hold-to-talk capture: the release callback fires the instant the key
comes up, taps are ignored, and transcription runs only on real speech.

record_held() is the button path (Home key), so these pin the timing
contract main.py depends on: on_release -> state "thinking" must happen
BEFORE the tail frames and the transcription, not after them -- the
person already thinks of themselves as done talking. The cap firing it
too is also pinned: a turn that hits 60s proceeds either way, and
leaving the UI waiting would be the wrong story.

The mic and the transcriber are faked, so no hardware and no whisper
model are involved -- this is pure control-flow timing."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                   # noqa: E402
from backtalk import ears                           # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print("    %s %s%s" % ("ok  " if cond else "FAIL", name,
                           ("  " + str(detail)) if detail else ""))
    if not cond:
        FAILURES.append(name)
    return cond


class FakeStream:
    def read(self, n):
        return np.zeros((n, 2), np.int16), None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def run(held_for, max_s=60.0):
    """record_held against a fake mic; returns (events, result)."""
    events = []
    ears._open_mic = FakeStream
    ears.transcribe = lambda pcm: (events.append("transcribe") or "heard")
    state = {"n": 0}

    def held():
        state["n"] += 1
        return held_for is None or state["n"] <= held_for

    result = ears.record_held(held, max_s=max_s,
                              on_release=lambda: events.append("release"))
    return events, result


print("=" * 66)
print("HOLD TO TALK")
print("=" * 66)

print("\n--- a real hold: release fires before the transcription ---")
events, result = run(held_for=12)
check("events in order", events == ["release", "transcribe"], events)
check("speech is transcribed", result == "heard", result)

print("\n--- a tap: callback still fires, nothing is transcribed ---")
events, result = run(held_for=1)
check("release fired at the tap", events == ["release"], events)
check("tap returns None", result is None, result)

print("\n--- the time cap: the turn proceeds, the UI is not left waiting ---")
events, result = run(held_for=None, max_s=0.3)
check("release fired at the cap", events == ["release", "transcribe"], events)
check("still transcribed", result == "heard", result)

print("\n--- no callback is fine ---")
ears._open_mic = FakeStream
ears.transcribe = lambda pcm: "heard"
check("records without on_release",
      ears.record_held(lambda: False) is None)

print("\n" + "=" * 66)
print("PTT OK" if not FAILURES else "PTT FAILURES: %s" % FAILURES)
sys.exit(1 if FAILURES else 0)
