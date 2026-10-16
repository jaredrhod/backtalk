"""One whole turn, wired for real but with no hardware and no model.

test_e2e.py drives the running stack; this stitches the two halves of a
turn together in-process so CI can run it on every push:

  scripted mic -> real endpointer -> real wake gate -> strip_wake
      -> real speak_reply -> fake streaming brain -> fake mouth -> bus

Nothing here loads Whisper or a voice, and nothing touches a device, so
it proves the WIRING -- endpointing, the wake phrase, the command coming
out intact, first-sentence-alone batching, captions, and the bus -- while
leaving the model behaviour to test_stt_langs and test_espeak_fallback.
"""
from pathlib import Path
import asyncio
import sys
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                       # noqa: E402
from backtalk import ears, main as btmain               # noqa: E402
from backtalk.ears import Ears, FRAME_LEN, strip_wake    # noqa: E402

PHRASES = ["hey seyon", "hey sayon"]
HEARD = "Hey Seyon, what's the weather"
COMMAND = "what's the weather"

FAILURES = []


def check(name, cond, detail=""):
    print("    %s %s%s" % ("ok  " if cond else "FAIL", name,
                           ("  " + str(detail)) if detail else ""))
    if not cond:
        FAILURES.append(name)
    return cond


class FakeVad:
    def __init__(self, pattern):
        self.pattern = pattern
        self.i = 0

    def is_speech(self, buf, rate):
        v = self.pattern[self.i % len(self.pattern)]
        self.i += 1
        return v


class FakeStream:
    def __init__(self, frames):
        self.frames = frames
        self.i = 0

    def read(self, n):
        f = self.frames[self.i % len(self.frames)]
        self.i += 1
        return f.reshape(-1, 1), None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeMouth:
    def __init__(self):
        self.chunks = []
        self.pending = []

    def say_chunk(self, s, pending=None):
        self.chunks.append(s)
        self.pending.append(list(pending or []))

    def say(self, s):
        pass


class StreamBrain:
    def __init__(self, sentences):
        self.sentences = sentences
        self.got = None

    async def ask_stream(self, text):
        self.got = text
        for s in self.sentences:
            yield s


def frame(amp):
    return np.full(FRAME_LEN, amp, np.int16)


def half_one_capture_command():
    """Mic -> text: the real endpointer and wake gate, a faked ear."""
    print("\n--- the ear half: mic -> wake phrase -> command ---")
    script = [(True, frame(3000))] * 20 + [(False, frame(0))] * 2
    stream = FakeStream([f for _, f in script])
    saved_mic, saved_tr = ears._open_mic, ears.transcribe
    ears._open_mic = lambda: stream
    ears.transcribe = lambda pcm, gate_no_speech=False: HEARD
    try:
        e = Ears(silence_ms=60)
        e.vad = FakeVad([s for s, _ in script])
        woke = e.wait_for_wake(PHRASES)
    finally:
        ears._open_mic, ears.transcribe = saved_mic, saved_tr
    check("the wake utterance comes back", woke == HEARD, woke)
    check("the phrase strips to the command",
          strip_wake(woke, PHRASES) == COMMAND, strip_wake(woke, PHRASES))
    return strip_wake(woke, PHRASES)


def half_two_speak():
    """Text -> spoken reply: the real speak_reply, faked brain and mouth."""
    print("\n--- the mouth half: command -> batched spoken reply -> bus ---")
    brain = StreamBrain(["First sentence.", "Second sentence.",
                         "Third sentence."])
    mouth = FakeMouth()
    with mock.patch.object(btmain, "signals") as sig, \
         mock.patch.object(btmain, "log"):
        asyncio.run(btmain.speak_reply(brain, mouth, COMMAND))
    check("the brain was handed the bare command", brain.got == COMMAND,
          brain.got)
    check("first sentence ships alone",
          mouth.chunks[:1] == ["First sentence."], mouth.chunks)
    check("the rest go in one two-sentence breath",
          mouth.chunks[1:] == ["Second sentence. Third sentence."],
          mouth.chunks)
    captions = [c.args[0] for c in sig.caption.call_args_list]
    check("each spoken chunk is captioned", captions == mouth.chunks, captions)
    check("a full reply does not park the bus (audio drains it)",
          not sig.set_state.called, sig.set_state.call_args_list)
    check("no error static on a clean turn",
          not sig.static_stop.called, sig.static_stop.call_args_list)


print("=" * 66)
print("WHOLE TURN (wired, no hardware, no model)")
print("=" * 66)

cmd = half_one_capture_command()
half_two_speak()

print("\n" + "=" * 66)
print("PIPELINE OK" if not FAILURES
      else "PIPELINE FAILURES: %s" % FAILURES)
sys.exit(1 if FAILURES else 0)
