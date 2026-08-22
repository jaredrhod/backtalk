# backtalk: talk to your Claude Code agent out loud.
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Lookahead mouth check, no speakers needed: a fake synth (0.3s to first
audio, then streamed blocks) and a fake output device (sleeps in real
time, records every write) stand in for Kokoro and sounddevice.

Asserts:
  1. chunk boundaries play back-to-back (gap < 80ms) — the point of the
     synth-ahead thread; the old one-thread mouth paid ~1s per boundary
  2. a chunk shorter than the prebuffer still plays and completes
  3. barge-in drops everything queued/rendering, nothing stale plays,
     and the mouth speaks fresh text afterwards

Run:  .venv/Scripts/python tests/test_mouth_lookahead.py
"""
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backtalk  # noqa: E402

# Stub the signal bus and the ducker: file writes and OS mixer calls
# have no place in a unit check.
_sig = types.SimpleNamespace(static_stop=lambda: None,
                             set_state=lambda s: None,
                             feed_waveform=lambda p: None)
sys.modules["backtalk.signals"] = _sig
backtalk.signals = _sig


class _Ducker:
    def speech_start(self): pass
    def speech_end(self): pass
    def restore_now(self): pass


_duck = types.SimpleNamespace(Ducker=_Ducker)
sys.modules["backtalk.ducking"] = _duck
backtalk.ducking = _duck

import backtalk.mouth as mouth  # noqa: E402

RATE = 24000
writes: list = []          # (t_start, value, seconds)


class FakeOut:
    def __init__(self, samplerate, channels, dtype):
        self.rate = samplerate
        self.active = False

    def start(self):
        self.active = True

    def close(self, ignore_errors=False):
        self.active = False

    def write(self, pcm):
        writes.append((time.perf_counter(), int(pcm[0]) if len(pcm) else 0,
                       len(pcm) / self.rate))
        time.sleep(len(pcm) / self.rate)


_seq = [0]


def fake_synth(text, timeout=30.0):
    """Each chunk renders as a distinct constant sample value, so the
    write log tells chunks apart. Short text -> one block shorter than
    the prebuffer; otherwise 4 streamed blocks (1.2s of audio)."""
    _seq[0] += 1
    v = _seq[0]
    time.sleep(0.3)                       # time-to-first-audio
    blocks = 1 if len(text) < 6 else 4
    for i in range(blocks):
        if i:
            time.sleep(0.1)               # streaming render
        yield RATE, np.full(int(RATE * 0.3), v, dtype=np.int16)


mouth.sd = types.SimpleNamespace(OutputStream=FakeOut)
mouth.synth_stream = fake_synth


def _spans():
    """value -> (first write start, last write end), ignoring the zero
    padding _cut() writes."""
    spans: dict = {}
    for t, v, d in writes:
        if v == 0:
            continue
        first, end = spans.get(v, (t, t + d))
        spans[v] = (min(first, t), max(end, t + d))
    return spans


def main():
    m = mouth.Mouth()

    # 1. back-to-back chunks
    writes.clear()
    m.say("Number one. Number two. Number three. Number four.")
    m.wait_done(timeout=30)
    spans = _spans()
    assert sorted(spans) == [1, 2, 3, 4], spans
    order = [v for _, v, _ in writes if v]
    assert order == sorted(order), "chunks played out of order"
    gaps = [spans[v + 1][0] - spans[v][1] for v in (1, 2, 3)]
    print(f"boundary gaps: {[f'{g*1000:.0f}ms' for g in gaps]}")
    assert all(g < 0.08 for g in gaps), gaps
    assert m._pending == 0 and not m.speaking

    # 2. short chunk (under the prebuffer) completes
    writes.clear()
    m.say("Hi.")
    m.wait_done(timeout=10)
    assert _spans() and list(_spans()) == [5], _spans()
    assert m._pending == 0 and not m.speaking
    print("short chunk: ok")

    # 3. barge-in
    writes.clear()
    m.say("Number six. Number seven. Number eight. Number nine.")
    time.sleep(1.2)                       # chunk 6 is playing, 7 rendering
    m.shut_up()
    t_cut = time.perf_counter()
    time.sleep(2.0)
    late = [(round(t - t_cut, 2), v) for t, v, d in writes
            if v and t > t_cut + 0.15]
    assert not late, f"stale audio played after barge-in: {late}"
    assert set(_spans()) == {6}, _spans()
    assert m._pending == 0 and not m.speaking
    print(f"barge-in: cut mid-chunk, nothing stale played")

    writes.clear()
    m.say("Ten.")
    m.wait_done(timeout=10)
    # its synth number is whatever came after the aborted chunk 7 —
    # chunks 8 and 9 were dropped before they ever reached the synth
    vals = list(_spans())
    assert len(vals) == 1 and vals[0] > 6, _spans()
    assert m._pending == 0 and not m.speaking
    print("speaks fresh text after barge-in: ok")
    print("ALL OK")


if __name__ == "__main__":
    main()
