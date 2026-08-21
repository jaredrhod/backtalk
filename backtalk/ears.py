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
"""The ears — mic capture with VAD endpointing, transcribed in-process
by faster-whisper. Local, free, no server, no API key.

record_held() is the hold-to-talk capture (the button is the VAD).
Ears.listen_once() is the legacy open-mic mode: blocks until one
complete utterance is heard, then returns its transcript. Endpointing:
an utterance opens after ~120ms of sustained speech, closes after
`silence_ms` of trailing quiet. A `gate` callable can suppress
listening (so the open mic ignores the speakers unless barge-in is on).
"""
import re
import threading

import numpy as np
import sounddevice as sd
import webrtcvad

from backtalk.config import CFG
from backtalk.vlog import log

RATE = 16000
FRAME_MS = 30
FRAME_LEN = RATE * FRAME_MS // 1000  # samples per frame
OPEN_FRAMES = 4        # ~120ms speech to open an utterance
MAX_UTTER_S = 30

_NONSPEECH = re.compile(r"[\[(][^\])]*[\])]")

_model = None
_model_lock = threading.Lock()


def warm():
    """Load the STT model (first call downloads it to the HF cache).
    Called at startup while the greeting plays, so the first real
    utterance doesn't pay the load."""
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel
            log(f"[ears] loading {CFG['stt_model']} "
                f"({CFG['stt_device']}/{CFG['stt_compute']})...")
            _model = WhisperModel(CFG["stt_model"],
                                  device=CFG["stt_device"],
                                  compute_type=CFG["stt_compute"])
            log("[ears] model ready")
    return _model


def transcribe(pcm: np.ndarray) -> str:
    """int16 mono 16kHz -> text. Bracketed non-speech markers that
    whisper emits ([BLANK_AUDIO], [SIGHS], (coughs)...) are stripped;
    if nothing remains, it was silence."""
    model = warm()
    audio = pcm.astype(np.float32) / 32768.0
    segments, _ = model.transcribe(audio, temperature=0.0, language="en"
                                   if CFG["stt_model"].endswith(".en")
                                   else None)
    text = "".join(s.text for s in segments).strip()
    return _NONSPEECH.sub("", text).strip()


class Ears:
    def __init__(self, aggressiveness: int = 2, silence_ms: int = 480):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.silence_frames = silence_ms // FRAME_MS

    def listen_once(self, gate=None, timeout_s: float | None = None,
                    abort=None) -> str | None:
        """Block until one utterance completes; return transcript
        (or None on timeout). An `abort` callable is checked every
        frame; returning True closes the mic and returns None, which
        is how a live switch back to push-to-talk shuts the open mic
        down promptly instead of after one more utterance."""
        frames: list[np.ndarray] = []
        ring: list[np.ndarray] = []   # pre-roll so the first syllable survives
        speech_run = 0
        silence_run = 0
        speech_total = 0
        in_utterance = False
        elapsed = 0.0

        with _input_stream() as stream:
            while True:
                block, _ = stream.read(FRAME_LEN)
                elapsed += FRAME_MS / 1000
                if abort and abort():
                    return None
                if timeout_s and elapsed > timeout_s and not in_utterance:
                    return None
                mono = block[:, 0].copy()
                if gate and gate():
                    # speakers are talking and barge-in isn't on: ignore
                    ring.clear()
                    continue
                is_speech = self.vad.is_speech(mono.tobytes(), RATE)
                if not in_utterance:
                    ring.append(mono)
                    if len(ring) > 8:
                        ring.pop(0)
                    speech_run = speech_run + 1 if is_speech else 0
                    if speech_run >= OPEN_FRAMES:
                        in_utterance = True
                        frames = ring[:]
                        silence_run = 0
                else:
                    frames.append(mono)
                    if is_speech:
                        speech_total += 1
                        silence_run = 0
                    else:
                        silence_run += 1
                    if silence_run >= self.silence_frames or \
                       len(frames) * FRAME_MS / 1000 > MAX_UTTER_S:
                        if speech_total < 8:
                            # <240ms of actual speech: a noise blip, not
                            # a sentence — keep listening
                            in_utterance = False
                            frames, ring = [], []
                            speech_run = speech_total = 0
                            continue
                        return transcribe(np.concatenate(frames))


def _input_stream() -> sd.InputStream:
    """Open the mic, surviving a device switch mid-session.

    Field-caught: PortAudio caches the device list when it initializes,
    and a Bluetooth headset flipping between listen and mic modes
    (AirPods do this every time the mic opens) invalidates the cached
    device. Every capture after that dies with -9986 until the process
    restarts, so the voice line looks alive and never hears another
    word. Re-initializing refreshes the list; one retry recovers.
    """
    opts = dict(samplerate=RATE, channels=1, dtype="int16",
                blocksize=FRAME_LEN)
    try:
        return sd.InputStream(**opts)
    except sd.PortAudioError:
        print("[ears] audio device changed — reopening the mic", flush=True)
        try:
            sd._terminate()
        except sd.PortAudioError:
            pass  # already torn down; re-initializing is the part that counts
        sd._initialize()
        return sd.InputStream(**opts)


def record_held(is_held, max_s: float = 60.0, min_s: float = 0.25) -> str | None:
    """Hold-to-talk capture: record raw audio while is_held() is True,
    then transcribe. The button is the VAD — no endpointing. Returns
    None for taps shorter than min_s (accidental presses)."""
    frames: list[np.ndarray] = []
    with _input_stream() as stream:
        while is_held() and len(frames) * FRAME_MS / 1000 < max_s:
            block, _ = stream.read(FRAME_LEN)
            frames.append(block[:, 0].copy())
        # a small tail so the last word isn't clipped at release
        for _ in range(6):
            block, _ = stream.read(FRAME_LEN)
            frames.append(block[:, 0].copy())
    if len(frames) * FRAME_MS / 1000 < min_s:
        return None
    return transcribe(np.concatenate(frames))


if __name__ == "__main__":
    import time
    print("[ears] listening — say something...", flush=True)
    ears = Ears()
    start = time.time()
    while time.time() - start < 30:
        text = ears.listen_once(timeout_s=30 - (time.time() - start))
        if text:
            print(f"[ears] heard: {text!r}", flush=True)
            break
        if text is None:
            print("[ears] timed out with no speech", flush=True)
            break
        print("[ears] (noise/empty — still listening)", flush=True)
