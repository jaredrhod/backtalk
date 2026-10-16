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
"""The ears — mic capture with VAD endpointing, transcribed in-process
by faster-whisper. Local, free, no server, no API key.

record_held() is the hold-to-talk capture (the button is the VAD).
Ears.listen_once() is the legacy open-mic mode: blocks until one
complete utterance is heard, then returns its transcript. Endpointing:
an utterance opens after ~120ms of sustained speech, closes after
`silence_ms` of trailing quiet. A `gate` callable can suppress
listening (so the open mic ignores the speakers unless barge-in is on).
"""
import importlib
import os
import platform
import re
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

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
# Hands-free loudness floor, in int16 RMS, that a captured "speech" run
# must clear before whisper sees it. 0 disables the gate.
MIN_SPEECH_RMS = 200

_NONSPEECH = re.compile(r"[\[(][^\])]*[\])]")


def _wake_re(phrase: str) -> re.Pattern:
    """A tolerant matcher for one wake phrase.

    Whisper does not punctuate consistently ("Hey, Seyon!"), so allow any
    run of spaces/punctuation between the words. \\b on the edges keeps
    the name from matching inside a longer word ("seyonic").

    English only on purpose. Wake phrases used to include Tamil forms,
    with a virama-tolerant matcher to catch whisper's spelling of them;
    that machinery is gone -- the wake phrase is Latin, and a Tamil-script
    phrase no longer matches anything.
    """
    words = [re.escape(w) for w in phrase.lower().split()]
    if not words:
        # A blank phrase must match nothing. Without this the pattern is
        # "\b\b", which matches any word boundary -- i.e. every utterance.
        return re.compile(r"(?!)")
    return re.compile(r"\b" + r"[\s,.\-!?]*".join(words) + r"\b",
                      re.IGNORECASE)


def is_wake(text: str, phrases) -> bool:
    """True if `text` contains any wake phrase. Tolerates punctuation and
    capitalisation; variants cover whisper's spellings of "Seyon"."""
    return any(_wake_re(p).search(text or "") for p in phrases)


def strip_wake(text: str, phrases) -> str:
    """Remove the first wake phrase from `text`, preserving what follows
    as spoken. So "Hey Seyon, what's the weather" leaves "what's the
    weather" (casing intact) and a bare "hey seyon" leaves ""."""
    for p in phrases:
        m = _wake_re(p).search(text or "")
        if m:
            joined = text[:m.start()] + " " + text[m.end():]
            # Collapse the hole the phrase leaves ("x seyon y" -> "x  y").
            return re.sub(r"\s+", " ", joined).strip(" ,.!?")
    return (text or "").strip()


_model: Any = None
_model_lock = threading.Lock()
_backend = None          # "mlx" once the GPU path loads, else "faster-whisper"
_LAST_NO_SPEECH = 0.0    # no_speech score of the last transcript, for logs


def _apple_gpu_available() -> bool:
    """Apple Silicon only. CTranslate2, the runtime under faster-whisper,
    has no Metal backend, so on every Mac it transcribes on the CPU while
    the GPU sits idle. mlx-whisper runs the SAME model on the GPU.

    Measured on an M4 Max, small.en, a 6.5s clip, warm: 0.88s on the CPU
    path against 0.12s on the GPU, with a character-identical transcript
    on three of four test clips and a two-comma difference on the fourth.

    Not a second product and not a user-facing choice: same model name
    from the same config key, same text out, one platform finally running
    it properly. Anything that is not an Apple Silicon Mac keeps
    faster-whisper, which already uses CUDA wherever it exists."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    # Deliberately unused: this is an availability probe, and an ImportError
    # is the answer. Kept as importlib so the intent is obvious.
    try:
        importlib.import_module("mlx_whisper")
    except ImportError:
        return False
    return True


def _mlx_repo(model_name: str) -> str:
    """A faster-whisper model name -> its MLX conversion on the Hub."""
    return f"mlx-community/whisper-{model_name}-mlx"


def _resolve_stt_model() -> str:
    """The best STT model that is actually on this machine.

    A bigger model only helps if its weights are already downloaded, and
    asking for an absent one stalls the first utterance on a fetch. So a
    preferred model is used only once its files are present locally, which
    means accuracy rises the moment a download finishes, with no config
    edit and no risk of a slow first turn.
    """
    want = str(CFG.get("stt_model") or "small.en")
    better = CFG.get("stt_model_if_cached")
    if not better or better == want:
        return want
    if _stt_cached(better):
        return better
    return want


def _hf_cache_dir(name: str) -> Path:
    """Where faster-whisper keeps a model's weights.

    The downloader writes them flat into models--<repo>; huggingface's own
    loader instead uses snapshots/<hash>/. Both count as downloaded, and
    both spell the repo differently -- "medium" is really
    Systran/faster-whisper-medium -- so check every form.
    """
    root = Path(os.path.expanduser(
        os.path.join("~", ".cache", "huggingface", "hub")))
    name = str(name)
    repos = []
    if "/" in name:
        repos.append(name)
    else:
        base = name[:-3] if name.endswith(".en") else name
        repos.append(f"Systran/faster-whisper-{base}")
        repos.append(f"Systran/faster-whisper-{name}")
    for repo in repos:
        folder = "models--" + repo.replace("/", "--")
        flat = root / folder
        if (flat / "model.bin").exists() and (flat / "config.json").exists():
            return flat
        snap = root / folder / "snapshots"
        if snap.is_dir():
            for child in snap.iterdir():
                if ((child / "model.bin").exists()
                        and (child / "config.json").exists()):
                    return child
    return root / ("models--" + repos[0].replace("/", "--"))


def _stt_cached(name: str) -> bool:
    d = _hf_cache_dir(name)
    return ((d / "model.bin").exists() and (d / "config.json").exists()
            and (d / "tokenizer.json").exists())


_stt_cache = None


def _stt() -> str:
    """Resolved once: the model name is fixed for the life of the process."""
    global _stt_cache
    if _stt_cache is None:
        _stt_cache = _resolve_stt_model()
    return _stt_cache


_mic_checked = False


_mic_device_warned = False


def _mic_index():
    """Resolve mic_device (a device NAME) to an index, or None for the default.

    A NAME and never an index, because indices shift every time a device
    connects or disconnects, which is the exact event this setting exists
    to survive. Measured on a real machine: plugging a USB microphone in
    moved the default pair from [-1, 1] to [1, 3], silently changing the
    OUTPUT device too.

    Re-resolved on every stream open rather than cached at startup, for
    the same reason. Exact name wins, then the first case-insensitive
    substring, so a precise name can never be beaten by a loose one.
    """
    global _mic_device_warned
    want = str(CFG.get("mic_device", "") or "").strip()
    if not want:
        return None
    try:
        devices = sd.query_devices()
    except Exception as e:
        log(f"[ears] could not list audio devices ({e}) -- using the "
            f"default mic")
        return None
    ins = [(i, d) for i, d in enumerate(devices)
           if d.get("max_input_channels", 0) > 0]
    for i, d in ins:
        if d["name"] == want:
            _mic_device_warned = False
            return i
    low = want.lower()
    for i, d in ins:
        if low in d["name"].lower():
            _mic_device_warned = False
            return i
    if not _mic_device_warned:          # once per disappearance, not per press
        _mic_device_warned = True
        log(f"[ears] mic_device {want!r} not found -- using the system "
            f"default. Inputs I can see: {[d['name'] for _, d in ins]}")
    return None


def _open_mic():
    """Open the capture stream on the configured mic.

    Degrades to the system default if that device will not open --
    unplugged between the lookup and the open, busy, or refusing the
    sample rate. The mic gets worse; it never goes mute.
    """
    dev = _mic_index()
    opts = dict(samplerate=RATE, channels=1, dtype="int16",
                blocksize=FRAME_LEN)
    try:
        return sd.InputStream(device=dev, **opts)
    except Exception as e:
        if dev is not None:
            log(f"[ears] could not open mic_device {CFG.get('mic_device')!r} "
                f"({e}) -- using the system default")
            try:
                return sd.InputStream(**opts)
            except Exception:
                pass                   # fall through to the rebuild below
        return _reopen_after_device_change(opts)


def _reopen_after_device_change(opts):
    """Last resort: rebuild the audio system, then open the mic once more.

    PortAudio caches the device list when it initialises, so a device that
    disappears afterwards leaves a stale entry behind. A Bluetooth headset
    flipping between listening and call modes does this every time the mic
    opens, and from then on EVERY capture fails while the voice line looks
    perfectly healthy and simply never hears another word.

    Rebuilding refreshes the list. It also closes every open stream, the
    speaking one included, which is why Mouth._get_out rebuilds a stream
    it finds dead rather than trusting the one it is holding. Do not
    remove that guard without removing this.
    """
    log("[ears] the audio devices changed -- rebuilding and reopening")
    try:
        sd._terminate()                # pyright: ignore[reportPrivateUsage]
    except Exception:
        pass                           # already down; re-initialising is the point
    sd._initialize()                   # pyright: ignore[reportPrivateUsage]
    return sd.InputStream(**opts)


_mic_warned = False

# Substrings PortAudio uses when the problem is the DEVICE rather than the
# audio. Matched on the message because the exception TYPE is the same
# PortAudioError whether a device vanished or a stream merely glitched.
_DEVICE_ERROR_HINTS = ("error querying device", "invalid device",
                       "device unavailable", "no default input",
                       "invalid number of channels", "device not found")


def _mic_message(detail: str) -> list[str]:
    """The one explanation, so startup and mid-session say the same thing."""
    return [
        "[ears] NO WORKING MICROPHONE. Nothing can be recorded on this "
        "machine, so the talk key will have nothing to send.",
        f"[ears] the audio system said: {detail}",
        "[ears] plug one in and start the voice line again. If one IS "
        "plugged in, check it is allowed in this system's microphone "
        "privacy settings -- and if you have several, put part of the "
        "one you want in \"mic_device\" in backtalk.json.",
    ]


def explain_audio_failure(exc) -> bool:
    """Turn a device-level audio failure into plain words. Returns True
    when it handled the message, so the caller can skip the raw repr.

    The startup pre-flight cannot cover a microphone that is unplugged or
    dies MID-SESSION, and that person gets the worst version of this:
    no warning at all, and a raw PortAudioError on every single press,
    forever. The key hook keeps working throughout, so it still looks
    like it is listening. This says the same sentences the pre-flight
    would have said, at the moment it becomes true.

    Said in full once, then briefly, because a message repeated on every
    key press stops being information and becomes noise.
    """
    global _mic_warned
    text = str(exc).lower()
    # THE TWO HALVES OF THIS TEST ARE NOT DOING THE SAME JOB. Do not
    # simplify it to one. Measured on Windows: the SAME missing microphone
    # produces "Error querying device -1" when it is absent at startup and
    # "A device ID has been used that is out of range for your system
    # [MME error 2]" when it is unplugged mid-stream. The second matches
    # not one hint below, and was caught only by the type check -- on the
    # very first real test of the case this function exists for. The
    # hints catch device failures raised as something other than a
    # PortAudioError; the type catches PortAudio wording nobody predicted.
    if type(exc).__name__ != "PortAudioError" and \
            not any(h in text for h in _DEVICE_ERROR_HINTS):
        return False
    if _mic_warned:
        log("[ears] still no working microphone.")
        return True
    _mic_warned = True
    for line in _mic_message(f"{type(exc).__name__}: {exc}"):
        log(line)
    return True


def check_microphone() -> bool:
    """Say whether recording is possible at all, BEFORE the greeting.

    Without this the voice line boots on a machine with no microphone,
    warms, speaks its greeting and presents a working push-to-talk
    prompt. The key hook works perfectly throughout, so the user is
    given every impression it is listening -- and the only sign of
    trouble is a raw PortAudioError AFTER they have held the key and
    spoken. It then repeats forever, because holding a key again cannot
    conjure a device.
    """
    global _mic_checked
    if _mic_checked:
        return True
    _mic_checked = True
    try:
        sd.check_input_settings(device=_mic_index(), channels=1,
                                samplerate=RATE, dtype="int16")
        return True
    except Exception as e:
        global _mic_warned
        _mic_warned = True      # said it here; do not repeat on first press
        for line in _mic_message(str(e)):
            log(line)
        return False


def _probe(model):
    """Run a tenth of a second of silence through the real path.

    faster-whisper is lazy: transcribe() returns a generator and does no
    work until it is iterated, so the list() is what actually exercises
    the backend and is not redundant.
    """
    probe_lang = str(CFG.get("stt_language") or "").strip().lower() or None
    segments, _ = model.transcribe(np.zeros(RATE // 10, dtype=np.float32),
                                   language=probe_lang or "en")
    list(segments)


def warm():
    """Load the STT model (first call downloads it to the HF cache).
    Called at startup while the greeting plays, so the first real
    utterance doesn't pay the load."""
    global _model, _backend
    check_microphone()
    with _model_lock:
        if _model is None:
            if _apple_gpu_available():
                import mlx_whisper  # pyright: ignore[reportMissingImports]
                repo = _mlx_repo(_stt())
                log(f"[ears] loading {_stt()} on the Apple GPU...")
                # This API has no separate load call: the first transcribe
                # pulls and caches the weights. Warm on a beat of silence so
                # the first real utterance does not pay for it.
                mlx_whisper.transcribe(np.zeros(RATE // 10, dtype=np.float32),
                                       path_or_hf_repo=repo, language="en",
                                       verbose=None)
                _model, _backend = repo, "mlx"
            else:
                from faster_whisper import WhisperModel
                want = CFG["stt_device"]
                log(f"[ears] loading {_stt()} "
                    f"({want}/{CFG['stt_compute']})...")
                _model = WhisperModel(_stt(), device=want,
                                      compute_type=CFG["stt_compute"])
                # PROVE the device before the greeting, not at the first
                # spoken sentence. WhisperModel CONSTRUCTS perfectly well
                # against a GPU it cannot actually use: "auto" picks CUDA
                # on any NVIDIA machine, and the CUDA runtime is not
                # loaded until the first inference. So warm-up logged
                # "model ready", startup reported healthy, and a missing
                # cublas DLL only surfaced when the user finally spoke --
                # long after the greeting, in a place they could not
                # connect to a setting. The Apple-GPU branch above has
                # always done this; this one never did.
                try:
                    _probe(_model)
                except Exception as e:
                    if want == "cpu":
                        raise
                    log(f"[ears] {want!r} does not work on this machine "
                        f"({type(e).__name__}: {e}).")
                    log("[ears] falling back to the CPU. Set "
                        "\"stt_device\": \"cpu\" in backtalk.json to skip "
                        "this check in future.")
                    _model = WhisperModel(_stt(), device="cpu",
                                          compute_type=CFG["stt_compute"])
                    _probe(_model)
                _backend = "faster-whisper"
            log(f"[ears] model ready ({_backend})")
    return _model


def transcribe(pcm: np.ndarray, gate_no_speech: bool = False) -> str:
    """int16 mono 16kHz -> text. Bracketed non-speech markers that
    whisper emits ([BLANK_AUDIO], [SIGHS], (coughs)...) are stripped;
    if whatever remains is silence."""
    return transcribe_language(pcm, gate_no_speech)[0]


def transcribe_language(pcm: np.ndarray,
                        gate_no_speech: bool = False) -> tuple[str, str | None]:
    """transcribe(), plus the language whisper itself detected.

    The code is None only when a caller forces English. The detection is
    reported for logging and tests; it no longer picks a voice -- Seyon
    answers in English whatever the ear heard, so nothing upstream keys
    off this any more.

    gate_no_speech is the hands-free path. With it, a clip the model
    itself scores as mostly non-speech is blanked, so a music bed that
    VAD passed is not handed on as an invented sentence. Hold-to-talk
    leaves it off: it records exactly what the person chose to say.
    """
    model = warm()
    audio = pcm.astype(np.float32) / 32768.0
    # The ".en" models are English-only and cannot detect anything, so
    # they must be TOLD "en". The multilingual ones detect on their own,
    # which is the entire reason they are used.
    forced = str(CFG.get("stt_language") or "").strip().lower() or None
    lang = forced if forced else ("en" if _stt().endswith(".en") else None)
    hint = (CFG.get("stt_prompt") or "").strip() or None
    detected = None
    nsp = 0.0
    if _backend == "mlx":
        import mlx_whisper  # pyright: ignore[reportMissingImports]
        res = mlx_whisper.transcribe(audio, path_or_hf_repo=model,
                                     temperature=0.0, language=lang,
                                     initial_prompt=hint,
                                     verbose=None)
        text = res["text"].strip()
        detected = res.get("language")
        for s in res.get("segments") or []:
            v = s.get("no_speech_prob") if isinstance(s, dict) \
                else getattr(s, "no_speech_prob", None)
            if v is not None:
                nsp = max(nsp, float(v))
    else:
        segments, info = model.transcribe(
            audio, temperature=0.0, language=lang, initial_prompt=hint,
            # A single utterance must never be decoded "in context" of the
            # previous one: that is what makes whisper repeat or invent
            # text. And a segment it judges to be mostly non-speech is
            # dropped outright -- the ambient-noise path that once
            # hallucinated a wake phrase + command.
            condition_on_previous_text=False, no_speech_threshold=0.8)
        segs = list(segments)
        text = "".join(s.text for s in segs).strip()
        detected = getattr(info, "language", None)
        for s in segs:
            nsp = max(nsp, float(getattr(s, "no_speech_prob", 0.0) or 0.0))
    global _LAST_NO_SPEECH
    _LAST_NO_SPEECH = nsp
    text = _NONSPEECH.sub("", text).strip()
    if text and gate_no_speech and not speech_is_confident(nsp):
        text = ""      # the model itself says this was never speech
    return text, detected or lang or None


def _rms(frame: np.ndarray) -> float:
    """Loudness of one int16 frame, as amplitude rather than power."""
    a = frame.astype(np.float32)
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0


def speech_is_audible(mean_speech_rms: float) -> bool:
    """True when captured "speech" is loud enough to be real, not room hum.

    VAD answers "is this shaped like speech", and a fan, a music bed, or
    the tail of a TV clip all pass that test; whisper then transcribes
    the noise into invented sentences ("Thanks for watching!"). This is
    the loudness test VAD does not make. Hands-free only -- the
    hold-to-talk path records exactly what the person chose to record and
    is never gated. A floor of 0 disables it.
    """
    floor = float(CFG.get("stt_min_rms", MIN_SPEECH_RMS) or 0)
    return floor <= 0 or mean_speech_rms >= floor


def speech_is_confident(no_speech_prob: float) -> bool:
    """False when the model itself judges a clip to be non-speech.

    VAD and the loudness floor both pass a music bed; whisper then
    sometimes prints confident text over it ("Thanks for watching!"). Its
    own no_speech score is the honest tell, and this is the check that
    believes it -- the score whisper's internal 0.8 filter lets through
    because the text was confident. Hands-free only; 0 disables it.
    """
    cutoff = float(CFG.get("stt_no_speech_prob", 0.0) or 0.0)
    return cutoff <= 0 or no_speech_prob < cutoff


def _min_speech_frames() -> int:
    """Hands-free minimum speech length before whisper, in VAD frames.

    A real phrase is longer than a cough or a one-word splash of noise.
    Config is in milliseconds, and there is always a 240ms floor so a
    short deliberate command still works but a cough is not one."""

    ms = float(CFG.get("stt_min_speech_ms", 0) or 0)
    return max(8, int(ms / FRAME_MS))


PARTIAL_MIN_FRAMES = 6      # ~180ms of audio before a partial is worth it


def _partial_interval() -> float:
    """Seconds between partial transcripts while a person is still talking.

    Config is in milliseconds; a floor keeps it from firing faster than the
    model can return, which would only queue partials behind the audio."""
    ms = float(CFG.get("partial_stt_ms", 700) or 0)
    return max(0.2, ms / 1000)


class PartialStream:
    """Partial transcripts of an utterance that is still being recorded.

    The capture loop hands every frame to push(); a background ticker calls
    tick() on an interval. tick() decides, purely, whether the buffer is
    worth transcribing yet and emits the growing text, so a face can caption
    the words as they are HEARD rather than after the whole turn. The
    throttle is the point: every pass is a whisper call, and a partial that
    fires faster than the model returns only piles work behind the audio.

    push and reset are called from the capture thread while tick runs on
    the ticker thread, so all three take a lock and tick transcribes a
    snapshot taken under it. That keeps the buffer from being cleared out
    from under a tick, and keeps the capture loop from ever waiting on the
    model (the transcribe runs with the lock released)."""

    def __init__(self, transcribe, emit, interval: float | None = None,
                 min_frames: int = PARTIAL_MIN_FRAMES,
                 clock: Callable[[], float] = time.monotonic):
        self._transcribe = transcribe
        self._emit = emit
        self._interval = _partial_interval() if interval is None else interval
        self._min_frames = min_frames
        self._clock = clock
        self._lock = threading.Lock()
        self._frames: list[np.ndarray] = []
        self._last_at = 0.0
        self._last_text = ""
        self._first = True
        self._gen = 0

    def push(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frames.append(frame)

    def reset(self) -> None:
        """Forget the audio so far -- used when a capture turns out to be
        noise, so the next real utterance starts a fresh partial."""
        with self._lock:
            self._frames.clear()
            self._last_text = ""
            self._first = True
            self._gen += 1

    def tick(self) -> str | None:
        """One partial transcribe, if the buffer is ripe. Returns the text
        emitted, else None. An unchanged reading is never re-emitted."""
        with self._lock:
            if len(self._frames) < self._min_frames:
                return None
            now = self._clock()
            if not self._first and now - self._last_at < self._interval:
                return None
            self._first = False
            self._last_at = now
            frames = list(self._frames)
            last = self._last_text
            gen = self._gen
        # Transcribe outside the lock: the capture thread must never block
        # on whisper, and frames is a private snapshot so reset cannot pull
        # the buffer out from under us.
        text = self._transcribe(np.concatenate(frames))
        if text and text != last:
            with self._lock:
                if self._gen != gen:
                    # A reset landed while we were decoding (a dropped blip
                    # or a non-owner): those words are not for the bus, and
                    # they must not poison the next utterance's first tick.
                    return None
                self._last_text = text
            self._emit(text)
            return text
        return None


class PartialTicker:
    """Drives a PartialStream from its own thread.

    The capture loop must never wait on whisper, so the partials run here:
    frames keep filling on the capture thread while this one transcribes
    snapshots. It is stopped and joined before the final transcript, so the
    model is never asked to decode two clips at once."""

    def __init__(self, stream: PartialStream, poll: float = 0.05):
        self._stream = stream
        self._poll = poll
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="bt-partial",
                                        daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._poll):
            self._stream.tick()

    def stop(self) -> None:
        """Idempotent: safe to call from both the success path and the
        finally, which is how every other exit is covered."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)



class Ears:
    def __init__(self, aggressiveness: int = 3, silence_ms: int = 300):
        # 3 is the strictest level: it takes clearer speech to open the
        # mic, which is what keeps a fan/AC/TV room from feeding whisper
        # a near-silence stream to hallucinate over. Drop to 2 if a
        # quiet or distant voice stops waking it.
        self.vad = webrtcvad.Vad(aggressiveness)
        self.silence_frames = silence_ms // FRAME_MS

    def listen_once(self, gate=None, timeout_s: float | None = None,
                    abort=None, on_partial=None,
                    owner_check=None) -> str | None:
        """Block until one utterance completes; return transcript
        (or None on timeout). An `abort` callable is checked every
        frame; returning True closes the mic and returns None, which
        is how a live switch back to push-to-talk shuts the open mic
        down promptly instead of after one more utterance.

        on_partial, when given, receives growing partial transcripts of the
        utterance while it is still being spoken (see PartialStream), so a
        face can caption the words live. It is stopped before the final
        transcription so the model decodes one clip at a time.

        owner_check, when given, is handed the finished utterance's audio
        and must return True for the owner. A False drops the utterance as
        if it were a noise blip and listening continues; a None means the
        gate is off and every voice is taken."""
        frames: list[np.ndarray] = []
        ring: list[np.ndarray] = []   # pre-roll so the first syllable survives
        speech_run = 0
        silence_run = 0
        speech_total = 0
        speech_rms_sum = 0.0
        in_utterance = False
        elapsed = 0.0

        partial = (PartialStream(transcribe, on_partial)
                   if on_partial is not None else None)
        ticker = PartialTicker(partial) if partial is not None else None
        if ticker is not None:
            ticker.start()
        try:
            with _open_mic() as stream:
                while True:
                    block, _ = stream.read(FRAME_LEN)
                    elapsed += FRAME_MS / 1000
                    if abort and abort():
                        return None
                    if timeout_s and elapsed > timeout_s and not in_utterance:
                        return None
                    mono = block[:, 0].copy()
                    if gate and gate():
                        # speakers are talking and barge-in isn't on: ignore.
                        # Reset the endpointer too, so a gate that rises
                        # mid-utterance cannot splice pre-gate and post-gate
                        # audio into one clip when it lifts.
                        ring.clear()
                        frames = []
                        in_utterance = False
                        speech_run = speech_total = 0
                        speech_rms_sum = 0.0
                        if partial is not None:
                            partial.reset()
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
                            if partial is not None:
                                for f in frames:
                                    partial.push(f)
                            silence_run = 0
                    else:
                        frames.append(mono)
                        if partial is not None:
                            partial.push(mono)
                        if is_speech:
                            speech_total += 1
                            speech_rms_sum += _rms(mono)
                            silence_run = 0
                        else:
                            silence_run += 1
                        if silence_run >= self.silence_frames or \
                           len(frames) * FRAME_MS / 1000 > MAX_UTTER_S:
                            mean_rms = speech_rms_sum / speech_total \
                                if speech_total else 0.0
                            min_frames = _min_speech_frames()
                            reject = None
                            if speech_total < min_frames or \
                               not speech_is_audible(mean_rms):
                                # Too short to be a phrase, or too quiet to be
                                # a person: a noise blip, not a sentence. Keep
                                # listening rather than hand it to whisper,
                                # which would invent words over the hum.
                                reject = "noise"
                            elif owner_check is not None and \
                                    not owner_check(np.concatenate(frames)):
                                # Someone else in the room. The words are not
                                # for this line: drop the utterance and keep
                                # listening, exactly like a noise blip.
                                reject = "other"
                            if reject is not None:
                                if reject == "noise" and \
                                   speech_total >= min_frames:
                                    log(f"[ears] dropped a {mean_rms:.0f}-rms "
                                        f"blip as room noise")
                                elif reject == "other":
                                    log("[ears] that was not the owner -- "
                                        "ignoring")
                                in_utterance = False
                                frames, ring = [], []
                                speech_run = speech_total = 0
                                speech_rms_sum = 0.0
                                if partial is not None:
                                    partial.reset()
                                continue
                            if ticker is not None:
                                ticker.stop()
                            return transcribe(np.concatenate(frames),
                                              gate_no_speech=True)
        finally:
            if ticker is not None:
                ticker.stop()

    def wait_for_wake(self, phrases, gate=None, abort=None,
                      on_partial=None, owner_check=None) -> str | None:
        """Block until an utterance contains a wake phrase; return it.

        Reuses the same VAD endpointer as listen_once, so a wake-word
        turn costs no more than a normal one. Returns None only when
        `abort` fires; otherwise it keeps the mic open indefinitely,
        which is the point of hands-free. The returned transcript may
        carry the command in the same breath ("hey seyon, what's it
        doing"), and the caller strips the phrase with strip_wake().

        owner_check gates the wake on the speaker: an utterance that is
        not the enrolled owner is dropped and listening continues, so
        the room's voices cannot wake the line."""
        while True:
            text = self.listen_once(gate=gate, abort=abort,
                                    on_partial=on_partial,
                                    owner_check=owner_check)
            if text is None:
                return None
            if is_wake(text, phrases):
                return text
            if not text:
                continue          # blanked as non-speech; nothing to report
            # Tuning hook: every heard-but-not-wake utterance goes to the
            # log, with the model's own no-speech score, so a session can
            # teach the phrase variants with real data instead of guesses.
            log(f"[ears] heard {text!r} - no wake phrase match "
                f"(nsp={_LAST_NO_SPEECH:.2f})")

    def record_clip(self, seconds: float = 3.0) -> np.ndarray:
        """Record a fixed-length clip from the mic, for voice enrollment.

        No VAD on purpose: enrollment wants exactly this many seconds of
        the speaker, silence and all, so the print sees the real voice
        rather than whichever fragments a gate chose to keep."""
        frames: list[np.ndarray] = []
        n = max(1, int(seconds * RATE / FRAME_LEN))
        with _open_mic() as stream:
            for _ in range(n):
                block, _ = stream.read(FRAME_LEN)
                frames.append(block[:, 0].copy())
        return np.concatenate(frames)


def record_held(is_held, max_s: float = 60.0, min_s: float = 0.25,
                on_release: Callable[[], None] | None = None,
                on_partial=None) -> str | None:
    """Hold-to-talk capture: record raw audio while is_held() is True,
    then transcribe. The button is the VAD — no endpointing. Returns
    None for taps shorter than min_s (accidental presses).

    on_partial, when given, shows the words as they are spoken during the
    hold (see PartialStream) instead of only after the key comes up."""
    frames: list[np.ndarray] = []
    partial = (PartialStream(transcribe, on_partial)
               if on_partial is not None else None)
    ticker = PartialTicker(partial) if partial is not None else None
    if ticker is not None:
        ticker.start()
    try:
        with _open_mic() as stream:
            while is_held() and len(frames) * FRAME_MS / 1000 < max_s:
                block, _ = stream.read(FRAME_LEN)
                mono = block[:, 0].copy()
                frames.append(mono)
                if partial is not None:
                    partial.push(mono)
            # Fires the instant the key comes up -- before the tail frames
            # and the transcription, not after them. The person already
            # thinks of themselves as done talking, so the UI moves on with
            # them. Reaching the time cap fires it too: the turn proceeds
            # either way, and leaving the UI waiting would be the wrong story.
            if on_release is not None:
                on_release()
            # a small tail so the last word isn't clipped at release
            for _ in range(6):
                block, _ = stream.read(FRAME_LEN)
                frames.append(block[:, 0].copy())
    finally:
        if ticker is not None:
            ticker.stop()
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
