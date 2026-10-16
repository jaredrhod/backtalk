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
"""The mouth — streaming sentence-chunked TTS, played through one
long-lived output stream.

Default engine: Kokoro, in-process. Local, free, no server, no API key,
~0.2s to first audio once warm. Optional premium engine: ElevenLabs on
YOUR key — read from the system keychain, never from a file (see
_get_elevenlabs_key) — with Kokoro as the automatic fallback: the voice
degrades instead of going mute if the cloud fails.

Sentences are synthesized one at a time and queued for playback, so the
first sentence is audible while later ones are still rendering. Playback
is cancellable mid-word: set the stop event and the speaker goes silent
within one audio block plus the device buffer (~0.15s).

HARD-WON AUDIO LAW #1 — ONE long-lived OutputStream, reused for every
sentence for the life of the process. A fresh stream per sentence gives
an audible onset blip or a beat of dead air on plenty of audio setups
(USB interfaces, Bluetooth, streaming mixers that latch onto each new
stream late). Proven by A/B test; do not "simplify" this away.

HARD-WON AUDIO LAW #2 — buffer ~0.75s of synthesized audio before a
sentence starts playing, so a slower machine never underruns into
slow-motion garble.
"""
import io
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading

import numpy as np
import sounddevice as sd

from backtalk.config import CFG
from backtalk.vlog import log

KOKORO_RATE = 24000
EL_RATE = 44100
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

_pipe = None
_pipes: dict = {}  # kokoro pipeline per language letter, loaded on demand
_pipe_lock = threading.Lock()


def _ensure_espeak():
    """kokoro phonemizes through system espeak-ng (its bundled loader
    ships a broken build path — found the hard way; upstream's own docs
    say install the system package). Help phonemizer find it in the
    usual homes when the env isn't already set."""
    if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        return
    candidates = (
        "/opt/homebrew/lib/libespeak-ng.dylib",       # macOS arm64 (brew)
        "/usr/local/lib/libespeak-ng.dylib",          # macOS intel (brew)
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",  # debian/ubuntu
        "/usr/lib/libespeak-ng.so.1",                 # other linux
        "C:\\Program Files\\eSpeak NG\\libespeak-ng.dll",       # windows
        "C:\\Program Files (x86)\\eSpeak NG\\libespeak-ng.dll",
    )
    for lib in candidates:
        if os.path.exists(lib):
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = lib
            break


# Every espeak library filename phonemizer might copy, on any platform. A
# directory holding exactly one of these and nothing else is a phonemizer
# scratch dir and is not plausibly anything else.
_ESPEAK_LIB_NAMES = (
    "espeak-ng.dll",
    "libespeak-ng.dll",
    "libespeak-ng.so",
    "libespeak-ng.so.1",
    "libespeak-ng.dylib",
)


def _is_orphan_espeak_tempdir(path: str) -> bool:
    """True only for a directory whose ENTIRE contents are one espeak
    library. That signature is what makes it safe to point a delete at a
    shared temp folder: one file, and its name is one of five."""
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    return len(entries) == 1 and entries[0] in _ESPEAK_LIB_NAMES


def _sweep_orphan_espeak_tempdirs():
    """Delete espeak scratch dirs left behind by previous runs.

    phonemizer copies the espeak shared library into a fresh temp dir for
    every backend it builds, because espeak-ng keeps its state in globals
    and the loader refuses the same file twice. Kokoro builds several
    backends, so ONE start leaves several behind.

    On POSIX that cleanup rides a finalizer and usually happens. On
    Windows phonemizer can only register it with atexit, and atexit does
    not run when a process is KILLED rather than exited -- so anything
    stopping the voice line by terminating it, which is most launchers and
    every supervisor, leaks every directory it ever made. Sixty had piled
    up on the machine where this was found, and fifteen were sitting on
    the author's own Mac when it was reviewed: the POSIX path is not as
    reliable as it looks either. The count only ever grows.

    Patching phonemizer where it is installed is not a fix, because the
    launcher runs a dependency sync that would overwrite it. Sweeping at
    our own startup bounds the total at one run's worth instead.

    Two things make deleting from a shared temp folder safe, and only the
    first is ours: the signature above is narrow enough that nothing else
    matches it, and anything we are not permitted to remove raises and is
    skipped. On Windows a loaded library cannot be deleted at all, so a
    live instance is protected by the OS rather than by us noticing it.
    POSIX does not work that way, but a process that has already mapped
    the library keeps it after the unlink, so a running instance is
    unharmed either way.
    """
    root = tempfile.gettempdir()
    swept = 0
    try:
        names = os.listdir(root)
    except OSError:
        return
    for name in names:
        path = os.path.join(root, name)
        if not os.path.isdir(path) or not _is_orphan_espeak_tempdir(path):
            continue
        try:
            shutil.rmtree(path)
            swept += 1
        except OSError:
            pass          # in use, or not ours. Leaving it is correct.
    if swept:
        log(f"[mouth] swept {swept} orphaned espeak temp dir(s)")


_turn_hint = None  # language of the current turn


def set_turn_language(lang: str | None) -> None:
    """Record the language of this turn. English is the only value it
    carries now; the hook stays because main() and the warm path read it
    and callers already exist."""
    global _turn_hint
    _turn_hint = (lang or None)


def _turn_lang() -> str | None:
    return _turn_hint


def _voices() -> dict:
    v = CFG.get("voices") or {}
    return v if isinstance(v, dict) and v else {"en": CFG.get("voice") or "bm_lewis"}


def voice_for(lang: str | None) -> str:
    """The voice for a language, falling back to the English default.

    A language with no configured voice is read in English rather than in
    some other language's accent -- Kokoro has no multilingual voice, so a
    mismatch would mangle the words instead of just sounding foreign.
    """
    table = _voices()
    if lang and lang in table:
        return table[lang]
    return table.get("en") or CFG.get("voice") or "bm_lewis"


# --- espeak-ng fallback -------------------------------------------------
# English (Kokoro) is the one voice, and espeak-ng stands behind it: it
# covers ~100 languages, sounds robotic next to a neural voice, so it
# speaks only when Kokoro itself cannot load or fails mid-sentence. It
# is also what tests use to render a non-English sentence for the ear --
# input, never a reply voice.
_ESPEAK = shutil.which("espeak-ng") or next(
    (p for p in (r"C:\Program Files\eSpeak NG\espeak-ng.exe",
                 r"C:\Program Files (x86)\eSpeak NG\espeak-ng.exe",
                 "/usr/bin/espeak-ng", "/opt/homebrew/bin/espeak-ng")
     if os.path.exists(p)), None)

# Whisper's two-letter codes, mapped to espeak-ng's voice names. They mostly
# agree; these are the ones that do not, plus the non-Latin cases where
# espeak-ng wants a base voice rather than a region (zh->cmn is Mandarin).
_ESPEAK_VOICE = {
    "zh": "cmn", "yue": "yue", "nb": "nn", "nn": "nn",
    "he": "he", "iw": "he", "jv": "jv",
}
_espeak_rate = 22050


def espeak_voice_for(lang: str | None) -> str | None:
    """The espeak-ng voice for a language, or None if it cannot speak it."""
    if not lang or not _ESPEAK:
        return None
    code = lang.lower().replace("_", "-").split("-")[0]
    # The table wins BEFORE the length check, including for three-letter
    # codes: whisper reports 'yue' for Cantonese, espeak-ng speaks it as
    # 'yue', and the mapping says so. The old length check ran first and
    # rejected it, so that entry could never be reached -- Cantonese fell
    # all the way back to a Kokoro English accent instead.
    if code in _ESPEAK_VOICE:
        return _ESPEAK_VOICE[code]
    if len(code) == 3:            # whisper sometimes reports 'tam', 'hin'
        return None               # let espeak guess from the full code instead
    return code


def _stream_espeak(text: str, lang: str | None):
    """One sentence -> int16 PCM at espeak's rate, via a UTF-8 temp file.

    Passing non-Latin text as a command-line argument or on stdin produces
    SILENCE on Windows (measured: Tamil, Chinese, Japanese, Korean, Arabic,
    Russian and Thai all came back with peak amplitude 0 and 0.35s of empty
    audio; only German, being Latin-1, worked through argv). Writing the
    sentence to a UTF-8 file and using -f is the one form that works for
    every script -- hence the temp file rather than the obvious subprocess
    call.
    """
    import wave as _wave
    voice = espeak_voice_for(lang)
    if not (voice and text.strip()):
        return
    espeak = _ESPEAK
    if not espeak:
        return
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="bt-espeak-")
    try:
        # os.fdopen is inside the try on purpose: if it itself raises, the
        # raw descriptor from mkstemp would never be closed.
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        try:
            rate = float(CFG.get("speed") or 1.0)
        except (TypeError, ValueError):
            rate = 1.0
        wpm = int(160 / max(rate, 0.5))
        proc = subprocess.run(
            [espeak, "-v", voice, "-s", str(wpm), "-f", path, "--stdout"],
            capture_output=True, timeout=60)
        if not proc.stdout:
            return
        with _wave.open(io.BytesIO(proc.stdout), "rb") as w:
            if w.getnchannels() > 1:      # espeak can emit stereo; we play mono
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                pcm = pcm.reshape(-1, w.getnchannels()).mean(axis=1).astype(np.int16)
            else:
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            sr = w.getframerate()
        if pcm.size:
            global _espeak_rate
            _espeak_rate = sr
            yield pcm
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def warm(lang: str | None = None) -> str:
    """Load (and cache) the Kokoro pipeline for a language's voice.

    Returns the voice name to speak with. The voice name's first letter IS
    the language pipeline: a=American English, b=British English,
    e/f/h/i/j/p/z = the other shipped languages. bm_lewis -> 'b'.

    Pipelines are cached per language letter -- the table is generic on
    purpose, even though English is the only reply voice today; loading
    pays its startup cost once per letter, never per sentence.
    """
    global _pipe
    voice = voice_for(lang)
    code = (voice or "bm_lewis")[0]
    with _pipe_lock:
        if _pipe is None:
            _ensure_espeak()
            # Before kokoro makes this run's scratch dirs, clear the ones
            # earlier runs could not clean up on their way out.
            _sweep_orphan_espeak_tempdirs()
        if code not in _pipes:
            from kokoro import KPipeline
            log(f"[mouth] loading kokoro (lang '{code}', voice {voice})...")
            _pipes[code] = KPipeline(lang_code=code)
            log("[mouth] voice ready")
        _pipe = _pipes[code]
    return voice


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_RE.split(text.strip()) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _stream_kokoro(text: str):
    """One sentence -> int16 PCM chunks at 24kHz, in-process."""
    voice = warm(_turn_lang())
    pipe = _pipe
    if pipe is None:
        return
    try:
        speed = float(CFG.get("speed") or 1.0)
    except (TypeError, ValueError):
        speed = 1.0
    for _, _, audio in pipe(text, voice=voice, speed=speed):
        a = np.asarray(audio, dtype=np.float32)
        if a.size:
            yield (np.clip(a, -1.0, 1.0) * 32767).astype(np.int16)


def _stream_elevenlabs(text: str, timeout: float):
    """ElevenLabs -> ffmpeg streaming decode -> int16 PCM at 44.1kHz.

    THE ELEVENLABS DOCTRINE, learned the expensive way:
    - fetch mp3_44100_128 and decode locally (raw 44.1k PCM needs their
      Pro tier; the mp3 decode hides inside network wait anyway)
    - turbo model, stability 0.5, similarity 0.75
    - never the multilingual model for English, never style above 0 —
      both make delivery slow and dull
    - their site previews are MASTERED demo clips; raw API output never
      matches them, so master locally (the ffmpeg chain in config)
    ffmpeg reads stdin as we feed it, so playback still starts before
    synthesis finishes."""
    import subprocess

    import httpx

    el = CFG["elevenlabs"]
    key = _get_elevenlabs_key()
    url = (f"https://api.elevenlabs.io/v1/text-to-speech/"
           f"{el['voice_id']}/stream?output_format=mp3_44100_128")
    proc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "quiet", "-i", "pipe:0",
         "-af", el["master"],
         "-f", "s16le", "-ar", str(EL_RATE), "-ac", "1", "pipe:1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stdin = proc.stdin
    stdout = proc.stdout
    if stdin is None or stdout is None:   # impossible with PIPE; narrows
        _end_ffmpeg(proc, stdin)
        return

    feed_error: list = []

    def _feed():
        try:
            with httpx.stream("POST", url, headers={"xi-api-key": key},
                              json={"text": text, "model_id": el["model"],
                                    "voice_settings": {
                                        "stability": 0.5,
                                        "similarity_boost": 0.75}},
                              timeout=timeout) as r:
                r.raise_for_status()
                for chunk in r.iter_bytes(chunk_size=4096):
                    stdin.write(chunk)
        except Exception as e:
            feed_error.append(e)
        finally:
            try:
                stdin.close()
            except Exception:
                pass

    t = threading.Thread(target=_feed, daemon=True)
    t.start()
    carry = b""
    got_audio = False
    try:
        while True:
            data = stdout.read(8820)
            if not data:
                break
            data = carry + data
            usable = len(data) - (len(data) % 2)
            carry = data[usable:]
            if usable:
                got_audio = True
                yield np.frombuffer(data[:usable], dtype=np.int16)
        proc.wait(timeout=10)
    finally:
        _end_ffmpeg(proc, stdin)
    if feed_error and not got_audio:
        raise feed_error[0]


def _end_ffmpeg(proc, stdin) -> None:
    """Reclaim a half-finished synth. A barge-in abandons this generator
    mid-stream (GeneratorExit at the yield); without this, ffmpeg and the
    thread feeding it outlive the sentence. Killing ffmpeg also closes the
    pipe the feeder is blocked on, so that daemon thread ends too."""
    try:
        stdin.close()
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


_el_key_cache: str | None = None


def _key_slot() -> str:
    """The credential-store entry name, so someone who already keeps a key
    under their own name points at it instead of storing a second copy."""
    return str(CFG["elevenlabs"].get("key_slot") or "backtalk-elevenlabs")


def _get_elevenlabs_key() -> str:
    """The API key, from the most secure store available — NEVER from a
    file in this repo. Lookup order:
      1. macOS Keychain, item `backtalk-elevenlabs` by default (change it
         with elevenlabs.key_slot) — seed it once with:
         security add-generic-password -a "$USER" -s backtalk-elevenlabs -T /usr/bin/security -w
         (it prompts for the secret; -T lets this code read it without a
         GUI prompt every launch)
      2. Linux secret-tool (libsecret):
         secret-tool store --label backtalk service backtalk-elevenlabs
      3. the ELEVENLABS_API_KEY environment variable — the last-resort
         fallback, and the only option on Windows for now. Know the
         tradeoff: an export line in a shell profile is a plaintext key
         on disk, which is exactly what the keychain path avoids."""
    global _el_key_cache
    if _el_key_cache is not None:
        return _el_key_cache
    import subprocess
    key = ""
    try:
        if sys.platform == "darwin":
            r = subprocess.run(["security", "find-generic-password",
                                "-s", _key_slot(), "-w"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                key = r.stdout.strip()
        elif sys.platform.startswith("linux"):
            from shutil import which
            if which("secret-tool"):
                r = subprocess.run(["secret-tool", "lookup", "service",
                                    _key_slot()],
                                   capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    key = r.stdout.strip()
    except Exception:
        pass
    _el_key_cache = key or os.environ.get("ELEVENLABS_API_KEY", "")
    return _el_key_cache


def _elevenlabs_ready() -> bool:
    el = CFG["elevenlabs"]
    return bool(el.get("enabled") and el.get("voice_id")
                and _get_elevenlabs_key())


def synth_stream(text: str, timeout: float = 30.0):
    """One sentence -> yields (sample_rate, pcm_chunk) as the TTS renders.

    ENGLISH ONLY: ElevenLabs when configured, Kokoro otherwise, espeak-ng
    if Kokoro itself fails. There is no language routing left to do --
    there is exactly one voice -- so even a reply the model wrote in
    some other script still reaches the English engines rather than a
    foreign accent, and the one outcome worse than a wrong accent,
    silence, stays covered by the espeak-ng fallback at the bottom.
    """
    if _elevenlabs_ready():
        spoke = False
        try:
            for pcm in _stream_elevenlabs(text, timeout):
                spoke = True
                yield EL_RATE, pcm
        except Exception as e:
            log(f"[mouth] elevenlabs failed ({str(e)[:60]}) - "
                f"falling back to {CFG['voice']}")
        if spoke:
            return
        # Empty output with no error is still silence; fall through to Kokoro
        # rather than leave the sentence unsaid.

    got = False
    try:
        for pcm in _stream_kokoro(text):
            got = True
            yield KOKORO_RATE, pcm
        if got:
            return
    except Exception as e:
        log(f"[mouth] kokoro failed ({str(e)[:60]}) - espeak-ng fallback")
    try:
        for pcm in _stream_espeak(text, "en"):
            log("[mouth] espeak-ng (en) speaking english")
            yield _espeak_rate, pcm
            return
    except Exception as e:
        log(f"[mouth] espeak-ng failed ({str(e)[:60]})")
    log("[mouth] no engine could speak this reply")


def _default_output() -> int | None:
    """The index of the system's current default OUTPUT device, or None for
    "whatever the system calls default". Read fresh on every stream open,
    never cached, and that is the point of it.

    On Windows a Bluetooth headset becomes the default output the moment it
    connects and stops being it when it goes away, so FOLLOWING the default
    is exactly the behaviour wanted here -- speak through the Bluetooth when
    it is there, through the normal speakers when it is not -- without
    naming a device or configuring anything. An unreadable default, or a
    lone -1 (Windows for "no device"), reads as None so the system default
    still applies.
    """
    try:
        dev = sd.default.device
    except Exception:
        return None
    out = dev[1] if isinstance(dev, (list, tuple)) and len(dev) > 1 else dev
    if isinstance(out, int) and out >= 0:
        return out
    return None


class Mouth:
    def __init__(self):
        from backtalk.ducking import Ducker
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._speaking = threading.Event()
        # The one persistent output stream (audio law #1).
        # Worker-thread-only — never touch from other threads.
        self._out: sd.OutputStream | None = None
        self._out_rate: int | None = None
        self._out_dev: int | None = None
        self.ducker = Ducker()  # public: PTT ducks for the USER's voice too
        # Barge-in epoch and the count of queued-but-not-yet-finished
        # utterances, both guarded by _lock. The epoch is how a sentence the
        # worker had already pulled -- but not yet started -- learns it was
        # interrupted; the count is how wait_done tells "between two
        # sentences of one reply" apart from "the reply is over" without ever
        # observing the worker's dequeue->playing gap. See _serve/wait_done.
        self._lock = threading.Lock()
        self._epoch = 0
        self._outstanding = 0
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    @property
    def speaking(self) -> bool:
        return self._speaking.is_set()

    def say(self, text: str):
        """Queue text (split to sentences) for speech."""
        for s in split_sentences(text):
            self._put(s, None)

    def say_chunk(self, text: str, directions=None):
        """Queue text as ONE TTS request, no sentence splitting — fuller
        chunks get livelier prosody (single short sentences come out
        dull).

        `directions` are the stage directions this chunk carried. They are
        published on the signal bus when this chunk's audio STARTS, which
        is why they travel with it instead of firing at parse time."""
        text = text.strip()
        if text:
            self._put(text, directions or None)

    def _put(self, sentence: str, directions=None):
        """Enqueue one utterance, stamped with the current barge-in epoch.

        The count and the epoch are read together under the lock, so a
        barge-in can never land in the gap between "the worker took this off
        the queue" and "the worker committed to speaking it": either the
        stamp is the old epoch (the barge-in wins and the item is dropped),
        or the epoch was already bumped and this is genuinely new work."""
        with self._lock:
            self._outstanding += 1
            epoch = self._epoch
        self._q.put((sentence, directions or None, epoch))

    def shut_up(self):
        """Barge-in: stop current playback and flush everything queued.

        Bumping the epoch under the lock is what drops a sentence the worker
        had already dequeued: it carries the old epoch, so when it reaches
        the front of _serve it is discarded unspoken instead of talking over
        the person who interrupted."""
        with self._lock:
            self._epoch += 1
            self._stop.set()
        drained = 0
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
            drained += 1
        if drained:
            self._finish(drained)

    def shutdown(self):
        """Exit path: stop playback and restore the music SYNCHRONOUSLY
        (the debounced restore timer dies with the process otherwise)."""
        self.shut_up()
        self.ducker.restore_now()

    def wait_done(self, timeout: float | None = None):
        """Block until everything queued has been played or dropped.

        The outstanding count, not the queue, is the thing to watch: it is
        raised the moment an utterance is queued and lowered only once it
        has finished, so the instant the worker pulls the last sentence (the
        queue is empty but playback has not started) is not mistaken for
        done. The speaking event stays in the test for the uncounted case."""
        import time
        deadline = None if timeout is None else time.time() + timeout
        while True:
            with self._lock:
                busy = self._outstanding > 0
            if not busy and self._q.empty() and not self._speaking.is_set():
                return
            time.sleep(0.05)
            if deadline and time.time() > deadline:
                return

    def _run(self):
        while True:
            self._serve()

    def _serve(self):
        """Pull one utterance and speak it -- unless a barge-in overtook it."""
        from backtalk import signals
        sentence, directions, epoch = self._q.get()
        if not sentence:
            self._finish(1)
            return
        with self._lock:
            interrupted = epoch < self._epoch
            if not interrupted:
                self._stop.clear()
                self._speaking.set()
        if interrupted:
            self._finish(1)
            return
        self.ducker.speech_start()
        signals.static_stop()     # thinking sound dies when speech starts
        signals.set_state("speaking")
        try:
            self._play_stream(sentence, directions)
        except Exception as e:
            log(f"[mouth] synth/play error: {e}")
        finally:
            self._finish(1)

    def _finish(self, n: int):
        """Book n utterances as done. Emptying the count means the reply has
        genuinely stopped talking, as opposed to the gap between two
        sentences of the same reply -- so that is where "speaking" clears and
        the idle signals fire. Guarded by was_speaking so a barge-in that
        drains queued-but-unstarted work never flashes the screen idle."""
        from backtalk import signals
        with self._lock:
            self._outstanding -= n
            idle = self._outstanding <= 0
            was_speaking = self._speaking.is_set()
            if idle:
                self._speaking.clear()
        if idle and was_speaking:
            # The caption goes only here: the queue drains well after the
            # model is done writing, so clearing it at enqueue time would
            # blank the words while they are still being spoken.
            signals.caption_clear()
            signals.reply_done()
            self.ducker.speech_end()
            signals.set_state("idle")

    def _get_out(self, rate: int) -> sd.OutputStream:
        """The long-lived stream (audio law #1). Reopened when the sample
        rate changes (ElevenLabs 44.1k <-> Kokoro 24k fallback), and when
        the system's default OUTPUT device changes -- so a Bluetooth
        headset that connects mid-session takes the next sentence, and the
        normal speakers take it back once it disconnects."""
        dev = _default_output()
        if (self._out is not None and self._out_rate == rate
                and self._out_dev == dev):
            # Guarded, because the stream can die UNDER us: the ears
            # rebuild the whole audio system to recover from a device
            # change (see ears._reopen_after_device_change), and that
            # closes every open stream including this one. Touching a
            # dead stream raises rather than returning False, so the
            # check has to be the try, not an `if`. Falling through
            # rebuilds it, which is what the rest of this method does.
            try:
                if not self._out.active:
                    self._out.start()
                return self._out
            except Exception:
                log("[mouth] the output stream went away, reopening")
        self._drop_out()
        self._out = sd.OutputStream(samplerate=rate, channels=1,
                                    dtype="int16", device=dev)
        self._out_rate = rate
        self._out_dev = dev
        self._out.start()
        return self._out

    def _cut(self):
        """Barge-in cut: stop feeding audio and pad the line with a beat
        of silence — the stream itself NEVER stops (an abort+restart here
        re-triggers the onset blip on latch-happy audio setups). Cost:
        the device buffer (~0.1s) plays out after the kill order — half a
        syllable of tail."""
        out = self._out
        if out is None:
            return
        try:
            zeros = np.zeros(2205, dtype=np.int16)
            for _ in range(3):
                out.write(zeros)
        except Exception:
            self._drop_out()

    def _drop_out(self):
        """Close and forget the stream — the next sentence reopens
        fresh. The self-heal path for device errors (interface
        unplugged, audio mixer restarted)."""
        if self._out is not None:
            try:
                self._out.close(ignore_errors=True)
            except Exception:
                pass
        self._out = None
        self._out_rate = None
        self._out_dev = None

    def _play_stream(self, sentence: str, directions=None, block: int = 2205,
                     prebuffer_s: float | None = None):
        """Stream-synthesize and play with the head-start buffer (audio
        law #2). stop() reacts ~50ms. The sample rate comes from
        whichever engine actually answered. The head start is configurable
        (prebuffer_s); None takes the configured value, which is small by
        default so the first word is not held back."""
        from backtalk import signals
        if prebuffer_s is None:
            prebuffer_s = float(CFG.get("prebuffer_s", 0.5))
        gen = synth_stream(sentence)
        head: list = []
        banked = 0
        rate = None
        for rate_, pcm in gen:
            rate = rate_
            head.append(pcm)
            banked += len(pcm)
            if banked >= int(rate * prebuffer_s):
                break
        if rate is None:
            return
        try:
            out = self._get_out(rate)
            # AUDIO STARTS HERE: the head buffer is full and the first write
            # is next. Publishing now is what puts a screen cue on the spoken
            # word rather than seconds ahead of it.
            if directions:
                from backtalk import signals as _sig
                _sig.direction(directions)

            def _write(pcm):
                for i in range(0, len(pcm), block):
                    if self._stop.is_set():
                        return False
                    out.write(pcm[i:i + block])
                    # Re-check after the blocking write: a barge-in
                    # landing mid-block must not let feed_waveform
                    # re-assert "speaking" over a fresh "listening".
                    if self._stop.is_set():
                        return False
                    signals.feed_waveform(pcm[i:i + block])
                return True
            for pcm in head:
                if not _write(pcm):
                    self._cut()
                    return
            for _, pcm in gen:
                if not _write(pcm):
                    self._cut()
                    return
        except Exception:
            self._drop_out()
            raise


if __name__ == "__main__":
    m = Mouth()
    m.say(sys.argv[1] if len(sys.argv) > 1 else
          "Voice check. The mouth is alive, and it is very good to be heard.")
    m.wait_done(timeout=60)
