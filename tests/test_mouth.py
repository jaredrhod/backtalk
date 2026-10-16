"""mouth.py offline: the sentence logic, the engine fallback chain, the
playback queue, and the credential lookup -- none of it touching a device.

Kokoro, sounddevice and espeak-ng are all faked or stubbed, so this runs
headless in CI. What it pins is the behaviour that is easy to break and
hard to notice: which engine speaks when the premium one fails, that the
one long-lived output stream is reused, that a barge-in actually cuts the
audio, and that the ElevenLabs key never comes from a file.
"""
from pathlib import Path
import io
import os
import shutil
import sys
import tempfile
import time
import types
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import backtalk.signals as sig  # noqa: E402
from backtalk import mouth  # noqa: E402
from backtalk.mouth import Mouth  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'ok' if cond else 'FAIL'} {name}"
          + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILURES.append(name)


class FakeOut:
    def __init__(self, active=True):
        self.active = active
        self.writes: list = []
        self.started = 0
        self.closed = False

    def start(self):
        self.started += 1

    def write(self, data):
        self.writes.append(data)

    def close(self, ignore_errors=False):
        self.closed = True


class DeadOut:
    """A stream whose .active raises, the way a device gone away does."""
    def __init__(self):
        self.started = 0
        self.closed = False

    @property
    def active(self):
        raise RuntimeError("device gone")

    def start(self):
        self.started += 1

    def write(self, data):
        pass

    def close(self, ignore_errors=False):
        self.closed = True


def quiet_mouth() -> Mouth:
    """A Mouth with no worker loop running, so tests drive the queue."""
    with mock.patch.object(Mouth, "_run", lambda self: None):
        return Mouth()


def sentences():
    print("\n--- split_sentences ---")
    check("splits on punctuation",
          mouth.split_sentences("One. Two! Three?") ==
          ["One.", "Two!", "Three?"], mouth.split_sentences("One. Two!"))
    check("collapses extra whitespace",
          mouth.split_sentences("One.   Two") == ["One.", "Two"])
    check("no punctuation -> one part",
          mouth.split_sentences("just a line") == ["just a line"])
    check("blank -> empty list", mouth.split_sentences("   ") == [])


def voices():
    print("\n--- voice_for / _voices ---")
    with mock.patch.object(mouth, "CFG",
                           {"voices": {"en": "voiceEN", "a": "voiceA"}}):
        check("table entry returned", mouth.voice_for("a") == "voiceA")
        check("unknown language falls back to the table's en",
              mouth.voice_for("zz") == "voiceEN", mouth.voice_for("zz"))
    with mock.patch.object(mouth, "CFG", {"voice": "myvoice"}):
        check("no voices table -> single default",
              mouth._voices() == {"en": "myvoice"}, mouth._voices())
        check("matching language uses the default",
              mouth.voice_for("en") == "myvoice")


def espeak_choices():
    print("\n--- espeak_voice_for ---")
    with mock.patch.object(mouth, "_ESPEAK", "/usr/bin/espeak-ng"):
        check("zh -> cmn (Mandarin, not a region)",
              mouth.espeak_voice_for("zh") == "cmn")
        check("yue -> yue even though it is three letters",
              mouth.espeak_voice_for("yue") == "yue")
        check("nb -> nn", mouth.espeak_voice_for("nb") == "nn")
        check("region stripped: en-US -> en",
              mouth.espeak_voice_for("en-US") == "en")
        check("underscore variant normalised",
              mouth.espeak_voice_for("pt_BR") == "pt")
        check("bare three-letter code -> None (let espeak guess)",
              mouth.espeak_voice_for("tam") is None)
        check("None -> None", mouth.espeak_voice_for(None) is None)
    with mock.patch.object(mouth, "_ESPEAK", ""):
        check("no espeak binary -> None", mouth.espeak_voice_for("en") is None)


def espeak_early_returns():
    print("\n--- _stream_espeak early exits (no binary needed) ---")
    with mock.patch.object(mouth, "_ESPEAK", ""):
        check("no binary -> no audio",
              list(mouth._stream_espeak("hi", "en")) == [])
    with mock.patch.object(mouth, "_ESPEAK", "/x/espeak"), \
         mock.patch.object(mouth, "espeak_voice_for", return_value=None):
        check("language espeak cannot speak -> silence",
              list(mouth._stream_espeak("hi", "xx")) == [])
    with mock.patch.object(mouth, "_ESPEAK", "/x/espeak"), \
         mock.patch.object(mouth, "espeak_voice_for", return_value="en"):
        check("blank text -> silence",
              list(mouth._stream_espeak("   ", "en")) == [])
    with mock.patch.object(mouth, "_ESPEAK", ""), \
         mock.patch.object(mouth, "espeak_voice_for", return_value="en"):
        check("a voice but no binary -> silence",
              list(mouth._stream_espeak("hi", "en")) == [])


def synth_chain():
    print("\n--- synth_stream: the engine fallback chain ---")
    pcm = np.zeros(4, dtype=np.int16)
    with mock.patch.object(mouth, "log"):
        with mock.patch.object(mouth, "_elevenlabs_ready", return_value=True), \
             mock.patch.object(mouth, "_stream_elevenlabs",
                               side_effect=lambda *a, **k: iter([pcm])):
            out = list(mouth.synth_stream("hi"))
        check("elevenlabs used when ready and keyed",
              bool(out) and out[0][0] == mouth.EL_RATE, out)

        with mock.patch.object(mouth, "_elevenlabs_ready", return_value=True), \
             mock.patch.object(mouth, "_stream_elevenlabs",
                               side_effect=lambda *a, **k: iter([])), \
             mock.patch.object(mouth, "_stream_kokoro",
                               side_effect=lambda t: iter([pcm])):
            out = list(mouth.synth_stream("hi"))
        check("empty elevenlabs output falls through to kokoro",
              bool(out) and out[0][0] == mouth.KOKORO_RATE, out)

        with mock.patch.object(mouth, "_elevenlabs_ready", return_value=True), \
             mock.patch.object(mouth, "_stream_elevenlabs",
                               side_effect=RuntimeError("cloud down")), \
             mock.patch.object(mouth, "_stream_kokoro",
                               side_effect=lambda t: iter([pcm])):
            out = list(mouth.synth_stream("hi"))
        check("elevenlabs failure degrades to kokoro",
              bool(out) and out[0][0] == mouth.KOKORO_RATE, out)

        with mock.patch.object(mouth, "_elevenlabs_ready", return_value=False), \
             mock.patch.object(mouth, "_stream_kokoro",
                               side_effect=lambda t: iter([])), \
             mock.patch.object(mouth, "_stream_espeak",
                               side_effect=lambda t, l: iter([pcm])), \
             mock.patch.object(mouth, "_espeak_rate", 22050):
            out = list(mouth.synth_stream("hi"))
        check("empty kokoro falls through to espeak",
              bool(out) and out[0][0] == 22050, out)

        with mock.patch.object(mouth, "_elevenlabs_ready", return_value=False), \
             mock.patch.object(mouth, "_stream_kokoro",
                               side_effect=RuntimeError("no pipe")), \
             mock.patch.object(mouth, "_stream_espeak",
                               side_effect=RuntimeError("no espeak")):
            out = list(mouth.synth_stream("hi"))
        check("every engine failing -> no audio at all", out == [], out)

        with mock.patch.object(mouth, "log") as logged, \
             mock.patch.object(mouth, "_elevenlabs_ready", return_value=False), \
             mock.patch.object(mouth, "_stream_kokoro",
                               side_effect=lambda t: iter([])), \
             mock.patch.object(mouth, "_stream_espeak",
                               side_effect=lambda t, l: iter([])):
            out = list(mouth.synth_stream("hi"))
        check("every engine silent -> no audio", out == [], out)
        check("the silence is admitted in the log", logged.called)


def keys():
    print("\n--- the ElevenLabs key comes from a store, not a file ---")
    cfg = {"elevenlabs": {"key_slot": "custom-slot"}}
    with mock.patch.object(mouth, "CFG", cfg):
        check("key slot honours config",
              mouth._key_slot() == "custom-slot", mouth._key_slot())
    with mock.patch.object(mouth, "CFG", {"elevenlabs": {}}):
        check("key slot default", mouth._key_slot() == "backtalk-elevenlabs")

    mouth._el_key_cache = None
    run = mock.Mock(return_value=mock.Mock(returncode=0, stdout="secret\n"))
    with mock.patch.object(mouth, "CFG", cfg), \
         mock.patch.object(mouth.sys, "platform", "darwin"), \
         mock.patch.object(mouth.subprocess, "run", run):
        got = mouth._get_elevenlabs_key()
    check("macOS keychain path returns the secret", got == "secret", got)
    with mock.patch.object(mouth.subprocess, "run") as run2:
        again = mouth._get_elevenlabs_key()
    check("cached: the store is not read twice",
          again == "secret" and not run2.called, again)

    mouth._el_key_cache = None
    with mock.patch.object(mouth, "CFG", cfg), \
         mock.patch.object(mouth.sys, "platform", "darwin"), \
         mock.patch.object(mouth.subprocess, "run",
                           return_value=mock.Mock(returncode=1, stdout="")), \
         mock.patch.dict(os.environ, {}, clear=True):
        got = mouth._get_elevenlabs_key()
    check("a keychain miss (non-zero) leaves no key", got == "", got)

    mouth._el_key_cache = None
    with mock.patch.object(mouth, "CFG", cfg), \
         mock.patch.object(mouth.sys, "platform", "win32"), \
         mock.patch.object(mouth.subprocess, "run") as run3, \
         mock.patch.dict(os.environ, {}, clear=True):
        got = mouth._get_elevenlabs_key()
    check("off mac and linux there is no store to read",
          got == "" and not run3.called, got)

    mouth._el_key_cache = None
    with mock.patch.object(mouth, "CFG", cfg), \
         mock.patch.object(mouth.sys, "platform", "linux"), \
         mock.patch.object(mouth.shutil, "which", return_value="/bin/secret-tool"), \
         mock.patch.object(mouth.subprocess, "run",
                           return_value=mock.Mock(returncode=1, stdout="")), \
         mock.patch.dict(os.environ, {}, clear=True):
        got = mouth._get_elevenlabs_key()
    check("a secret-tool miss leaves no key", got == "", got)

    mouth._el_key_cache = None
    with mock.patch.object(mouth, "CFG", cfg), \
         mock.patch.object(mouth.sys, "platform", "linux"), \
         mock.patch.object(mouth.subprocess, "run",
                           return_value=mock.Mock(returncode=1, stdout="")):
        os.environ["ELEVENLABS_API_KEY"] = "envkey"
        try:
            got = mouth._get_elevenlabs_key()
        finally:
            os.environ.pop("ELEVENLABS_API_KEY", None)
    check("failed store lookup falls back to the env var",
          got == "envkey", got)

    mouth._el_key_cache = None
    with mock.patch.object(
            mouth, "CFG",
            {"elevenlabs": {"enabled": True, "voice_id": "v", "model": "m"}}), \
         mock.patch.object(mouth, "_get_elevenlabs_key", return_value="k"):
        check("ready with enabled+voice id+key", mouth._elevenlabs_ready())
    with mock.patch.object(mouth, "CFG",
                           {"elevenlabs": {"enabled": True, "model": "m"}}), \
         mock.patch.object(mouth, "_get_elevenlabs_key", return_value="k"):
        check("no voice id -> not ready", not mouth._elevenlabs_ready())
    with mock.patch.object(
            mouth, "CFG",
            {"elevenlabs": {"enabled": True, "voice_id": "v", "model": "m"}}), \
         mock.patch.object(mouth, "_get_elevenlabs_key", return_value=""):
        check("no key -> not ready", not mouth._elevenlabs_ready())
    with mock.patch.object(
            mouth, "CFG",
            {"elevenlabs": {"voice_id": "v", "model": "m"}}), \
         mock.patch.object(mouth, "_get_elevenlabs_key", return_value="k"):
        check("disabled -> not ready", not mouth._elevenlabs_ready())


def sweep():
    print("\n--- orphan espeak scratch dirs ---")
    root = tempfile.mkdtemp(prefix="bt-test-")
    try:
        orphan = os.path.join(root, "orphan")
        os.mkdir(orphan)
        open(os.path.join(orphan, "espeak-ng.dll"), "w").close()
        used = os.path.join(root, "used")
        os.mkdir(used)
        open(os.path.join(used, "espeak-ng.dll"), "w").close()
        open(os.path.join(used, "other.txt"), "w").close()
        empty = os.path.join(root, "empty")
        os.mkdir(empty)
        plain = os.path.join(root, "note.txt")
        open(plain, "w").close()
        check("one-library dir is an orphan",
              mouth._is_orphan_espeak_tempdir(orphan))
        check("two-file dir is not",
              not mouth._is_orphan_espeak_tempdir(used))
        check("empty dir is not", not mouth._is_orphan_espeak_tempdir(empty))
        check("missing path is not",
              not mouth._is_orphan_espeak_tempdir(
                  os.path.join(root, "nope")))
        with mock.patch.object(mouth.tempfile, "gettempdir",
                               return_value=root), \
             mock.patch.object(mouth, "log"):
            mouth._sweep_orphan_espeak_tempdirs()
        check("orphan removed", not os.path.exists(orphan))
        check("dir with another file kept", os.path.exists(used))
        check("plain file kept", os.path.exists(plain))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def queue_behaviour():
    print("\n--- the Mouth queue (no worker) ---")
    m = quiet_mouth()
    m.say("One. Two. Three.")
    got = []
    while not m._q.empty():
        got.append(m._q.get_nowait())
    check("say splits into sentences",
          [g[0] for g in got] == ["One.", "Two.", "Three."], got)

    m.say_chunk("  a single chunk  ", ["wave"])
    item = m._q.get_nowait()
    check("say_chunk keeps one request and strips it",
          item[:2] == ("a single chunk", ["wave"]), item)
    m.say_chunk("   ")
    check("blank say_chunk queues nothing", m._q.empty())

    m.say("One. Two.")
    m.shut_up()
    check("shut_up flushes the queue", m._q.empty())
    check("shut_up raises the stop flag", m._stop.is_set())

    m2 = quiet_mouth()
    check("speaking reflects the event", not m2.speaking)
    m2._speaking.set()
    check("speaking set", m2.speaking)

    m3 = quiet_mouth()
    t0 = time.time()
    m3.wait_done(timeout=0.2)
    check("wait_done returns at once on an empty queue",
          time.time() - t0 < 0.15, time.time() - t0)

    m4 = quiet_mouth()
    m4.ducker = mock.Mock()
    m4.shutdown()
    check("shutdown restores the music immediately",
          m4.ducker.restore_now.called)


def worker_loop():
    print("\n--- the worker loop parks the bus when speech drains ---")
    played = []
    with mock.patch.object(Mouth, "_play_stream",
                           new=lambda self, s, d=None: played.append((s, d))), \
         mock.patch.object(sig, "static_stop"), \
         mock.patch.object(sig, "set_state") as set_state, \
         mock.patch.object(sig, "caption"), \
         mock.patch.object(sig, "caption_clear") as cap_clear, \
         mock.patch.object(sig, "reply_done") as reply_done, \
         mock.patch.object(sig, "feed_waveform"), \
         mock.patch.object(sig, "direction"):
        m = Mouth()
        m.say_chunk("Hello there.", ["wave"])
        deadline = time.time() + 5
        while not reply_done.called and time.time() < deadline:
            time.sleep(0.01)
        m.shut_up()
    check("the sentence was handed to playback",
          played == [("Hello there.", ["wave"])], played)
    check("speaking was announced", set_state.called)
    states = [c.args[0] for c in set_state.call_args_list]
    check("bus goes speaking then idle",
          states[:1] == ["speaking"] and states[-1:] == ["idle"], states)
    check("caption cleared and reply marked done",
          cap_clear.called and reply_done.called)


def streams():
    print("\n--- the one long-lived output stream ---")
    m = quiet_mouth()
    made = []

    def new_stream(**k):
        s = FakeOut()
        made.append(s)
        return s

    with mock.patch.object(mouth.sd, "OutputStream", side_effect=new_stream):
        a = m._get_out(24000)
        b = m._get_out(24000)
        c = m._get_out(44100)
    check("reused while the rate is unchanged", a is b)
    check("reopened when the rate changes", c is not a)
    check("exactly two streams built for two rates", len(made) == 2, len(made))

    m2 = quiet_mouth()
    inactive = FakeOut(active=False)
    with mock.patch.object(mouth.sd, "OutputStream",
                           side_effect=lambda **k: inactive):
        x = m2._get_out(24000)
        y = m2._get_out(24000)
    check("an inactive stream is restarted, not rebuilt",
          x is y and inactive.started >= 2, inactive.started)

    m3 = quiet_mouth()
    dead, fresh = DeadOut(), FakeOut()
    seq = iter([dead, fresh])
    with mock.patch.object(mouth.sd, "OutputStream",
                           side_effect=lambda **k: next(seq)), \
         mock.patch.object(mouth, "log"):
        first = m3._get_out(24000)
        second = m3._get_out(24000)
    check("a stream that raises .active is dropped and replaced",
          first is dead and second is fresh)
    check("the dead stream is closed on the way out", dead.closed)

    # A changed default output (a Bluetooth headset connecting, or leaving)
    # must move the next sentence onto the new device.
    m4 = quiet_mouth()
    made4 = []

    def new_stream4(**k):
        s = FakeOut()
        s.device = k.get("device")
        made4.append(s)
        return s

    defaults = iter([10, 10, 20])  # the default flips on the third open
    with mock.patch.object(mouth, "_default_output",
                           side_effect=lambda: next(defaults)), \
         mock.patch.object(mouth.sd, "OutputStream", side_effect=new_stream4):
        m4._get_out(24000)
        m4._get_out(24000)
        m4._get_out(24000)
    check("a changed default device reopens the stream", len(made4) == 2,
          len(made4))
    check("the rebuilt stream targets the new default",
          made4[-1].device == 20, made4[-1].device)


def default_output():
    print("\n--- _default_output: which speaker the system means ---")

    class Holder:
        def __init__(self, device):
            self.device = device

    def resolve(device):
        with mock.patch.object(mouth.sd, "default", Holder(device)):
            return mouth._default_output()

    check("an (input, output) pair yields the output index",
          resolve((1, 4)) == 4, resolve((1, 4)))
    check("a lone index is the output", resolve(4) == 4, resolve(4))
    check("a one-item sequence is not an output", resolve((4,)) is None)
    check("Windows' -1 reads as no device", resolve((1, -1)) is None)
    check("a non-index output reads as no device",
          resolve(("a", "b")) is None)

    class Boom:
        @property
        def device(self):
            raise RuntimeError("no default")

    with mock.patch.object(mouth.sd, "default", Boom()):
        check("an unreadable default reads as no device",
              mouth._default_output() is None)


def teardown_paths():
    print("\n--- _cut / _drop_out ---")
    m = quiet_mouth()
    m._out = FakeOut()
    m._cut()
    check("cut writes three beats of silence",
          len(m._out.writes) == 3, len(m._out.writes))
    quiet_mouth()._cut()  # no stream: must not raise
    check("cut with no stream is safe", True)

    m2 = quiet_mouth()
    fake = FakeOut()
    m2._out, m2._out_rate = fake, 24000
    m2._drop_out()
    check("drop closes and forgets the stream",
          fake.closed and m2._out is None and m2._out_rate is None)


def playback():
    print("\n--- _play_stream: buffer, write, publish, cut ---")

    def one_chunk(text, timeout=30.0):
        yield (24000, np.zeros(24000, dtype=np.int16))

    m = quiet_mouth()
    fake = FakeOut()
    with mock.patch.object(mouth, "synth_stream", side_effect=one_chunk), \
         mock.patch.object(Mouth, "_get_out", return_value=fake), \
         mock.patch.object(sig, "feed_waveform") as feed, \
         mock.patch.object(sig, "direction") as direction, \
         mock.patch.object(sig, "set_state"), \
         mock.patch.object(sig, "caption"), \
         mock.patch.object(sig, "caption_clear"), \
         mock.patch.object(sig, "reply_done"):
        m._play_stream("Hello there.", ["wave"])
    check("audio was written to the stream", len(fake.writes) > 0)
    check("waveform fed to the bus", feed.called)
    check("directions published when audio starts",
          direction.called and direction.call_args.args[0] == ["wave"])

    m2 = quiet_mouth()
    m2._stop.set()
    cut = []
    with mock.patch.object(mouth, "synth_stream", side_effect=one_chunk), \
         mock.patch.object(Mouth, "_get_out", return_value=FakeOut()), \
         mock.patch.object(Mouth, "_cut",
                           lambda self: cut.append(True)), \
         mock.patch.object(sig, "feed_waveform"), \
         mock.patch.object(sig, "direction"), \
         mock.patch.object(sig, "set_state"), \
         mock.patch.object(sig, "caption"), \
         mock.patch.object(sig, "caption_clear"), \
         mock.patch.object(sig, "reply_done"):
        m2._play_stream("Hello there.")
    check("a raised stop flag cuts before any audio", bool(cut) and True)

    m3 = quiet_mouth()
    with mock.patch.object(mouth, "synth_stream",
                           side_effect=lambda t, timeout=30.0: iter([])):
        m3._play_stream("nothing at all")
    check("no synthesized audio -> returns quietly", m3._out is None)


def turn_language():
    print("\n--- set_turn_language / _turn_lang ---")
    saved = mouth._turn_lang()
    mouth.set_turn_language("fr")
    check("a language is recorded", mouth._turn_lang() == "fr")
    mouth.set_turn_language(None)
    check("None clears it", mouth._turn_lang() is None)
    mouth.set_turn_language("")
    check("an empty string is None, not ''", mouth._turn_lang() is None)
    mouth.set_turn_language(saved)


def ensure_espeak():
    print("\n--- _ensure_espeak ---")
    with mock.patch.dict(os.environ,
                         {"PHONEMIZER_ESPEAK_LIBRARY": "/already"}):
        mouth._ensure_espeak()
        check("an already-set library is left alone",
              os.environ["PHONEMIZER_ESPEAK_LIBRARY"] == "/already")

    env = {k: v for k, v in os.environ.items()
           if k != "PHONEMIZER_ESPEAK_LIBRARY"}
    with mock.patch.dict(os.environ, env, clear=True), \
         mock.patch.object(mouth.os.path, "exists",
                           side_effect=lambda p: p.endswith(
                               "libespeak-ng.dll")):
        mouth._ensure_espeak()
        check("a found library is exported",
              os.environ.get("PHONEMIZER_ESPEAK_LIBRARY", "")
              .endswith("libespeak-ng.dll"))
    with mock.patch.dict(os.environ, env, clear=True), \
         mock.patch.object(mouth.os.path, "exists", return_value=False):
        mouth._ensure_espeak()
        check("no candidate leaves the env unset",
              "PHONEMIZER_ESPEAK_LIBRARY" not in os.environ)


def sweep_errors():
    print("\n--- the sweep tolerates a hostile temp dir ---")
    with mock.patch.object(mouth.os, "listdir",
                           side_effect=OSError("nope")):
        mouth._sweep_orphan_espeak_tempdirs()
    check("an unreadable temp dir is skipped, not fatal", True)

    root = tempfile.mkdtemp(prefix="bt-sweep-")
    try:
        orphan = os.path.join(root, "orphan")
        os.mkdir(orphan)
        open(os.path.join(orphan, "espeak-ng.dll"), "w").close()
        with mock.patch.object(mouth.tempfile, "gettempdir",
                               return_value=root), \
             mock.patch.object(mouth.shutil, "rmtree",
                               side_effect=OSError("in use")), \
             mock.patch.object(mouth, "log"):
            mouth._sweep_orphan_espeak_tempdirs()
        check("a dir that will not delete is left alone",
              os.path.exists(orphan))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def espeak_render():
    print("\n--- _stream_espeak renders a WAV to PCM ---")
    import wave as _wave

    def wav(samples, channels, rate=22050):
        buf = io.BytesIO()
        with _wave.open(buf, "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(np.asarray(samples, dtype=np.int16).tobytes())
        return buf.getvalue()

    mono = wav([100, -100, 200], 1)
    with mock.patch.object(mouth, "_ESPEAK", "espeak-ng"), \
         mock.patch.object(mouth.subprocess, "run",
                           return_value=mock.Mock(stdout=mono)), \
         mock.patch.object(mouth, "CFG", {"speed": "fast"}):
        out = list(mouth._stream_espeak("hi", "en"))
    check("mono audio is yielded as int16",
          out and out[0].tolist() == [100, -100, 200], out)

    stereo = wav([100, 200, -100, -200], 2)
    with mock.patch.object(mouth, "_ESPEAK", "espeak-ng"), \
         mock.patch.object(mouth.subprocess, "run",
                           return_value=mock.Mock(stdout=stereo)), \
         mock.patch.object(mouth, "CFG", {"speed": 1.0}):
        out = list(mouth._stream_espeak("hi", "en"))
    check("stereo is mixed to mono",
          out and out[0].tolist() == [150, -150], out)

    with mock.patch.object(mouth, "_ESPEAK", "espeak-ng"), \
         mock.patch.object(mouth.subprocess, "run",
                           return_value=mock.Mock(stdout=b"")), \
         mock.patch.object(mouth, "CFG", {"speed": 1.0}):
        check("empty stdout yields nothing",
              list(mouth._stream_espeak("hi", "en")) == [])

    silent = wav([], 1)
    with mock.patch.object(mouth, "_ESPEAK", "espeak-ng"), \
         mock.patch.object(mouth.subprocess, "run",
                           return_value=mock.Mock(stdout=silent)), \
         mock.patch.object(mouth, "CFG", {"speed": 1.0}):
        check("a zero-frame WAV yields nothing",
              list(mouth._stream_espeak("hi", "en")) == [])

    with mock.patch.object(mouth, "_ESPEAK", "espeak-ng"), \
         mock.patch.object(mouth.subprocess, "run",
                           return_value=mock.Mock(stdout=mono)), \
         mock.patch.object(mouth.os, "remove",
                           side_effect=OSError("in use")), \
         mock.patch.object(mouth, "CFG", {"speed": 1.0}):
        out = list(mouth._stream_espeak("hi", "en"))
    check("audio still returned when the temp file will not delete",
          out and out[0].tolist() == [100, -100, 200], out)


def warm_and_kokoro():
    print("\n--- warm() / _stream_kokoro ---")
    fake_kokoro = types.ModuleType("kokoro")
    built = []

    class FakeKPipeline:
        def __init__(self, lang_code):
            built.append(lang_code)

    fake_kokoro.KPipeline = FakeKPipeline
    mouth._pipes.clear()
    mouth._pipe = None
    with mock.patch.object(mouth, "CFG", {"voice": "bm_lewis"}), \
         mock.patch.object(mouth, "_ensure_espeak"), \
         mock.patch.object(mouth, "_sweep_orphan_espeak_tempdirs") as swept, \
         mock.patch.dict(sys.modules, {"kokoro": fake_kokoro}), \
         mock.patch.object(mouth, "log"):
        v1 = mouth.warm(None)
        v2 = mouth.warm(None)
    check("warm returns the configured voice", v1 == "bm_lewis", v1)
    check("the pipeline is built once per language letter",
          built == ["b"], built)
    check("the second warm is served from the cache", v2 == "bm_lewis")
    check("the orphan sweep runs as the pipe is first built", built and True)

    class FakePipe:
        def __call__(self, text, voice=None, speed=None):
            yield (None, None, np.array([0.5, -0.5], dtype=np.float32))

    with mock.patch.object(mouth, "warm", return_value="v"), \
         mock.patch.object(mouth, "_pipe", None):
        check("no pipeline -> no audio",
              list(mouth._stream_kokoro("hi")) == [])
    with mock.patch.object(mouth, "warm", return_value="bm_lewis"), \
         mock.patch.object(mouth, "_pipe", _FloatPipe()), \
         mock.patch.object(mouth, "CFG", {"speed": "fast"}):
        chunks = list(mouth._stream_kokoro("hi"))
    check("kokoro audio is clipped to int16",
          chunks and chunks[0].dtype == np.int16
          and chunks[0].tolist() == [16383, -16383], chunks)

    class _EmptyPipe:
        def __call__(self, text, voice=None, speed=None):
            yield (None, None, np.array([], dtype=np.float32))

    with mock.patch.object(mouth, "warm", return_value="bm_lewis"), \
         mock.patch.object(mouth, "_pipe", _EmptyPipe()), \
         mock.patch.object(mouth, "CFG", {"speed": 1.0}):
        chunks = list(mouth._stream_kokoro("hi"))
    check("an empty kokoro chunk yields nothing", chunks == [], chunks)


class _FloatPipe:
    def __call__(self, text, voice=None, speed=None):
        yield (None, None, np.array([0.5, -0.5], dtype=np.float32))


def key_linux():
    print("\n--- _get_elevenlabs_key on linux ---")
    clean = {k: v for k, v in os.environ.items() if k != "ELEVENLABS_API_KEY"}
    mouth._el_key_cache = None
    with mock.patch.dict(os.environ, clean, clear=True), \
         mock.patch.object(mouth.sys, "platform", "linux"), \
         mock.patch.object(mouth.shutil, "which", return_value="/bin/secret-tool"), \
         mock.patch.object(mouth.subprocess, "run",
                           return_value=mock.Mock(returncode=0,
                                                  stdout="sekret\n")):
        check("secret-tool supplies the key",
              mouth._get_elevenlabs_key() == "sekret")

    mouth._el_key_cache = None
    with mock.patch.dict(os.environ, clean, clear=True), \
         mock.patch.object(mouth.sys, "platform", "linux"), \
         mock.patch.object(mouth.shutil, "which", return_value=None):
        check("no secret-tool -> empty", mouth._get_elevenlabs_key() == "")

    mouth._el_key_cache = None
    with mock.patch.dict(os.environ, {**clean, "ELEVENLABS_API_KEY": "envk"},
                         clear=True), \
         mock.patch.object(mouth.sys, "platform", "linux"), \
         mock.patch.object(mouth.shutil, "which", return_value="/bin/secret-tool"), \
         mock.patch.object(mouth.subprocess, "run",
                           side_effect=RuntimeError("boom")):
        check("a store that raises falls back to the env",
              mouth._get_elevenlabs_key() == "envk")


def barge_in():
    print("\n--- a barge-in cannot lose a sentence mid-handoff ---")
    played = []
    with mock.patch.object(Mouth, "_play_stream",
                           new=lambda self, s, d=None: played.append(s)), \
         mock.patch.object(sig, "static_stop"), \
         mock.patch.object(sig, "set_state"), \
         mock.patch.object(sig, "caption"), \
         mock.patch.object(sig, "caption_clear"), \
         mock.patch.object(sig, "reply_done") as reply_done, \
         mock.patch.object(sig, "feed_waveform"), \
         mock.patch.object(sig, "direction"):
        # The worker had already pulled this sentence when the barge-in
        # landed: stamped with the old epoch, it must be dropped unspoken.
        m = quiet_mouth()
        m._put("stale.", None)
        m._epoch = 1                    # shut_up's bump, but it was pulled
        m._stop.set()
        m._serve()
        check("a sentence caught mid-handoff is dropped, not spoken",
              played == [], played)

        # A fresh, post-barge-in sentence must still play.
        m._put("fresh.", None)
        m._serve()
        check("work queued after the barge-in still plays",
              played == ["fresh."], played)

        # Barge in on the last outstanding sentence: the drop must still
        # clear 'speaking' and announce idle, or the mouth hangs "speaking".
        reply_done.reset_mock()
        m2 = quiet_mouth()
        m2._put("said.", None)
        m2._speaking.set()              # mid-sentence
        m2._epoch = 1
        m2._stop.set()
        m2._serve()
        check("dropping the last item clears speaking and goes idle",
              not m2.speaking and reply_done.called)

        # Draining work that never started must NOT flash the screen idle.
        reply_done.reset_mock()
        m2._put("unstarted.", None)
        m2.shut_up()
        check("draining unstarted work stays silent", not reply_done.called)


def wait_done_timeout():
    print("\n--- wait_done gives up at its deadline ---")
    m = quiet_mouth()
    m._speaking.set()      # speaking and never finishing
    t0 = time.time()
    m.wait_done(timeout=0.1)
    m._speaking.clear()
    check("wait_done returns at the deadline",
          time.time() - t0 < 1.0, time.time() - t0)


def worker_edges():
    print("\n--- the worker skips blank items and survives play errors ---")
    played = []
    with mock.patch.object(Mouth, "_play_stream",
                           new=lambda self, s, d=None: played.append(s)), \
         mock.patch.object(sig, "static_stop"), \
         mock.patch.object(sig, "set_state"), \
         mock.patch.object(sig, "caption"), \
         mock.patch.object(sig, "caption_clear"), \
         mock.patch.object(sig, "reply_done") as reply_done, \
         mock.patch.object(sig, "feed_waveform"), \
         mock.patch.object(sig, "direction"):
        m = Mouth()
        m._put("", None)
        m._put("real.", None)
        deadline = time.time() + 5
        while not reply_done.called and time.time() < deadline:
            time.sleep(0.01)
        m.shut_up()
    check("a blank item is skipped, the real one plays",
          played == ["real."], played)

    with mock.patch.object(Mouth, "_play_stream",
                           side_effect=RuntimeError("boom")), \
         mock.patch.object(sig, "static_stop"), \
         mock.patch.object(sig, "set_state"), \
         mock.patch.object(sig, "caption"), \
         mock.patch.object(sig, "caption_clear"), \
         mock.patch.object(sig, "reply_done") as reply_done, \
         mock.patch.object(sig, "feed_waveform"), \
         mock.patch.object(sig, "direction"), \
         mock.patch.object(mouth, "log") as logged:
        m2 = Mouth()
        m2.say_chunk("hello")
        deadline = time.time() + 5
        while not reply_done.called and time.time() < deadline:
            time.sleep(0.01)
        m2.shut_up()
    check("a playback error is logged, not fatal", logged.called)


def teardown_errors():
    print("\n--- _cut / _drop_out on a broken stream ---")
    m = quiet_mouth()
    bad = mock.Mock()
    bad.write.side_effect = RuntimeError("gone")
    m._out, m._out_rate = bad, 24000
    m._cut()
    check("a failing cut drops the stream", m._out is None)

    m2 = quiet_mouth()
    bad2 = mock.Mock()
    bad2.close.side_effect = RuntimeError("gone")
    m2._out, m2._out_rate = bad2, 24000
    m2._drop_out()
    check("a failing close is swallowed", m2._out is None)

    m3 = quiet_mouth()
    bad3 = mock.Mock()
    bad3.write.side_effect = RuntimeError("gone")
    with mock.patch.object(mouth, "synth_stream",
                           side_effect=playback_chunk), \
         mock.patch.object(Mouth, "_get_out", return_value=bad3), \
         mock.patch.object(Mouth, "_drop_out") as drop, \
         mock.patch.object(sig, "feed_waveform"), \
         mock.patch.object(sig, "direction"), \
         mock.patch.object(sig, "set_state"), \
         mock.patch.object(sig, "caption"), \
         mock.patch.object(sig, "caption_clear"), \
         mock.patch.object(sig, "reply_done"):
        raised = False
        try:
            m3._play_stream("Hello.")
        except RuntimeError:
            raised = True
    check("a write failure drops the stream and re-raises",
          raised and drop.called)


def playback_chunk(text, timeout=30.0):
    yield (24000, np.zeros(24000, dtype=np.int16))


def playback_edges():
    print("\n--- _play_stream: prebuffer, extra chunks, mid-write stop ---")

    def two_chunks(text, timeout=30.0):
        yield (24000, np.zeros(24000, dtype=np.int16))
        yield (24000, np.zeros(24000, dtype=np.int16))

    m = quiet_mouth()
    fake = FakeOut()
    with mock.patch.object(mouth, "synth_stream", side_effect=two_chunks), \
         mock.patch.object(Mouth, "_get_out", return_value=fake), \
         mock.patch.object(sig, "feed_waveform"), \
         mock.patch.object(sig, "direction"), \
         mock.patch.object(sig, "set_state"), \
         mock.patch.object(sig, "caption"), \
         mock.patch.object(sig, "caption_clear"), \
         mock.patch.object(sig, "reply_done"):
        m._play_stream("Hello.", ["wave"], prebuffer_s=2.0)
    check("chunks spanning the prebuffer are all written",
          len(fake.writes) >= 2, len(fake.writes))

    m2 = quiet_mouth()
    out2 = FakeOut()
    cut = []

    def write_then_stop(data):
        out2.writes.append(data)
        m2._stop.set()

    out2.write = write_then_stop
    with mock.patch.object(mouth, "synth_stream", side_effect=playback_chunk), \
         mock.patch.object(Mouth, "_get_out", return_value=out2), \
         mock.patch.object(Mouth, "_cut", lambda self: cut.append(True)), \
         mock.patch.object(sig, "feed_waveform"), \
         mock.patch.object(sig, "direction"), \
         mock.patch.object(sig, "set_state"), \
         mock.patch.object(sig, "caption"), \
         mock.patch.object(sig, "caption_clear"), \
         mock.patch.object(sig, "reply_done"):
        m2._play_stream("Hello.")
    check("a stop landing mid-write cuts the line", bool(cut))

    m3 = quiet_mouth()
    out3 = FakeOut()
    seen = {"n": 0}

    def write3(data):
        out3.writes.append(data)
        seen["n"] += 1
        if seen["n"] == 2:
            m3._stop.set()

    out3.write = write3
    cut3 = []

    def small(text, timeout=30.0):
        yield (1000, np.ones(1000, dtype=np.int16))
        yield (1000, np.full(1000, 2, dtype=np.int16))

    m3._stop.clear()
    with mock.patch.object(mouth, "synth_stream", side_effect=small), \
         mock.patch.object(Mouth, "_get_out", return_value=out3), \
         mock.patch.object(Mouth, "_cut", lambda self: cut3.append(True)), \
         mock.patch.object(sig, "feed_waveform"), \
         mock.patch.object(sig, "direction"), \
         mock.patch.object(sig, "set_state"), \
         mock.patch.object(sig, "caption"), \
         mock.patch.object(sig, "caption_clear"), \
         mock.patch.object(sig, "reply_done"):
        m3._play_stream("Hello.", prebuffer_s=1.0)
    check("a stop after the head buffer still cuts", bool(cut3))

    m4 = quiet_mouth()
    out4 = FakeOut()

    def small2(text, timeout=30.0):
        yield (1000, np.ones(1000, dtype=np.int16))
        yield (1000, np.full(1000, 2, dtype=np.int16))

    m4._stop.clear()
    with mock.patch.object(mouth, "synth_stream", side_effect=small2), \
         mock.patch.object(Mouth, "_get_out", return_value=out4), \
         mock.patch.object(sig, "feed_waveform"), \
         mock.patch.object(sig, "direction"), \
         mock.patch.object(sig, "set_state"), \
         mock.patch.object(sig, "caption"), \
         mock.patch.object(sig, "caption_clear"), \
         mock.patch.object(sig, "reply_done"):
        m4._play_stream("Hello.", prebuffer_s=1.0)
    check("every generated chunk plays when nothing stops it",
          len(out4.writes) == 2, len(out4.writes))


def elevenlabs_stream():
    print("\n--- _stream_elevenlabs: fetch, decode, feed ---")

    class FakeStdin:
        def __init__(self, close_boom=False):
            self.written = []
            self.closed = False
            self._close_boom = close_boom

        def write(self, b):
            self.written.append(b)

        def close(self):
            self.closed = True
            if self._close_boom:
                raise OSError("close failed")

    class FakeStdout:
        def __init__(self, data, stdin=None):
            self._data = data
            self._stdin = stdin

        def read(self, n):
            if not self._data:
                for _ in range(500):
                    if self._stdin is None or self._stdin.closed:
                        return b""
                    time.sleep(0.002)
                return b""
            chunk, self._data = self._data[:n], self._data[n:]
            return chunk

    class FakeProc:
        def __init__(self, data, stdin=True, close_boom=False, kill_boom=False):
            self.stdin = FakeStdin(close_boom) if stdin else None
            self.stdout = FakeStdout(data, self.stdin) if stdin else None
            self.killed = False
            self._kill_boom = kill_boom

        def wait(self, timeout=None):
            return 0

        def kill(self):
            self.killed = True
            if self._kill_boom:
                raise OSError("no such process")

    class FakeStream:
        def __init__(self, chunks, fail=False):
            self._chunks = chunks
            self._fail = fail

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            if self._fail:
                raise RuntimeError("http 500")

        def iter_bytes(self, chunk_size=4096):
            for c in self._chunks:
                yield c

    fake_httpx = types.ModuleType("httpx")
    holder = {"stream": FakeStream([])}
    fake_httpx.stream = lambda *a, **k: holder["stream"]

    cfg = {"elevenlabs": {"voice_id": "v", "model": "m",
                          "master": "atempo=1.0"},
           "voice": "fallback"}
    pcm_bytes = np.array([5, -5, 10], dtype=np.int16).tobytes()

    holder["stream"] = FakeStream([pcm_bytes])
    with mock.patch.dict(sys.modules, {"httpx": fake_httpx}), \
         mock.patch.object(mouth, "CFG", cfg), \
         mock.patch.object(mouth, "_get_elevenlabs_key", return_value="k"), \
         mock.patch.object(mouth.subprocess, "Popen",
                           return_value=FakeProc(pcm_bytes)):
        out = list(mouth._stream_elevenlabs("hello", 5.0))
    check("mp3 bytes decode to int16 pcm",
          out and out[0].tolist() == [5, -5, 10], out)

    holder["stream"] = FakeStream([], fail=True)
    with mock.patch.dict(sys.modules, {"httpx": fake_httpx}), \
         mock.patch.object(mouth, "CFG", cfg), \
         mock.patch.object(mouth, "_get_elevenlabs_key", return_value="k"), \
         mock.patch.object(mouth.subprocess, "Popen",
                           return_value=FakeProc(b"")):
        raised = False
        try:
            list(mouth._stream_elevenlabs("hello", 5.0))
        except RuntimeError:
            raised = True
    check("a feed that failed with no audio raises", raised)

    with mock.patch.dict(sys.modules, {"httpx": fake_httpx}), \
         mock.patch.object(mouth, "CFG", cfg), \
         mock.patch.object(mouth, "_get_elevenlabs_key", return_value="k"), \
         mock.patch.object(mouth.subprocess, "Popen",
                           return_value=FakeProc(b"", stdin=False)):
        check("a process with no pipes returns quietly",
              list(mouth._stream_elevenlabs("hello", 5.0)) == [])

    holder["stream"] = FakeStream([pcm_bytes])
    with mock.patch.dict(sys.modules, {"httpx": fake_httpx}), \
         mock.patch.object(mouth, "CFG", cfg), \
         mock.patch.object(mouth, "_get_elevenlabs_key", return_value="k"), \
         mock.patch.object(mouth.subprocess, "Popen",
                           return_value=FakeProc(b"\x05")):
        check("a lone trailing byte yields no chunk",
              list(mouth._stream_elevenlabs("hello", 5.0)) == [])

    holder["stream"] = FakeStream([pcm_bytes])
    with mock.patch.dict(sys.modules, {"httpx": fake_httpx}), \
         mock.patch.object(mouth, "CFG", cfg), \
         mock.patch.object(mouth, "_get_elevenlabs_key", return_value="k"), \
         mock.patch.object(mouth.subprocess, "Popen",
                           return_value=FakeProc(pcm_bytes, close_boom=True)):
        out = list(mouth._stream_elevenlabs("hello", 5.0))
    check("a failing stdin.close is swallowed",
          out and out[0].tolist() == [5, -5, 10], out)

    holder["stream"] = FakeStream([pcm_bytes])
    proc = FakeProc(pcm_bytes)
    with mock.patch.dict(sys.modules, {"httpx": fake_httpx}), \
         mock.patch.object(mouth, "CFG", cfg), \
         mock.patch.object(mouth, "_get_elevenlabs_key", return_value="k"), \
         mock.patch.object(mouth.subprocess, "Popen", return_value=proc):
        gen = mouth._stream_elevenlabs("hello", 5.0)
        next(gen)
        gen.close()
    check("abandoning the stream mid-way kills ffmpeg", proc.killed)

    holder["stream"] = FakeStream([pcm_bytes])
    with mock.patch.dict(sys.modules, {"httpx": fake_httpx}), \
         mock.patch.object(mouth, "CFG", cfg), \
         mock.patch.object(mouth, "_get_elevenlabs_key", return_value="k"), \
         mock.patch.object(mouth.subprocess, "Popen",
                           return_value=FakeProc(pcm_bytes, kill_boom=True)):
        out = list(mouth._stream_elevenlabs("hello", 5.0))
    check("a kill that fails is swallowed",
          out and out[0].tolist() == [5, -5, 10], out)


def worker_keeps_talking():
    print("\n--- the worker keeps talking while the queue has more ---")
    played = []

    def play(self, s, d=None):
        played.append(s)
        if len(played) == 1:
            self._put("chaser.", None)

    with mock.patch.object(Mouth, "_play_stream", new=play), \
         mock.patch.object(sig, "static_stop"), \
         mock.patch.object(sig, "set_state"), \
         mock.patch.object(sig, "caption"), \
         mock.patch.object(sig, "caption_clear") as cap_clear, \
         mock.patch.object(sig, "reply_done") as reply_done, \
         mock.patch.object(sig, "feed_waveform"), \
         mock.patch.object(sig, "direction"):
        m = Mouth()
        m.say_chunk("lead.")
        deadline = time.time() + 5
        while not reply_done.called and time.time() < deadline:
            time.sleep(0.01)
        m.shut_up()
    check("the queued follow-on played too",
          played == ["lead.", "chaser."], played)
    check("the bus parked only after both", cap_clear.called)


print("=" * 66)
print("THE MOUTH (offline, no device, no model)")
print("=" * 66)
sentences()
voices()
espeak_choices()
espeak_early_returns()
synth_chain()
keys()
sweep()
queue_behaviour()
worker_loop()
streams()
default_output()
teardown_paths()
playback()
turn_language()
ensure_espeak()
sweep_errors()
espeak_render()
warm_and_kokoro()
key_linux()
wait_done_timeout()
barge_in()
worker_edges()
teardown_errors()
playback_edges()
elevenlabs_stream()
worker_keeps_talking()
print("\n" + "=" * 66)
print("MOUTH OK" if not FAILURES else f"MOUTH FAILURES: {FAILURES}")
sys.exit(1 if FAILURES else 0)
