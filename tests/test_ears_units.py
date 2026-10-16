"""ears.py offline units: which model, which mic, and the transcribe
wrapper that turns a clip into text.

The whisper model itself is never loaded here and no device is ever
opened. What is pinned is the DECISION layer around them -- the model
resolver that only picks a bigger model once it is on disk, the device
resolver that turns a name into an index, the rebuild-on-device-change
path, and the transcribe() wrapper (including its no-speech gate) driven
by a scripted fake model.
"""
from pathlib import Path
import sys
import tempfile
import types
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from backtalk import ears  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'ok' if cond else 'FAIL'} {name}"
          + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILURES.append(name)


def apple_gpu():
    print("\n--- _apple_gpu_available / _mlx_repo ---")
    with mock.patch.object(ears.sys, "platform", "win32"):
        check("a non-mac is never the GPU path",
              ears._apple_gpu_available() is False)
    with mock.patch.object(ears.sys, "platform", "darwin"), \
         mock.patch.object(ears.platform, "machine", return_value="x86_64"):
        check("an intel mac has no MLX", ears._apple_gpu_available() is False)
    with mock.patch.object(ears.sys, "platform", "darwin"), \
         mock.patch.object(ears.platform, "machine", return_value="arm64"), \
         mock.patch.object(ears.importlib, "import_module",
                           side_effect=ImportError("no mlx")):
        check("an arm mac without mlx-whisper is not the GPU path",
              ears._apple_gpu_available() is False)
    with mock.patch.object(ears.sys, "platform", "darwin"), \
         mock.patch.object(ears.platform, "machine", return_value="arm64"), \
         mock.patch.object(ears.importlib, "import_module",
                           return_value=types.ModuleType("mlx_whisper")):
        check("an arm mac with mlx-whisper is the GPU path",
              ears._apple_gpu_available() is True)
    check("the MLX repo name is derived from the model",
          ears._mlx_repo("small.en") == "mlx-community/whisper-small.en-mlx")


def model_resolution():
    print("\n--- _resolve_stt_model / _hf_cache_dir / _stt_cached / _stt ---")
    with mock.patch.object(ears, "CFG", {"stt_model": "small.en"}):
        check("no preference -> the plain model",
              ears._resolve_stt_model() == "small.en")
    with mock.patch.object(ears, "CFG",
                           {"stt_model": "small.en",
                            "stt_model_if_cached": "small.en"}):
        check("a preference equal to the model is ignored",
              ears._resolve_stt_model() == "small.en")
    with mock.patch.object(ears, "CFG", {"stt_model": "small.en",
                                         "stt_model_if_cached": "medium"}), \
         mock.patch.object(ears, "_stt_cached", return_value=True):
        check("a downloaded preference wins",
              ears._resolve_stt_model() == "medium")
    with mock.patch.object(ears, "CFG", {"stt_model": "small.en",
                                         "stt_model_if_cached": "medium"}), \
         mock.patch.object(ears, "_stt_cached", return_value=False):
        check("an absent preference is ignored",
              ears._resolve_stt_model() == "small.en")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        flat = root / "models--Systran--faster-whisper-small"
        flat.mkdir(parents=True)
        (flat / "model.bin").write_text("x")
        (flat / "config.json").write_text("{}")
        with mock.patch.object(ears.os.path, "expanduser", return_value=tmp):
            check("a flat downloaded model is found",
                  ears._hf_cache_dir("small") == flat)
            check("the .en alias finds the same folder",
                  ears._hf_cache_dir("small.en") == flat)
            check("a missing model returns its would-be path",
                  ears._hf_cache_dir("mlx-community/foo")
                  == Path(tmp) / "models--mlx-community--foo")

        snap = (Path(tmp) / "models--Systran--faster-whisper-medium"
                / "snapshots" / "abc123")
        snap.mkdir(parents=True)
        (snap / "model.bin").write_text("x")
        (snap / "config.json").write_text("{}")
        with mock.patch.object(ears.os.path, "expanduser", return_value=tmp):
            check("a snapshot layout is found too",
                  ears._hf_cache_dir("medium") == snap)

        bad = (Path(tmp) / "models--Systran--faster-whisper-base"
               / "snapshots" / "bad")
        bad.mkdir(parents=True)
        (bad / "model.bin").write_text("x")
        with mock.patch.object(ears.os.path, "expanduser", return_value=tmp):
            check("a snapshot missing its config is skipped",
                  ears._hf_cache_dir("base")
                  == Path(tmp) / "models--Systran--faster-whisper-base")

    with tempfile.TemporaryDirectory() as tmp:
        flat = (Path(tmp) / "models--Systran--faster-whisper-small")
        flat.mkdir(parents=True)
        for f in ("model.bin", "config.json", "tokenizer.json"):
            (flat / f).write_text("x")
        with mock.patch.object(ears.os.path, "expanduser", return_value=tmp):
            check("a fully downloaded model is cached",
                  ears._stt_cached("small") is True)
            (flat / "tokenizer.json").unlink()
            check("a model with no tokenizer is not cached",
                  ears._stt_cached("small") is False)

    with mock.patch.object(ears, "_stt_cache", None), \
         mock.patch.object(ears, "_resolve_stt_model",
                           return_value="small.en"):
        check("the model resolves once", ears._stt() == "small.en")
    with mock.patch.object(ears, "_stt_cache", "tiny"):
        check("a cached resolution is returned as-is", ears._stt() == "tiny")


def mic_index():
    print("\n--- _mic_index ---")
    devs = [{"name": "Speakers", "max_input_channels": 0},
            {"name": "Microphone Array", "max_input_channels": 2},
            {"name": "Headset Mic", "max_input_channels": 1}]
    with mock.patch.object(ears, "CFG", {}), \
         mock.patch.object(ears.sd, "query_devices", return_value=devs):
        check("no mic_device -> the system default",
              ears._mic_index() is None)
    with mock.patch.object(ears, "CFG", {"mic_device": "Headset Mic"}), \
         mock.patch.object(ears.sd, "query_devices", return_value=devs):
        check("an exact name resolves to its index", ears._mic_index() == 2)
    with mock.patch.object(ears, "CFG", {"mic_device": "headset"}), \
         mock.patch.object(ears.sd, "query_devices", return_value=devs):
        check("a substring resolves, case-insensitively",
              ears._mic_index() == 2)
    ears._mic_device_warned = False
    with mock.patch.object(ears, "CFG", {"mic_device": "Nope"}), \
         mock.patch.object(ears.sd, "query_devices", return_value=devs), \
         mock.patch.object(ears, "log") as logged:
        check("an absent name falls to the default",
              ears._mic_index() is None)
        check("and says so once", logged.called)
        check("a second press stays quiet", ears._mic_index() is None)
    ears._mic_device_warned = False
    with mock.patch.object(ears, "CFG", {"mic_device": "Mic"}), \
         mock.patch.object(ears.sd, "query_devices",
                           side_effect=RuntimeError("boom")), \
         mock.patch.object(ears, "log"):
        check("a failing device query falls to the default",
              ears._mic_index() is None)


def mic_open():
    print("\n--- _open_mic / _reopen_after_device_change ---")
    stream = object()
    with mock.patch.object(ears, "_mic_index", return_value=1), \
         mock.patch.object(ears.sd, "InputStream",
                           return_value=stream) as ins:
        check("the configured mic opens", ears._open_mic() is stream)
        check("with the configured device",
              ins.call_args.kwargs["device"] == 1)

    def stream_side(*a, **k):
        if k.get("device") is not None:
            raise RuntimeError("busy")
        return stream

    with mock.patch.object(ears, "_mic_index", return_value=1), \
         mock.patch.object(ears.sd, "InputStream", side_effect=stream_side), \
         mock.patch.object(ears, "log"):
        check("a device that will not open falls to the default",
              ears._open_mic() is stream)

    with mock.patch.object(ears, "_mic_index", return_value=1), \
         mock.patch.object(ears.sd, "InputStream",
                           side_effect=RuntimeError("dead")), \
         mock.patch.object(ears, "_reopen_after_device_change",
                           return_value=stream) as rebuild, \
         mock.patch.object(ears, "log"):
        check("a fully dead device rebuilds the audio system",
              ears._open_mic() is stream)
        check("the rebuild is the last resort", rebuild.called)

    with mock.patch.object(ears, "_mic_index", return_value=None), \
         mock.patch.object(ears.sd, "InputStream",
                           side_effect=RuntimeError("dead")), \
         mock.patch.object(ears, "_reopen_after_device_change",
                           return_value=stream) as rebuild2, \
         mock.patch.object(ears, "log"):
        check("the default device failing also rebuilds",
              ears._open_mic() is stream and rebuild2.called)

    with mock.patch.object(ears.sd, "_terminate"), \
         mock.patch.object(ears.sd, "_initialize") as init, \
         mock.patch.object(ears.sd, "InputStream", return_value=stream):
        check("the rebuild reopens once",
              ears._reopen_after_device_change({}) is stream)
        check("and re-initialises portaudio", init.called)
    with mock.patch.object(ears.sd, "_terminate",
                           side_effect=RuntimeError("already down")), \
         mock.patch.object(ears.sd, "_initialize"), \
         mock.patch.object(ears.sd, "InputStream", return_value=stream):
        check("a rebuild tolerates an already-down system",
              ears._reopen_after_device_change({}) is stream)


def mic_failure_words():
    print("\n--- explain_audio_failure / check_microphone ---")

    class PortAudioError(Exception):
        pass

    ears._mic_warned = False
    with mock.patch.object(ears, "log") as logged:
        check("a PortAudioError is handled by name",
              ears.explain_audio_failure(PortAudioError("bad")) is True)
        check("the full explanation is logged", logged.call_count >= 3)
    with mock.patch.object(ears, "log") as logged:
        check("a repeat is handled briefly",
              ears.explain_audio_failure(PortAudioError("bad")) is True)
        check("briefly means one line", logged.call_count == 1)
    ears._mic_warned = False
    with mock.patch.object(ears, "log"):
        check("a hint phrase is also handled",
              ears.explain_audio_failure(
                  RuntimeError("invalid device")) is True)
    check("an unrelated error is not ours",
          ears.explain_audio_failure(ValueError("nope")) is False)
    ears._mic_warned = False

    ears._mic_checked = False
    with mock.patch.object(ears, "_mic_index", return_value=None), \
         mock.patch.object(ears.sd, "check_input_settings"):
        check("a working mic passes the pre-flight",
              ears.check_microphone() is True)
        check("the pre-flight is remembered", ears.check_microphone() is True)
    ears._mic_checked = False
    with mock.patch.object(ears, "_mic_index", return_value=None), \
         mock.patch.object(ears.sd, "check_input_settings",
                           side_effect=RuntimeError("no mic")), \
         mock.patch.object(ears, "log"):
        check("a missing mic fails the pre-flight",
              ears.check_microphone() is False)


def probe():
    print("\n--- _probe ---")
    seen = {}

    def fake_transcribe(audio, language=None):
        seen["len"] = len(audio)
        seen["lang"] = language
        return [], None

    with mock.patch.object(ears, "CFG", {"stt_language": ""}):
        ears._probe(mock.Mock(transcribe=fake_transcribe))
    check("a probe runs a beat of silence through the model",
          seen.get("len") == ears.RATE // 10, seen)
    check("with English when none is forced", seen.get("lang") == "en", seen)
    with mock.patch.object(ears, "CFG", {"stt_language": "fr"}):
        ears._probe(mock.Mock(transcribe=fake_transcribe))
    check("with the forced language when set", seen.get("lang") == "fr", seen)


class Seg:
    def __init__(self, text, nsp):
        self.text = text
        self.no_speech_prob = nsp


class Info:
    language = "en"


def transcribe_path():
    print("\n--- transcribe / transcribe_language ---")
    pcm = np.zeros(ears.RATE, dtype=np.int16)

    class GoodModel:
        def transcribe(self, audio, **kw):
            return [Seg("Hello ", 0.1), Seg("world.", 0.2)], Info()

    with mock.patch.object(ears, "warm", return_value=GoodModel()), \
         mock.patch.object(ears, "_backend", "faster-whisper"), \
         mock.patch.object(ears, "_stt", return_value="small.en"), \
         mock.patch.object(ears, "CFG", {"stt_language": "",
                                         "stt_prompt": ""}):
        text, lang = ears.transcribe_language(pcm)
    check("segments are joined", text == "Hello world.", text)
    check("the detected language comes back", lang == "en", lang)
    check("the no-speech score is recorded",
          ears._LAST_NO_SPEECH == 0.2, ears._LAST_NO_SPEECH)

    class NoisyModel:
        def transcribe(self, audio, **kw):
            return [Seg("Thanks for watching!", 0.9)], Info()

    with mock.patch.object(ears, "warm", return_value=NoisyModel()), \
         mock.patch.object(ears, "_backend", "faster-whisper"), \
         mock.patch.object(ears, "_stt", return_value="small.en"), \
         mock.patch.object(ears, "CFG", {"stt_language": "",
                                         "stt_prompt": "",
                                         "stt_no_speech_prob": 0.5}):
        text, _ = ears.transcribe_language(pcm, gate_no_speech=True)
    check("the model's own no-speech score blanks a noisy clip",
          text == "", text)

    class MarkerModel:
        def transcribe(self, audio, **kw):
            return [Seg("[BLANK_AUDIO] real words", 0.1)], Info()

    with mock.patch.object(ears, "warm", return_value=MarkerModel()), \
         mock.patch.object(ears, "_backend", "faster-whisper"), \
         mock.patch.object(ears, "_stt", return_value="small.en"), \
         mock.patch.object(ears, "CFG", {"stt_language": "",
                                         "stt_prompt": ""}):
        text, _ = ears.transcribe_language(pcm)
    check("bracketed non-speech markers are stripped",
          text == "real words", text)

    fake_mlx = types.ModuleType("mlx_whisper")

    def mlx_transcribe(audio, **kw):
        return {"text": " Hi there. ", "language": "en",
                "segments": [{"no_speech_prob": 0.3}, {}]}

    fake_mlx.transcribe = mlx_transcribe
    with mock.patch.object(ears, "warm", return_value="repo"), \
         mock.patch.object(ears, "_backend", "mlx"), \
         mock.patch.object(ears, "_stt", return_value="small"), \
         mock.patch.object(ears, "CFG", {"stt_language": "",
                                         "stt_prompt": ""}), \
         mock.patch.dict(sys.modules, {"mlx_whisper": fake_mlx}):
        text, lang = ears.transcribe_language(pcm)
    check("the mlx backend returns text", text == "Hi there.", text)
    check("and its language", lang == "en", lang)
    check("and its no-speech score",
          ears._LAST_NO_SPEECH == 0.3, ears._LAST_NO_SPEECH)

    with mock.patch.object(ears, "warm", return_value=GoodModel()), \
         mock.patch.object(ears, "_backend", "faster-whisper"), \
         mock.patch.object(ears, "_stt", return_value="small.en"), \
         mock.patch.object(ears, "CFG", {"stt_language": "",
                                         "stt_prompt": ""}):
        check("transcribe is the text half of transcribe_language",
              ears.transcribe(pcm) == "Hello world.")


def gates():
    print("\n--- _rms / speech_* / _min_speech_frames ---")
    check("rms of silence is zero",
          ears._rms(np.zeros(10, dtype=np.int16)) == 0.0)
    check("rms of a loud frame is its amplitude",
          ears._rms(np.full(10, 1000, dtype=np.int16)) == 1000.0)
    check("no frame -> 0",
          ears._rms(np.array([], dtype=np.int16)) == 0.0)
    with mock.patch.object(ears, "CFG", {"stt_min_rms": 0}):
        check("a 0 loudness floor disables the gate",
              ears.speech_is_audible(0.0) is True)
    with mock.patch.object(ears, "CFG", {"stt_min_rms": 200}):
        check("loud enough passes", ears.speech_is_audible(500.0) is True)
        check("too quiet fails", ears.speech_is_audible(100.0) is False)
    with mock.patch.object(ears, "CFG", {"stt_no_speech_prob": 0.0}):
        check("a 0 no-speech cutoff disables the gate",
              ears.speech_is_confident(0.99) is True)
    with mock.patch.object(ears, "CFG", {"stt_no_speech_prob": 0.8}):
        check("confident speech passes", ears.speech_is_confident(0.2) is True)
        check("non-speech fails", ears.speech_is_confident(0.9) is False)
    with mock.patch.object(ears, "CFG", {"stt_min_speech_ms": 0}):
        check("the floor is the minimum, 8 frames",
              ears._min_speech_frames() == 8)
    with mock.patch.object(ears, "CFG", {"stt_min_speech_ms": 300}):
        check("milliseconds convert to frames",
              ears._min_speech_frames() == 10)
    with mock.patch.object(ears, "CFG", {"stt_min_speech_ms": 1}):
        check("a tiny value still floors at 8",
              ears._min_speech_frames() == 8)


def warm_path():
    print("\n--- warm ---")
    with mock.patch.object(ears, "check_microphone"), \
         mock.patch.object(ears, "_apple_gpu_available", return_value=False), \
         mock.patch.object(ears, "_stt", return_value="small.en"), \
         mock.patch.object(ears, "CFG",
                           {"stt_device": "cpu", "stt_compute": "int8"}), \
         mock.patch.object(ears, "_probe") as probe, \
         mock.patch.object(ears, "log"), \
         mock.patch.object(ears, "_model", None), \
         mock.patch.object(ears, "_backend", None):
        made = []

        def factory(_model, device, compute_type):
            made.append(device)
            return object()

        fake_fw = types.ModuleType("faster_whisper")
        fake_fw.WhisperModel = factory
        with mock.patch.dict(sys.modules, {"faster_whisper": fake_fw}):
            model = ears.warm()
            check("warm builds the faster-whisper model", model is not None)
            check("on the configured device", made == ["cpu"], made)
            check("and probes it before reporting ready", probe.called)
            check("and records the backend",
                  ears._backend == "faster-whisper", ears._backend)

    with mock.patch.object(ears, "check_microphone"), \
         mock.patch.object(ears, "_apple_gpu_available", return_value=False), \
         mock.patch.object(ears, "_stt", return_value="small"), \
         mock.patch.object(ears, "CFG",
                           {"stt_device": "cuda", "stt_compute": "float16"}), \
         mock.patch.object(ears, "log"), \
         mock.patch.object(ears, "_model", None), \
         mock.patch.object(ears, "_backend", None):
        made2 = []
        calls = {"n": 0}

        def factory2(_model, device, compute_type):
            made2.append(device)
            return object()

        def probe_side(_m):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("no cublas")

        fake_fw2 = types.ModuleType("faster_whisper")
        fake_fw2.WhisperModel = factory2
        with mock.patch.dict(sys.modules, {"faster_whisper": fake_fw2}), \
             mock.patch.object(ears, "_probe", side_effect=probe_side):
            ears.warm()
            check("a device that fails its probe falls to the CPU",
                  made2 == ["cuda", "cpu"], made2)

    with mock.patch.object(ears, "check_microphone"), \
         mock.patch.object(ears, "_apple_gpu_available", return_value=False), \
         mock.patch.object(ears, "_stt", return_value="small"), \
         mock.patch.object(ears, "CFG",
                           {"stt_device": "cpu", "stt_compute": "int8"}), \
         mock.patch.object(ears, "log"), \
         mock.patch.object(ears, "_model", None):
        fake_fw3 = types.ModuleType("faster_whisper")
        fake_fw3.WhisperModel = lambda *a, **k: object()
        raised = False
        with mock.patch.dict(sys.modules, {"faster_whisper": fake_fw3}), \
             mock.patch.object(ears, "_probe",
                               side_effect=RuntimeError("dead")):
            try:
                ears.warm()
            except RuntimeError:
                raised = True
        check("a CPU model that will not run raises", raised)

    with mock.patch.object(ears, "check_microphone"), \
         mock.patch.object(ears, "_apple_gpu_available", return_value=True), \
         mock.patch.object(ears, "_stt", return_value="small"), \
         mock.patch.object(ears, "_mlx_repo", return_value="repo"), \
         mock.patch.object(ears, "log"), \
         mock.patch.object(ears, "_model", None), \
         mock.patch.object(ears, "_backend", None):
        seen = {}
        fake_mlx = types.ModuleType("mlx_whisper")

        def mlx_tx(audio, path_or_hf_repo=None, language=None, verbose=None):
            seen["repo"] = path_or_hf_repo
            return {}

        fake_mlx.transcribe = mlx_tx
        with mock.patch.dict(sys.modules, {"mlx_whisper": fake_mlx}):
            model = ears.warm()
            check("warm loads the MLX repo", model == "repo", model)
            check("warming on the derived repo",
                  seen.get("repo") == "repo", seen)
            check("and records the mlx backend", ears._backend == "mlx")

    with mock.patch.object(ears, "check_microphone"), \
         mock.patch.object(ears, "_model", "already"):
        check("an already-warm model is returned as-is",
              ears.warm() == "already")


def wake_loop():
    print("\n--- Ears.wait_for_wake ---")
    e = ears.Ears()
    seq = ["", "hey seyon"]
    calls = {"n": 0}

    def listen_once(self, gate=None, abort=None, timeout_s=None,
                    on_partial=None, owner_check=None):
        i = calls["n"]
        calls["n"] += 1
        return seq[i] if i < len(seq) else None

    with mock.patch.object(ears.Ears, "listen_once", new=listen_once):
        got = e.wait_for_wake(["hey seyon"])
    check("a blanked clip is skipped, the wake phrase returned",
          got == "hey seyon", got)

    def listen_none(self, gate=None, abort=None, timeout_s=None,
                    on_partial=None, owner_check=None):
        return None

    with mock.patch.object(ears.Ears, "listen_once", new=listen_none):
        got = e.wait_for_wake(["hey seyon"])
    check("an aborted listen ends the wait", got is None)


print("=" * 66)
print("EARS UNITS (offline, no mic, no model)")
print("=" * 66)
apple_gpu()
model_resolution()
mic_index()
mic_open()
mic_failure_words()
probe()
transcribe_path()
gates()
warm_path()
wake_loop()

print("\n" + "=" * 66)
print("EARS UNITS OK" if not FAILURES else f"EARS FAILURES: {FAILURES}")
sys.exit(1 if FAILURES else 0)
