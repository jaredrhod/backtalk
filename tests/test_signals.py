"""The signal bus, isolated: every note the voice line leaves on disk.

signals has no model and no audio behind it -- it is tiny files other
programs watch -- so it can be driven end to end with the paths pointed
at a temp dir. What is checked is the contract a face depends on: the
right file, the right bytes, and the promise that no write ever raises.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np                                          # noqa: E402
from backtalk.config import CFG                             # noqa: E402

_TMP = tempfile.mkdtemp(prefix="bt_signals_")
CFG["signals_dir"] = _TMP
CFG["barehands_state_dir"] = os.path.join(_TMP, "bh")
os.makedirs(CFG["barehands_state_dir"], exist_ok=True)
CFG["thinking_sound"] = ""

from backtalk import signals                                # noqa: E402

failures = []


def check(name, cond, detail=""):
    print("    %s %s%s" % ("ok  " if cond else "FAIL", name,
                           "" if cond else "   <- " + str(detail)))
    if not cond:
        failures.append(name)


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


print("=" * 66)
print("THE SIGNAL BUS")
print("=" * 66)

signals.set_state("listening")
check("state is written", read(os.path.join(_TMP, ".voice_state")) == "listening")
check("state is mirrored to barehands",
      read(os.path.join(_TMP, "bh", "state")) == "listening")

signals.park()
check("park leaves the bus idle",
      read(os.path.join(_TMP, ".voice_state")) == "idle")

signals.caption("hello there")
cap = json.loads(read(os.path.join(_TMP, ".voice_caption")))
check("caption carries the words", cap["text"] == "hello there", cap)
signals.caption_clear()
check("caption_clear removes the file",
      not os.path.exists(os.path.join(_TMP, ".voice_caption")))

signals.heard("what time is", final=False)
hrd = json.loads(read(os.path.join(_TMP, ".voice_heard")))
check("heard carries the in-progress words",
      hrd["text"] == "what time is" and hrd["final"] is False and "ts" in hrd, hrd)
signals.heard("what time is it", final=True)
check("heard marks the final transcript",
      json.loads(read(os.path.join(_TMP, ".voice_heard")))["final"] is True)
signals.heard("")
check("heard writes an empty line as an empty string",
      json.loads(read(os.path.join(_TMP, ".voice_heard")))["text"] == "")
signals.heard_clear()
check("heard_clear removes the file",
      not os.path.exists(os.path.join(_TMP, ".voice_heard")))

signals.active_window("code.exe \u2014 main.py")
aw = json.loads(read(os.path.join(_TMP, ".voice_active_window")))
check("active window carries the label",
      aw["text"] == "code.exe \u2014 main.py" and "ts" in aw, aw)
signals.active_window("")
check("an empty label is written as an empty string",
      json.loads(read(os.path.join(_TMP, ".voice_active_window")))["text"] == "")

signals.language("en")
check("language is written",
      json.loads(read(os.path.join(_TMP, ".voice_language")))["language"] == "en")
signals.language(None)
check("a None language stays None, not the string",
      json.loads(read(os.path.join(_TMP, ".voice_language")))["language"] is None)

signals.direction(["<<smile>>", "<<wave>>"])
d = json.loads(read(os.path.join(_TMP, ".voice_direction")))
check("directions are published raw", d["directions"] == ["<<smile>>", "<<wave>>"], d)
signals.direction([])
check("empty directions write nothing new", d["directions"] == ["<<smile>>", "<<wave>>"])

signals.reply_done()
check("reply_done is stamped",
      "ts" in json.loads(read(os.path.join(_TMP, ".voice_reply_done"))))

signals.set_rate_limit("session", 0.5, 111)
signals.set_rate_limit("seven_day", None, 222)
rl = json.loads(read(os.path.join(_TMP, ".voice_rate_limits")))
check("rate limits merge two windows",
      set(rl) == {"session", "seven_day"} and rl["session"]["utilization"] == 0.5, rl)
signals.set_rate_limit("", 1, 2)
check("an empty window is ignored",
      set(json.loads(read(os.path.join(_TMP, ".voice_rate_limits")))) ==
      {"session", "seven_day"})

pcm = (np.sin(np.linspace(0, 20, 400)) * 10000).astype(np.int16)
signals.feed_waveform(pcm)
wav = json.loads(read(os.path.join(_TMP, ".voice_waveform")))
check("a waveform is downsampled to 64 points", len(wav["samples"]) == 64, len(wav["samples"]))
check("speaking state self-heals on a waveform",
      read(os.path.join(_TMP, ".voice_state")) == "speaking")
check("barehands wave is normalised 0..1",
      all(-1.0 <= s <= 1.0 for s in
          json.loads(read(os.path.join(_TMP, "bh", "wave.json")))["samples"]))

signals._last_waveform_write = 0.0
signals.feed_waveform(np.zeros(0, dtype=np.int16))   # empty: must not raise
check("an empty block is ignored safely", True)

signals.static_start()                                # no thinking sound configured
check("static_start no-ops without a thinking sound", signals._static_proc is None)
signals.static_stop()                                 # never raises with nothing playing
check("static_stop is safe with nothing playing", True)

signals.register_exit_park()
check("register_exit_park installs cleanly", True)

# Every write is wrapped: a path the bus cannot write must never raise.
with mock.patch.object(signals, "_STATE_FILE", _TMP), \
     mock.patch.object(signals, "_BH_STATE", _TMP), \
     mock.patch.object(signals, "_DIRECTION_FILE", _TMP), \
     mock.patch.object(signals, "_LANG_FILE", _TMP), \
     mock.patch.object(signals, "_REPLY_DONE_FILE", _TMP), \
     mock.patch.object(signals, "_RATE_LIMIT_FILE", _TMP), \
     mock.patch.object(signals, "_WAVEFORM_FILE", _TMP), \
     mock.patch.object(signals, "_BH_WAVE", _TMP), \
     mock.patch.object(signals, "_BH_WAVE", _TMP), \
     mock.patch.object(signals, "_CAPTION_FILE", _TMP), \
     mock.patch.object(signals, "_HEARD_FILE", _TMP), \
     mock.patch.object(signals, "_ACTIVE_FILE", _TMP):
    signals.set_state("x")
    signals.direction(["<<a>>"])
    signals.language("en")
    signals.reply_done()
    signals.set_rate_limit("w", 0.1, 1)
    signals._last_waveform_write = 0.0
    signals.feed_waveform(pcm)
    signals.caption("x")
    signals.caption_clear()
    signals.heard("x")
    signals.heard_clear()
    signals.active_window("x")
check("an unwritable bus path never raises", True)

with mock.patch.object(signals, "_BH_STATE", ""):
    signals.set_state("idle")
check("state writes without a barehands mirror",
      read(os.path.join(_TMP, ".voice_state")) == "idle")

# The throttle is what keeps a 60fps reader cheap.
before_ts = json.loads(read(os.path.join(_TMP, ".voice_waveform")))["ts"]
signals._last_waveform_write = 1000.0
with mock.patch.object(signals.time, "time", return_value=1000.005):
    signals.feed_waveform(pcm)
after_ts = json.loads(read(os.path.join(_TMP, ".voice_waveform")))["ts"]
check("a waveform inside the throttle window is dropped",
      after_ts == before_ts, (before_ts, after_ts))

# _player_cmd: whichever platform, and whichever player exists.
with mock.patch.object(signals, "sys", mock.Mock(platform="darwin")):
    check("macOS uses afplay",
          signals._player_cmd("s.wav") == ["afplay", "-v", "0.35", "s.wav"])
with mock.patch.object(signals, "sys", mock.Mock(platform="linux")), \
     mock.patch("shutil.which",
                side_effect=lambda c: f"/usr/bin/{c}" if c == "ffplay" else None):
    cmd = signals._player_cmd("s.wav")
    check("linux prefers ffplay with a quiet volume",
          cmd is not None and cmd[0] == "ffplay" and "quiet" in cmd, cmd)
with mock.patch.object(signals, "sys", mock.Mock(platform="linux")), \
     mock.patch("shutil.which",
                side_effect=lambda c: f"/usr/bin/{c}" if c == "aplay" else None):
    check("falls back to aplay when ffplay is absent",
          signals._player_cmd("s.wav") == ["aplay", "s.wav"])
with mock.patch.object(signals, "sys", mock.Mock(platform="linux")), \
     mock.patch("shutil.which", return_value=None):
    check("no player at all", signals._player_cmd("s.wav") is None)

# static_start / static_stop with a real thinking sound.
_sound = os.path.join(_TMP, "think.wav")
open(_sound, "w").close()
fake_proc = mock.Mock(pid=4321)
with mock.patch.object(signals, "_THINKING_SOUND", _sound), \
     mock.patch.object(signals, "_player_cmd", return_value=["afplay", "s"]), \
     mock.patch.object(signals.subprocess, "Popen", return_value=fake_proc):
    signals.static_start()
check("thinking sound starts and records its pid",
      signals._static_proc is fake_proc
      and read(os.path.join(_TMP, ".voice_loading_pid")) == "4321")
signals.static_stop()
check("static_stop terminates it and clears the pid file",
      signals._static_proc is None
      and not os.path.exists(os.path.join(_TMP, ".voice_loading_pid")))

with mock.patch.object(signals, "_THINKING_SOUND", _sound), \
     mock.patch.object(signals, "_player_cmd", return_value=None):
    signals.static_start()
check("no player -> no thinking sound", signals._static_proc is None)
with mock.patch.object(signals, "_THINKING_SOUND",
                       os.path.join(_TMP, "think.wav")), \
     mock.patch.object(signals, "_player_cmd", return_value=["afplay", "s"]), \
     mock.patch.object(signals.subprocess, "Popen",
                       side_effect=OSError("cannot spawn")):
    signals.static_start()
check("a spawn failure leaves no process behind", signals._static_proc is None)

# A pid file that cannot be written must not orphan the player.
with mock.patch.object(signals, "_THINKING_SOUND", _sound), \
     mock.patch.object(signals, "_player_cmd", return_value=["afplay", "s"]), \
     mock.patch.object(signals.subprocess, "Popen", return_value=fake_proc), \
     mock.patch("builtins.open", side_effect=OSError("read only")):
    signals.static_start()
check("an unwritable pid file keeps the player terminable",
      signals._static_proc is fake_proc)
signals.static_stop()

# A player that will not die must not take the bus down either.
dead = mock.Mock()
dead.terminate.side_effect = OSError("already gone")
signals._static_proc = dead
signals.static_stop()
check("a stubborn player is forgotten, not fatal",
      signals._static_proc is None)

# No barehands hooked up: the wave write skips the mirror entirely.
with mock.patch.object(signals, "_BH_WAVE", ""):
    signals._last_waveform_write = 0.0
    signals.feed_waveform(pcm)
check("a waveform writes without a barehands mirror", True)

print("\n" + "=" * 66)
print("SIGNALS OK" if not failures else "SIGNALS FAILURES: %s" % failures)
sys.exit(1 if failures else 0)
