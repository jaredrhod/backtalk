"""The small pure modules: Spotify ducking and the permission vocabulary.

Neither needs a Mac, Spotify, or a running voice line. Ducking's
AppleScript bridge is stubbed so the duck/restore/debounce logic is
driven directly, and the "not on macOS" no-op paths are checked as-is.
The permission dataclasses are the shared vocabulary main.py's gate and
brain.py's translation both speak, so their exact shape is pinned here.
"""
import sys
import threading
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtalk import ducking, permresult                       # noqa: E402

real_osa = ducking._osa
real_set_volume = ducking._set_volume
real_spotify_volume = ducking._spotify_volume

failures = []


def check(name, cond, detail=""):
    print("    %s %s%s" % ("ok  " if cond else "FAIL", name,
                           "" if cond else "   <- " + str(detail)))
    if not cond:
        failures.append(name)


print("=" * 66)
print("SPOTIFY DUCKING & PERMISSION RESULTS")
print("=" * 66)

# ---- non-macOS: every bridge call is a silent no-op ---------------------
ducking._DARWIN = False
check("_osa is a no-op off macOS", ducking._osa("anything") is None)
check("_spotify_volume is None off macOS", ducking._spotify_volume() is None)

calls = []
ducking._set_volume = lambda level: calls.append(level)

d = ducking.Ducker()
d.speech_start()
check("off macOS, speech_start does nothing", d._original is None and calls == [])

# ---- the macOS logic, with the bridge stubbed ---------------------------
ducking._DARWIN = True
ducking._osa = lambda script, timeout=2.0: "true" if "is running" in script else "55"
check("a running Spotify reports its volume", ducking._spotify_volume() == 55)

ducking._osa = lambda script, timeout=2.0: "false"
check("no Spotify means no volume", ducking._spotify_volume() is None)

ducking._osa = lambda script, timeout=2.0: "not a number"
check("a garbled volume reads as None", ducking._spotify_volume() is None)

# quiet music is left alone
ducking._spotify_volume = lambda: 20
calls.clear()
d = ducking.Ducker()
d.speech_start()
check("quiet music is not ducked", d._original is None and calls == [])

# loud music ducks to max(threshold, current * pct)
ducking._spotify_volume = lambda: 80
calls.clear()
d = ducking.Ducker()
d.speech_start()
check("loud music is ducked once", d._original == 80 and calls == [48], calls)
d.speech_start()
check("a second start does not duck twice", calls == [48], calls)

# end schedules a debounced restore; resumed speech cancels it
d.speech_end(debounce=0.05)
d.speech_end(debounce=0.05)          # replacing the timer must not double-restore
time.sleep(0.25)
check("the restore fires once and is cleared",
      calls == [48, 80] and d._original is None, calls)

# a pending restore is cancelled by speech resuming
ducking._spotify_volume = lambda: 90
calls.clear()
d = ducking.Ducker()
d.speech_start()
d.speech_end(debounce=0.2)
d.speech_start()                     # resume before the debounce elapses
time.sleep(0.35)
check("resumed speech cancels the pending restore",
      calls == [54], calls)
check("still ducked after resume", d._original == 90)

# restore_now restores synchronously for shutdown
d.restore_now()
check("restore_now restores on the spot",
      calls == [54, 90] and d._original is None, calls)

# speech_end with nothing ducked is a no-op
d2 = ducking.Ducker()
calls.clear()
d2.speech_end()
check("speech_end with nothing ducked is a no-op", calls == [])

# ---- the permission vocabulary ------------------------------------------
a = permresult.PermissionResultAllow()
check("allow defaults to behavior 'allow'", a.behavior == "allow")
check("allow carries updated_input None", a.updated_input is None)

a2 = permresult.PermissionResultAllow(updated_input={"k": 1})
check("allow can carry an updated input", a2.updated_input == {"k": 1})

deny = permresult.PermissionResultDeny()
check("deny defaults to behavior 'deny'", deny.behavior == "deny")
check("deny defaults to no message", deny.message == "")
check("deny defaults to no interrupt", deny.interrupt is False)

# ---- the pieces the AppleScript bridge hides ----------------------------
ducking._DARWIN = True


class _Run:
    stdout = "  42  \n"


with mock.patch.object(ducking.subprocess, "run", lambda *a, **k: _Run()):
    check("the real bridge strips osascript's stdout",
          real_osa("script") == "42")
with mock.patch.object(ducking.subprocess, "run",
                       side_effect=OSError("no osascript")):
    check("a failed osascript reads as None", real_osa("script") is None)

scripts = []
ducking._osa = lambda script, timeout=2.0: scripts.append(script) or "true"
real_set_volume(42)
check("_set_volume asks Spotify for the new level",
      any("42" in s for s in scripts), scripts)

ducking._osa = lambda script, timeout=2.0: "true" if "is running" in script else None
check("Spotify with no volume read reads as None",
      real_spotify_volume() is None)

ducking._osa = lambda script, timeout=2.0: "true" if "is running" in script else "not a number"
check("a garbled volume reads as None",
      real_spotify_volume() is None)

calls.clear()
idle = ducking.Ducker()
ducking._spotify_volume = lambda: 80
ducking._set_volume = lambda level: calls.append(level)
idle._restore()
check("_restore with nothing ducked is a no-op",
      calls == [] and idle._timer is None)

ducking._spotify_volume = lambda: 80
calls.clear()
d4 = ducking.Ducker()
d4.speech_start()
d4.speech_end(debounce=5.0)          # long timer, not yet fired
d4.restore_now()
check("restore_now cancels the timer and restores at once",
      d4._original is None and d4._timer is None and calls == [48, 80], calls)

# A duck already in flight blocks a second start from ducking again.
# The mouth worker and a PTT press both call speech_start from different
# threads, so this concurrency is real: without the in-flight claim the
# second caller reads the already-ducked volume and ducks twice, and the
# restore then returns to the ducked level, leaving music too quiet.
calls.clear()
d5 = ducking.Ducker()
entered = threading.Event()
release = threading.Event()


def _slow_probe():
    entered.set()
    release.wait(1.0)
    return 80


ducking._spotify_volume = _slow_probe
t = threading.Thread(target=d5.speech_start)
t.start()
entered.wait(1.0)                    # the first duck has claimed and is probing
d5.speech_start()                    # the second must not duck again
release.set()
t.join(1.0)
check("a duck still in flight is not ducked twice",
      calls == [48] and d5._original == 80, calls)

print("\n" + "=" * 66)
print("DUCKING OK" if not failures else "DUCKING FAILURES: %s" % failures)
sys.exit(1 if failures else 0)
