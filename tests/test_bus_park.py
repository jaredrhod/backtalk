"""The bus must never be left saying `speaking` by a voice that is gone.

Three ways it used to happen, and where each one is closed:

  1. The voice dies mid-speech and never reaches amain()'s finally, so
     nobody writes `idle`. `signals.register_exit_park` covers the
     deaths a Python process can still clean up after.

  2. The supervisor restarts a wedged voice with `taskkill /F`, which
     runs NO handler in the dying process at all. Closed from the other
     side: `main._take_the_bus` parks the bus the moment the successor
     claims the instance lock, because holding that lock is proof that
     nothing else can be speaking.

  3. Whatever the old process left behind (state, thinking-sound pid
     file) is inherited by the new one, which starts by claiming the
     bus rather than believing it.

Run: uv run python tests/test_bus_park.py
"""
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtalk import main as btmain                   # noqa: E402
from backtalk import signals                          # noqa: E402
from backtalk.config import REPO                      # noqa: E402


class BusCase(unittest.TestCase):
    """Points the bus at a scratch directory so nothing real is touched."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = str(Path(self.tmp.name) / ".voice_state")
        self.loading = str(Path(self.tmp.name) / ".voice_loading_pid")
        self.caption = str(Path(self.tmp.name) / ".voice_caption")
        for target, value in (("_STATE_FILE", self.state),
                              ("_LOADING_PID_FILE", self.loading),
                              ("_CAPTION_FILE", self.caption),
                              ("_BH_STATE", "")):
            p = mock.patch.object(signals, target, value)
            p.start()
            self.addCleanup(p.stop)

    def write_state(self, text):
        Path(self.state).write_text(text, encoding="utf-8")

    def read_state(self):
        return Path(self.state).read_text(encoding="utf-8")


class TestPark(BusCase):

    def test_a_speaking_bus_is_parked(self):
        self.write_state("speaking")
        signals.park()
        self.assertEqual(self.read_state(), "idle")

    def test_a_stale_thinking_sound_pid_is_dropped(self):
        Path(self.loading).write_text("4242", encoding="utf-8")
        signals.park()
        self.assertFalse(Path(self.loading).exists(),
                         "the dead process's pid file would be inherited")

    def test_the_barehands_mirror_is_parked_too(self):
        mirror = str(Path(self.tmp.name) / "bh_state")
        with mock.patch.object(signals, "_BH_STATE", mirror):
            self.write_state("speaking")
            signals.park()
        self.assertEqual(Path(mirror).read_text(encoding="utf-8"), "idle")

    def test_park_never_raises(self):
        """The bus rule: a write failure must not take the voice with it.
        A voice dying because it could not report that it was dying would
        be the worst possible way to leave this file behind."""
        signals._STATE_FILE = str(Path(self.tmp.name) / "no" / "dir" / "x")
        signals._LOADING_PID_FILE = str(Path(self.tmp.name) / "no" / "y")
        signals.park()                              # must not raise

    def test_the_park_is_registered_at_exit(self):
        with mock.patch.object(signals.atexit, "register") as reg:
            signals.register_exit_park()
        reg.assert_called_once_with(signals.park)


class TestExitParkActuallyFires(unittest.TestCase):
    """The real thing: a child process that dies mid-speech.

    atexit is only worth registering if the interpreter actually runs it
    on the death that matters — an uncaught exception escaping the
    process, which is how a crashed voice line leaves."""

    def test_an_uncaught_crash_leaves_the_bus_idle(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = str(Path(tmp) / ".voice_state")
            loading = str(Path(tmp) / ".voice_loading_pid")
            code = textwrap.dedent(f"""
                from backtalk import signals as s
                s._STATE_FILE = {state!r}
                s._LOADING_PID_FILE = {loading!r}
                s._BH_STATE = ""
                s.register_exit_park()
                s.set_state("speaking")
                raise RuntimeError("voice died mid-speech")
            """)
            subprocess.run([sys.executable, "-c", code], cwd=str(REPO),
                           capture_output=True, timeout=180)
            self.assertEqual(Path(state).read_text(encoding="utf-8"), "idle",
                             "the crash left the bus saying 'speaking'")
            self.assertFalse(Path(loading).exists())


class TestTakeTheBus(unittest.TestCase):

    def test_the_successor_parks_and_registers(self):
        with mock.patch.object(btmain, "signals") as sig:
            btmain._take_the_bus()
        sig.register_exit_park.assert_called_once_with()
        sig.park.assert_called_once_with()

    def test_the_instance_lock_is_taken_before_the_bus(self):
        """The whole reason a successor may write `idle`: it holds the
        lock, so no other voice can be speaking. Reorder this and the
        park becomes a lie that a live voice could contradict."""
        order = []

        def claim():
            order.append("claim")
            return True

        # amain must be patched too: its argument is evaluated before
        # asyncio.run is ever reached, and the real one opens a mic.
        with mock.patch.object(btmain, "_claim_single_instance", claim), \
             mock.patch.object(btmain, "_take_the_bus",
                               lambda: order.append("take")), \
             mock.patch.object(btmain, "amain", mock.Mock()), \
             mock.patch.object(btmain, "asyncio", mock.Mock()):
            btmain.main()
        self.assertEqual(order, ["claim", "take"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
