"""ptt.py offline units: the hold-to-talk key logic with a fake keyboard.

pynput needs a real desktop session, which a headless box (and CI) does
not have. The listener's whole job is a state machine on top of key
events, so a fake keyboard that records _on_press/_on_release and lets us
call them directly pins the two traps the module exists to survive: the
key-repeat flood on press, and the auto-repeat down/up PAIRS on release.
"""
from pathlib import Path
import importlib
import sys
import types
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtalk import ptt  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'ok' if cond else 'FAIL'} {name}"
          + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILURES.append(name)


class FakeKey:
    home = "key:home"
    end = "key:end"
    alt_r = "key:alt_r"
    alt_l = "key:alt_l"
    ctrl_r = "key:ctrl_r"
    ctrl_l = "key:ctrl_l"
    cmd_r = "key:cmd_r"
    cmd_l = "key:cmd_l"
    shift_r = "key:shift_r"
    shift_l = "key:shift_l"


class FakeKeyCode:
    @staticmethod
    def from_char(c):
        return f"char:{c}"


class FakeListener:
    def __init__(self, on_press=None, on_release=None):
        self.on_press = on_press
        self.on_release = on_release
        self.daemon = False
        self.started = False

    def start(self):
        self.started = True


FAKE = types.SimpleNamespace(Key=FakeKey, KeyCode=FakeKeyCode,
                             Listener=FakeListener)


def resolution():
    print("\n--- resolve_key ---")
    with mock.patch.object(ptt, "keyboard", FAKE):
        check("a single character becomes a KeyCode",
              ptt.resolve_key("a") == "char:a")
        check("a friendly name maps to pynput's",
              ptt.resolve_key("right_alt") == "key:alt_r")
        check("an option name maps too",
              ptt.resolve_key("left_option") == "key:alt_l")
        check("a plain key name resolves",
              ptt.resolve_key("home") == "key:home")
        check("case and blanks are ignored",
              ptt.resolve_key("  Home ") == "key:home")
        check("empty falls back to home", ptt.resolve_key("") == "key:home")
        with mock.patch("builtins.print"):
            check("an unknown name falls back to home",
                  ptt.resolve_key("bogus") == "key:home")
    with mock.patch.object(ptt, "keyboard", None):
        check("with no backend the name passes through",
              ptt.resolve_key("home") == "home")


def listener_lifecycle():
    print("\n--- PTTListener press/release/settle ---")
    with mock.patch.object(ptt, "keyboard", type(
            "FakeKeyboard", (), {"Key": FakeKey, "KeyCode": FakeKeyCode,
                                 "Listener": FakeListener})):
        lis = ptt.PTTListener("home")
        check("the listener key is the resolved key", lis._key == "key:home")
        check("the listener is started", lis._listener.started)
        check("nothing is held at rest", lis.is_held() is False)

        lis._on_press("key:end")
        check("a press of another key is ignored", lis.is_held() is False)

        lis._on_press("key:home")
        check("a press is held", lis._held is True)
        check("the press event is set", lis._press_evt.is_set())
        lis._on_press("key:home")
        check("a key-repeat keeps it held", lis._held is True)

        lis._on_release("key:end")
        check("a release of another key is ignored", lis._release_t is None)
        lis._on_release("key:home")
        check("a release is provisional", lis._release_t is not None)

        lis._settle()
        check("settle keeps it held inside the grace window",
              lis._held is True and lis._release_t is not None)
        lis._release_t -= (ptt.PTTListener.RELEASE_GRACE + 1)
        lis._settle()
        check("settle commits a standing release", lis._held is False)
        check("and forgets the release time", lis._release_t is None)


def wait_press_and_is_held():
    print("\n--- wait_press / is_held ---")
    fake = type("FakeKeyboard", (), {"Key": FakeKey, "KeyCode": FakeKeyCode,
                                     "Listener": FakeListener})
    with mock.patch.object(ptt, "keyboard", fake):
        lis = ptt.PTTListener("home")
        lis._on_press("key:home")
        lis.wait_press()      # the event is already set: returns at once
        check("wait_press returns once the event is set", True)
        lis._on_release("key:home")
        lis._release_t -= (ptt.PTTListener.RELEASE_GRACE + 1)
        check("is_held settles a stale release", lis.is_held() is False)

        class FlakyEvent:
            def __init__(self):
                self.n = 0

            def wait(self, timeout=None):
                self.n += 1
                return self.n >= 2

            def clear(self):
                pass

        lis2 = ptt.PTTListener("home")
        lis2._press_evt = FlakyEvent()
        lis2.wait_press()
        check("wait_press rides out a timeout and returns on the press",
              lis2._press_evt.n == 2, lis2._press_evt.n)


def working_backend_branch():
    print("\n--- the module takes a working keyboard backend ---")
    saved = sys.modules.get("pynput")
    saved_sub = sys.modules.get("pynput.keyboard")
    fake_pkg = types.ModuleType("pynput")
    fake_kb = types.ModuleType("pynput.keyboard")
    fake_pkg.keyboard = fake_kb
    sys.modules["pynput"] = fake_pkg
    sys.modules["pynput.keyboard"] = fake_kb
    try:
        reloaded = importlib.reload(ptt)
        check("with a backend, keyboard is set", reloaded.keyboard is fake_kb)
        check("and no error is recorded", reloaded._PTT_ERROR is None)
    finally:
        if saved is not None:
            sys.modules["pynput"] = saved
        else:
            sys.modules.pop("pynput", None)
        if saved_sub is not None:
            sys.modules["pynput.keyboard"] = saved_sub
        else:
            sys.modules.pop("pynput.keyboard", None)
        importlib.reload(ptt)


def no_backend_branch():
    print("\n--- the module survives a missing keyboard backend ---")
    saved = sys.modules.get("pynput")
    saved_sub = sys.modules.pop("pynput.keyboard", None)
    sys.modules["pynput"] = types.ModuleType("pynput")  # no .keyboard
    try:
        reloaded = importlib.reload(ptt)
        check("without a backend, keyboard is None", reloaded.keyboard is None)
        check("and the reason is kept", reloaded._PTT_ERROR is not None)
        check("resolve_key passes names through",
              reloaded.resolve_key("home") == "home")
        raised = None
        try:
            reloaded.PTTListener("home")
        except RuntimeError as exc:
            raised = exc
        check("PTTListener refuses without a backend", raised is not None)
    finally:
        if saved is not None:
            sys.modules["pynput"] = saved
        else:
            sys.modules.pop("pynput", None)
        if saved_sub is not None:
            sys.modules["pynput.keyboard"] = saved_sub
        importlib.reload(ptt)


print("=" * 66)
print("PTT UNITS (offline, fake keyboard)")
print("=" * 66)
resolution()
listener_lifecycle()
wait_press_and_is_held()
working_backend_branch()
no_backend_branch()

print("\n" + "=" * 66)
print("PTT UNITS OK" if not FAILURES else f"PTT FAILURES: {FAILURES}")
sys.exit(1 if FAILURES else 0)
