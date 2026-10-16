"""Test screen awareness offline: the one-shot capture the spoken "look at
my screen" rides on, the per-platform focused-window reading the activity
journal is built from, and the watch loop that only reports changes.

Nothing here touches a real display. mss is replaced by a fake grabber, the
window readers are driven through fakes, and the module is reloaded under a
broken `mss` so the import-failure path is exercised too -- a machine with
no screen must still import and run the voice line.
"""
import asyncio
import importlib
from pathlib import Path
import sys
import types
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtalk import screen  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print("    %s %s%s" % ("ok  " if cond else "FAIL", name,
                           ("  " + str(detail)) if detail else ""))
    if not cond:
        FAILURES.append(name)
    return cond


def describe_lines():
    print("\n--- describe: one line for the journal ---")
    check("app and title",
          screen.describe("code.exe", "main.py") == "code.exe \u2014 main.py",
          screen.describe("code.exe", "main.py"))
    check("app only", screen.describe("code.exe", "") == "code.exe")
    check("title only", screen.describe("", "main.py") == "main.py")
    check("neither -> empty", screen.describe("", "") == "")
    check("whitespace is trimmed",
          screen.describe("  code.exe  ", "  main.py  ")
          == "code.exe \u2014 main.py")


def _make_ctypes(hwnd=1, handle=2, title="Notes",
                 exe="C:/Apps/notepad.exe"):
    class _Buf:
        def __init__(self, n=0):
            self.value = ""

    class _Dword:
        def __init__(self, v=0):
            self.value = v

    class _User32:
        def GetForegroundWindow(self):
            return hwnd

        def GetWindowTextLengthW(self, h):
            return len(title) + 1

        def GetWindowTextW(self, h, buf, n):
            buf.value = title

        def GetWindowThreadProcessId(self, h, ref):
            ref.value = 42

    class _Kernel32:
        def OpenProcess(self, access, inherit, pid):
            return handle

        def QueryFullProcessImageNameW(self, h, flags, buf, size):
            buf.value = exe

        def CloseHandle(self, h):
            pass

    class _FakeCtypes:
        create_unicode_buffer = staticmethod(_Buf)
        byref = staticmethod(lambda x: x)
        c_ulong = _Dword

        class windll:
            user32 = _User32()
            kernel32 = _Kernel32()

    return _FakeCtypes


def windows_reading():
    print("\n--- active_window: Windows reads user32 ---")
    fake = _make_ctypes(hwnd=1, handle=2, title="Notes",
                        exe="C:/Apps/notepad.exe")
    with mock.patch.object(screen.sys, "platform", "win32"), \
         mock.patch.dict(sys.modules, {"ctypes": fake}):
        got = screen.active_window()
    check("app is the process name, title is the window",
          got == ("notepad.exe", "Notes"), got)

    with mock.patch.object(screen.sys, "platform", "win32"), \
         mock.patch.dict(sys.modules, {"ctypes": _make_ctypes(hwnd=0)}):
        got = screen.active_window()
    check("no foreground window -> unknown", got == ("", ""), got)

    with mock.patch.object(screen.sys, "platform", "win32"), \
         mock.patch.dict(sys.modules,
                         {"ctypes": _make_ctypes(hwnd=1, handle=0,
                                                 title="Notes")}):
        got = screen.active_window()
    check("a refused process handle still names the window",
          got == ("", "Notes"), got)


def macos_reading():
    print("\n--- active_window: macOS asks System Events ---")
    calls = []

    def fake_run(args):
        calls.append(args)
        return "Safari" if len(calls) == 1 else "main.py - Editor"

    with mock.patch.object(screen.sys, "platform", "darwin"), \
         mock.patch.object(screen, "_run", fake_run):
        got = screen.active_window()
    check("app and title come back", got == ("Safari", "main.py - Editor"), got)
    check("it asked osascript twice", len(calls) == 2, calls)


def linux_reading():
    print("\n--- active_window: Linux reads xdotool and /proc ---")

    def fake_run(args):
        return "main.py - Editor" if "getwindowname" in args else "42"

    with mock.patch.object(screen.sys, "platform", "linux"), \
         mock.patch.object(screen, "_run", fake_run), \
         mock.patch("builtins.open", mock.mock_open(read_data="code\n")):
        got = screen.active_window()
    check("app comes from /proc, title from xdotool",
          got == ("code", "main.py - Editor"), got)

    def no_pid(args):
        return "main.py - Editor" if "getwindowname" in args else ""

    with mock.patch.object(screen.sys, "platform", "linux"), \
         mock.patch.object(screen, "_run", no_pid):
        got = screen.active_window()
    check("a non-numeric pid leaves the app blank",
          got == ("", "main.py - Editor"), got)

    with mock.patch.object(screen.sys, "platform", "linux"), \
         mock.patch.object(screen, "_run", fake_run), \
         mock.patch("builtins.open", side_effect=OSError("no /proc")):
        got = screen.active_window()
    check("a missing /proc entry leaves the app blank",
          got == ("", "main.py - Editor"), got)


def run_helper():
    print("\n--- _run: a helper failure is never worth raising ---")
    ok = mock.Mock(returncode=0, stdout="hello\n")
    with mock.patch.object(screen.subprocess, "run", return_value=ok):
        check("stdout is stripped", screen._run(["x"]) == "hello")
    bad = mock.Mock(returncode=1, stdout="oops")
    with mock.patch.object(screen.subprocess, "run", return_value=bad):
        check("a nonzero exit reads as nothing", screen._run(["x"]) == "")
    with mock.patch.object(screen.subprocess, "run",
                           side_effect=OSError("no such tool")):
        check("a missing helper reads as nothing", screen._run(["x"]) == "")


def dispatch_and_errors():
    print("\n--- active_window: a backend that raises is not fatal ---")
    with mock.patch.object(screen.sys, "platform", "win32"), \
         mock.patch.object(screen, "_window_windows",
                           side_effect=RuntimeError("gdi down")):
        got = screen.active_window()
    check("a raising backend reads as unknown", got == ("", ""), got)


def capture_paths():
    print("\n--- capture: one screenshot ---")
    seen = {}

    class _Grab:
        monitors = [{"top": 0}, {"top": 1}]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def grab(self, mon):
            seen["monitor"] = mon
            return types.SimpleNamespace(rgb=b"rgb", size=(1, 1))

    def to_png(rgb, size):
        seen["png"] = (rgb, size)
        return b"PNG"

    old_mss, old_tools = screen.mss, screen.mss_tools
    screen.mss = types.SimpleNamespace(mss=lambda: _Grab())
    screen.mss_tools = types.SimpleNamespace(to_png=to_png)
    try:
        check("available when a backend is present", screen.available() is True)
        check("capture returns the png bytes", screen.capture() == b"PNG")
        check("the encoder got the raw image and size",
              seen.get("png") == (b"rgb", (1, 1)), seen)
        check("it grabbed the primary monitor",
              seen.get("monitor") == {"top": 1}, seen)
    finally:
        screen.mss, screen.mss_tools = old_mss, old_tools

    class _Bad:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def grab(self, mon):
            raise RuntimeError("no display")

    old_mss, old_tools = screen.mss, screen.mss_tools
    screen.mss = types.SimpleNamespace(mss=lambda: _Bad())
    screen.mss_tools = types.SimpleNamespace(to_png=lambda *a: b"")
    try:
        try:
            screen.capture()
            raised = False
        except RuntimeError:
            raised = True
        check("a grab failure becomes a RuntimeError the caller can speak",
              raised)
    finally:
        screen.mss, screen.mss_tools = old_mss, old_tools


def import_failures():
    print("\n--- a machine with no mss still imports and looks safely ---")
    saved = sys.modules.get("mss", "MISSING")
    saved_tools = sys.modules.get("mss.tools", "MISSING")
    sys.modules["mss"] = None
    try:
        reloaded = importlib.reload(screen)
        check("available() is False without mss",
              reloaded.available() is False)
        check("the reason is kept", reloaded._SCREEN_ERROR is not None)
        raised = None
        try:
            reloaded.capture()
        except RuntimeError as exc:
            raised = exc
        check("capture refuses with a RuntimeError", raised is not None)
    finally:
        if saved == "MISSING":
            sys.modules.pop("mss", None)
        else:
            sys.modules["mss"] = saved
        if saved_tools == "MISSING":
            sys.modules.pop("mss.tools", None)
        else:
            sys.modules["mss.tools"] = saved_tools
        importlib.reload(screen)
    check("reloading restored a working screen", screen.available() is True)


def watch_loop():
    print("\n--- watch: only a real change is reported ---")
    seq = iter([("", ""), ("code.exe", "main.py"), ("code.exe", "main.py"),
                RuntimeError("boom"), ("code.exe", "other.py")])
    seen = []

    def fake_active():
        try:
            item = next(seq)
        except StopIteration:
            return ("code.exe", "other.py")
        if isinstance(item, Exception):
            raise item
        return item

    async def drive():
        old = screen.active_window
        screen.active_window = fake_active
        try:
            task = asyncio.ensure_future(screen.watch(0.001, seen.append))
            for _ in range(500):
                await asyncio.sleep(0.005)
                if len(seen) >= 2:
                    break
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        finally:
            screen.active_window = old

    asyncio.run(drive())
    check("only the two real changes were reported",
          seen == ["code.exe \u2014 main.py", "code.exe \u2014 other.py"], seen)


print("=" * 66)
print("SCREEN (offline, fake display)")
print("=" * 66)
describe_lines()
windows_reading()
macos_reading()
linux_reading()
run_helper()
dispatch_and_errors()
capture_paths()
import_failures()
watch_loop()

print("\n" + "=" * 66)
if FAILURES:
    print("SCREEN FAILURES: %s" % (FAILURES,))
    sys.exit(1)
print("SCREEN OK")
