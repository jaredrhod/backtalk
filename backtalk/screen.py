# backtalk: talk to your opencode agent out loud.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Screen awareness, in two deliberately separate halves.

The first half is a LOOK: when the user says "look at my screen" (or one of
its phrases) the voice line grabs one screenshot and sends it to the brain
along with the question, so the answer can be about what is actually on the
display. That is the whole of it -- a picture is taken when, and only when,
a look is asked for.

The second half is an ACTIVITY JOURNAL: a low-frequency sample of which
window is focused ("notepad.exe -- notes.txt"), recorded in the session
journal and published on the signal bus so a face can show it. It is OFF by
default (`screen_enabled`), local only, and it records a LABEL -- never a
picture, never keystrokes. It is a diary of which app you were in, not a
record of what you typed.

The look works on every platform mss does. The activity label is
best-effort: Windows reads the foreground window through user32, macOS asks
System Events, Linux shells out to xdotool. Any of them can be missing --
a headless box has no focused window at all -- so every failure reads as
("", "") and the tracker simply reports nothing rather than crashing.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
import os
import subprocess
import sys
from typing import Any

# mss ships no type information; keep it Any so pyright stops at the edge.
mss: Any = None
mss_tools: Any = None
try:
    import mss as mss
    import mss.tools as mss_tools
except Exception as _err:      # noqa: BLE001 -- any import failure is "no screen"
    mss = None
    mss_tools = None
    _SCREEN_ERROR: BaseException | None = _err
else:
    _SCREEN_ERROR = None


def available() -> bool:
    """Whether a screenshot can be taken at all on this machine."""
    return mss is not None and mss_tools is not None


def capture() -> bytes:
    """Grab the primary monitor once and return it as PNG bytes.

    Raises RuntimeError when there is no backend -- the caller speaks the
    failure instead of the turn dying on an AttributeError.
    """
    if mss is None or mss_tools is None:
        raise RuntimeError("mss is not available: %s" % (_SCREEN_ERROR,))
    try:
        with mss.mss() as sct:
            shot = sct.grab(sct.monitors[1])
            return mss_tools.to_png(shot.rgb, shot.size)
    except Exception as e:      # no display, no monitor, a dead session
        raise RuntimeError("could not grab the screen: %s" % (e,)) from e


def _run(args: list[str]) -> str:
    """Run a helper and return its stdout, stripped. "" on any failure --
    a window title is never worth raising for."""
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""
    if out.returncode != 0:
        return ""
    return (out.stdout or "").strip()


def _window_windows() -> tuple[str, str]:
    import ctypes
    user32 = ctypes.windll.user32            # pyright: ignore[reportAttributeAccessIssue]
    kernel32 = ctypes.windll.kernel32        # pyright: ignore[reportAttributeAccessIssue]
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return "", ""
    length = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    pid = ctypes.c_ulong()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    app = ""
    handle = kernel32.OpenProcess(0x1000, False, pid.value)   # QUERY_LIMITED
    if handle:
        path = ctypes.create_unicode_buffer(260)
        size = ctypes.c_ulong(260)
        kernel32.QueryFullProcessImageNameW(handle, 0, path,
                                            ctypes.byref(size))
        app = os.path.basename(path.value)
        kernel32.CloseHandle(handle)
    return app, buf.value.strip()


def _window_macos() -> tuple[str, str]:
    app = _run(["osascript", "-e",
                'tell application "System Events" to get name of first '
                'application process whose frontmost is true'])
    title = _run(["osascript", "-e",
                  'tell application "System Events" to get name of front '
                  'window of (first application process whose frontmost '
                  'is true)'])
    return app, title


def _window_linux() -> tuple[str, str]:
    title = _run(["xdotool", "getactivewindow", "getwindowname"])
    pid = _run(["xdotool", "getactivewindow", "getwindowpid"])
    app = ""
    if pid.isdigit():
        try:
            with open("/proc/%s/comm" % pid) as f:
                app = f.read().strip()
        except OSError:
            app = ""
    return app, title


def active_window() -> tuple[str, str]:
    """(app, title) of the focused window, or ("", "") when unknown.

    Never raises: a headless box, a missing helper, or a refused
    permission all read as "nothing to report" rather than killing the
    tracker with a traceback.
    """
    try:
        if sys.platform == "win32":
            return _window_windows()
        if sys.platform == "darwin":
            return _window_macos()
        return _window_linux()
    except Exception:
        return "", ""


_DASH = " \u2014 "


def describe(app: str, title: str) -> str:
    """One line for the journal and the bus: "App -- window title"."""
    app = (app or "").strip()
    title = (title or "").strip()
    if app and title:
        return app + _DASH + title
    return app or title


async def watch(interval: float, on_change: Callable[[str], None]) -> None:
    """Sample the focused window every `interval` seconds and hand
    on_change(label) each CHANGED label. Runs until cancelled.

    Repeats and blanks are skipped: recording the same window over and
    over would drown the journal, and only a real change is news.
    """
    last = ""
    while True:
        await asyncio.sleep(interval)
        try:
            label = describe(*active_window())
        except Exception:
            label = ""
        if not label or label == last:
            continue
        last = label
        on_change(label)
