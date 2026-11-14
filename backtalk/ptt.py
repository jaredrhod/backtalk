# backtalk: talk to your Claude Code agent out loud.
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
"""Hold-to-talk — a global key listener.

HOLD the key -> mic opens. RELEASE -> mic closes and the utterance is
processed. The button IS the voice-activity detector, which is why this
mode is speaker-safe with no headphones: the mic simply isn't open while
the assistant talks, unless you press the key — and pressing while it
talks interrupts it.

THE KEY-REPEAT TRAP (the bug that kills every naive build): the OS fires
on_press events CONTINUOUSLY while a key is held. Without the held-state
filter below, every repeat reads as a fresh press and keeps cancelling
the reply before it can speak.

AND THE HALF THAT TRAP HIDES: some keyboards send auto-repeat as full
DOWN/UP PAIRS rather than the repeated DOWN-only stream. Filtering the
presses and trusting every release then breaks the OTHER way -- a single
hold is chopped into dozens of ~50ms recordings, each too short to
transcribe, and the whole thing is SILENT. No exception, no log line,
nothing to search for; it simply reads as "the microphone does not work".
Measured in the field on a Logitech MX Mechanical through a Bolt
receiver: one 2.6-second hold produced 186 key events and about fifty
recordings. So a release is never trusted on sight -- see _HeldKeyState.

macOS needs Input Monitoring permission for the hosting terminal
(System Settings -> Privacy & Security -> Input Monitoring). Windows
works out of the box. Linux is the one platform with two real backends:

PYNPUT VS EVDEV, AND WHY THERE ARE TWO: pynput's global grab is built on
Xlib, so it only works on an X11 session. Under Wayland the compositor's
security model blocks any app from grabbing keys system-wide -- and it
fails SILENTLY: no exception, no log line, the configured key simply
never fires, which reads exactly like a dead microphone. There is no way
to ask "did pynput actually attach" and get a straight answer, so this
module doesn't try to detect the failure after the fact -- it decides
the backend up front from `XDG_SESSION_TYPE` and routes Linux+Wayland to
an evdev backend that reads /dev/input directly, bypassing the
compositor's key-grab restriction entirely. That needs the user in the
`input` group (`sudo usermod -aG input $USER`, then log out and back in
-- group membership is resolved at login, not re-read live) since the
device nodes are `root:input 0660`.
"""
import os
import sys
import threading
import time


class _HeldKeyState:
    """The press/release debounce state machine, shared by every backend.

    Deliberately source-agnostic: it doesn't know or care whether a press
    came from pynput's Xlib hook or a raw evdev event, which means the
    DOWN/UP-pairs trap documented above gets handled once, here, instead
    of being re-solved (or re-broken) per backend.
    """
    # How long a release must stand unchallenged before it is believed.
    # Comfortably longer than any keyboard's auto-repeat period (measured
    # at ~50ms on the hardware that exposed this; Windows' fastest setting
    # is ~30ms) and short enough that letting go still feels instant.
    RELEASE_GRACE = 0.12

    def __init__(self):
        self._held = False
        self._release_t = None          # a release awaiting confirmation
        self._press_evt = threading.Event()

    def _press(self):
        # A press cancels any pending release: that release was auto-repeat,
        # not a human letting go.
        self._release_t = None
        if not self._held:                      # filter key-repeat
            self._held = True
            self._press_evt.set()

    def _release(self):
        # PROVISIONAL. Believed only if no press follows; see _settle().
        self._release_t = time.monotonic()

    def _settle(self):
        """Commit a release that has stood unchallenged for the grace window."""
        r = self._release_t
        if self._held and r is not None and \
                time.monotonic() - r >= self.RELEASE_GRACE:
            self._held = False
            self._release_t = None

    def wait_press(self):
        """Block until the key goes DOWN (one event per physical press)."""
        # Settled on a loop, not once. A release landing after the last
        # is_held() poll leaves _held provisionally True, and a single
        # settle-then-wait would then block forever: the next press is
        # filtered as key-repeat, so nothing ever sets the event again.
        while True:
            self._settle()
            if self._press_evt.wait(timeout=self.RELEASE_GRACE):
                self._press_evt.clear()
                return

    def is_held(self) -> bool:
        self._settle()
        return self._held


def resolve_key(name: str):
    """'home' / 'f13' / 'right_alt' / any single character -> pynput key."""
    from pynput import keyboard

    name = (name or "home").strip().lower()
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    # Friendly names -> pynput's names. pynput calls the right option key
    # alt_r, not right_alt; the docs speak human, this map translates.
    # (Field-caught: right_alt silently fell back to home, which Mac
    # laptops cannot press, so the voice looked healthy and never fired.)
    aliases = {
        "right_alt": "alt_r", "left_alt": "alt_l",
        "right_option": "alt_r", "left_option": "alt_l",
        "right_ctrl": "ctrl_r", "left_ctrl": "ctrl_l",
        "right_cmd": "cmd_r", "left_cmd": "cmd_l",
        "right_shift": "shift_r", "left_shift": "shift_l",
    }
    name = aliases.get(name, name)
    try:
        return getattr(keyboard.Key, name)
    except AttributeError:
        print(f"[ptt] unknown key {name!r} — falling back to 'home'",
              flush=True)
        return keyboard.Key.home


def resolve_evdev_key(name: str) -> int:
    """'home' / 'f13' / 'right_alt' / any single character -> an evdev
    KEY_* code. Same alias surface as resolve_key() above so a config's
    ptt_key means the same thing on every backend."""
    from evdev import ecodes

    name = (name or "home").strip().lower()
    aliases = {
        "right_alt": "RIGHTALT", "left_alt": "LEFTALT",
        "right_option": "RIGHTALT", "left_option": "LEFTALT",
        "right_ctrl": "RIGHTCTRL", "left_ctrl": "LEFTCTRL",
        "right_cmd": "RIGHTMETA", "left_cmd": "LEFTMETA",
        "right_shift": "RIGHTSHIFT", "left_shift": "LEFTSHIFT",
    }
    attr = "KEY_" + aliases.get(name, name).upper()
    try:
        return getattr(ecodes, attr)
    except AttributeError:
        print(f"[ptt] unknown key {name!r} — falling back to 'home'",
              flush=True)
        return ecodes.KEY_HOME


class _PynputPTTListener(_HeldKeyState):
    """X11, macOS, Windows — pynput's global hook works natively on all three."""

    def __init__(self, key="home"):
        super().__init__()
        from pynput import keyboard

        self._key = resolve_key(key) if isinstance(key, str) else key
        self._listener = keyboard.Listener(on_press=self._on_press,
                                           on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()

    def _on_press(self, k):
        if k == self._key:
            self._press()

    def _on_release(self, k):
        if k == self._key:
            self._release()


class _EvdevPTTListener(_HeldKeyState):
    """Linux under Wayland — reads /dev/input directly, bypassing the
    compositor's global-key-grab restriction that silently defeats pynput.

    Needs the user in the `input` group: the device nodes are
    `crw-rw---- root input`, so without that membership `InputDevice()`
    raises PermissionError on every matching device, which __init__
    surfaces as a clear RuntimeError instead of the caller finding out
    the hard way later that nothing ever fires.
    """

    def __init__(self, key="home"):
        super().__init__()
        from evdev import InputDevice, ecodes, list_devices

        self._ecodes = ecodes
        self._code = resolve_evdev_key(key) if isinstance(key, str) else key
        self._devices = []
        denied = 0
        for path in list_devices():
            try:
                dev = InputDevice(path)
            except PermissionError:
                denied += 1
                continue
            if self._code in dev.capabilities().get(ecodes.EV_KEY, []):
                self._devices.append(dev)
            else:
                dev.close()
        if not self._devices:
            hint = (" — found keyboard-like devices but couldn't open "
                    f"{denied} of them (permission denied); add this user "
                    "to the `input` group and log out/in again"
                    if denied else "")
            raise RuntimeError(
                f"[ptt] no accessible /dev/input device exposes key "
                f"{key!r}{hint}")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import select

        devices = {d.fd: d for d in self._devices}
        while devices:
            ready, _, _ = select.select(devices.keys(), [], [])
            for fd in ready:
                dev = devices[fd]
                try:
                    events = list(dev.read())
                except OSError:
                    # Device vanished (unplugged, suspend/resume). Drop it;
                    # keep listening on whatever else still matches the key.
                    del devices[fd]
                    continue
                for event in events:
                    if event.type != self._ecodes.EV_KEY or \
                            event.code != self._code:
                        continue
                    if event.value == 1:          # down
                        self._press()
                    elif event.value == 0:         # up
                        self._release()
                    # value == 2 is the kernel's own auto-repeat signal —
                    # already held, so it's a deliberate no-op here, same
                    # as the key-repeat filter in _HeldKeyState._press().


def PTTListener(key="home"):
    """Build the right backend for this platform/session and hand back
    something with wait_press()/is_held() — callers don't need to know
    which one they got.

    Not a class on purpose: Linux needs a runtime choice (session type),
    not a compile-time one, and this keeps call sites (`PTTListener(cfg)`)
    unchanged regardless of which backend answers.
    """
    if sys.platform.startswith("linux") and \
            os.environ.get("XDG_SESSION_TYPE") == "wayland":
        return _EvdevPTTListener(key)
    return _PynputPTTListener(key)
