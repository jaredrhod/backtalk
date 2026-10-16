"""Gentle self-care nudges about the machine the line is running on.

Off unless ``care_enabled`` is true. When it is on, a check-in picks up ONE
small, true thing about the person's machine -- a battery well down and off
the charger, a nearly-full disk, a long unbroken session -- and lets the
brain weave it into its own short hello. Nothing here ever speaks on its
own; care is only ever a colour on a hello the brain was already going to
give.

The decision is pure and easy to test. Only the three probes read the
machine, and they are the one part marked no-cover, so a laptop with a full
battery or a roomy disk can never fail the suite.
"""

from __future__ import annotations

import os
import shutil
import time

# A nudge repeats no more often than this, in seconds.
NAG_EVERY = 1800.0
# Battery at or below this, off the charger, is worth a word.
BATTERY_LOW = 20
# Free space under this many gigabytes is worth a word.
DISK_LOW_GB = 5.0
# A stretch at the machine this long, unbroken, is worth a word.
SESSION_LONG_MIN = 120.0


def battery_nudge(pct, on_ac):
    """A line when the battery is well down and unplugged, else None. Pure."""
    if on_ac or pct is None or pct > BATTERY_LOW:
        return None
    return (f"the battery is at {int(pct)} percent and the charger is "
            "not plugged in")


def disk_nudge(free_gb):
    """A line when the main disk is nearly full, else None. Pure."""
    if free_gb is None or free_gb >= DISK_LOW_GB:
        return None
    return ("the main disk is nearly full, about "
            f"{max(0, int(free_gb))} gigabytes left")


def session_nudge(minutes):
    """A line after a long unbroken stretch, else None. Pure."""
    if minutes is None or minutes < SESSION_LONG_MIN:
        return None
    return (f"they have been at this about {int(minutes // 60)} hours "
            "without a real break")


def facts(*, pct=None, on_ac=False, free_gb=None, minutes=None):
    """The candidate nudges for one machine state, in the order to prefer
    them: battery first, then a full disk, then a long session. Pure."""
    return (battery_nudge(pct, on_ac), disk_nudge(free_gb),
            session_nudge(minutes))


def pick(candidates):
    """The first candidate worth saying, or None. Pure: one nudge at a
    time, so a hello never turns into a health report."""
    for c in candidates:
        if c:
            return c
    return None


def due(now, last, every=NAG_EVERY):
    """True when enough quiet has passed since the last nudge. Pure."""
    return (now - last) >= every


def read_battery():  # pragma: no cover -- reads the real machine
    """(percent, on_ac) for this machine, or (None, False) if unknown.

    No dependency is added for this: psutil may not be installed, so the
    probe reads what the OS already exposes and stays silent when it can't.
    """
    try:
        import psutil  # type: ignore

        b = psutil.sensors_battery()
        if b is None:
            return (None, False)
        return (float(b.percent), bool(b.power_plugged))
    except Exception:
        return (None, False)


def read_disk_gb():  # pragma: no cover -- reads the real machine
    """Free gigabytes on the volume backtalk lives on, or None."""
    try:
        return shutil.disk_usage(os.path.abspath(os.sep)).free / 1e9
    except Exception:
        return None


def current_nudge(now=None, last=0.0, every=NAG_EVERY):  # pragma: no cover
    """Read the machine and return (line, now) when a nudge is due, else
    (None, last). The pure pieces above do the thinking; this only wires the
    probes to them, so it is the one function kept out of coverage."""
    now = time.monotonic() if now is None else now
    if not due(now, last, every):
        return (None, last)
    pct, on_ac = read_battery()
    line = pick(facts(pct=pct, on_ac=on_ac, free_gb=read_disk_gb(),
                      minutes=_session_minutes()))
    return (line, now if line else last)


def _session_minutes():  # pragma: no cover -- tracks the real uptime
    """Minutes since this voice line started, best-effort."""
    try:
        return (time.monotonic() - _SESSION_START) / 60.0
    except Exception:
        return None


_SESSION_START = time.monotonic()  # pragma: no cover
