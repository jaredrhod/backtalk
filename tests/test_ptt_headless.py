"""Hold-to-talk on a machine with no keyboard backend.

pynput hooks a global key through the desktop session. On a headless box
(a server, or CI) its import raises "failed to acquire X connection" --
which used to take the whole voice line down at import time, because
`from backtalk import main` pulls in ptt. A missing keyboard hook must be
optional: importing main still works, and asking for a key listener fails
cleanly with advice instead of a raw ImportError. Both are pinned here by
forcing the no-backend state.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtalk import main as btmain          # noqa: E402  import must survive
from backtalk import ptt                      # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print("    %s %s%s" % ("ok  " if cond else "FAIL", name,
                           ("  " + str(detail)) if detail else ""))
    if not cond:
        FAILURES.append(name)
    return cond


print("=" * 66)
print("HOLD TO TALK WITHOUT A KEYBOARD BACKEND")
print("=" * 66)

check("importing main survives a missing keyboard", btmain is not None)

saved = ptt.keyboard
try:
    ptt.keyboard = None
    check("resolve_key falls back to the plain name",
          ptt.resolve_key("home") == "home")
    raised = None
    try:
        ptt.PTTListener("home")
    except RuntimeError as e:
        raised = e
    check("PTTListener raises a clear RuntimeError", raised is not None, raised)
    check("the message points at the open mic",
          raised is not None and "open" in str(raised).lower(), raised)
finally:
    ptt.keyboard = saved

print("\n" + "=" * 66)
print("PTT HEADLESS OK" if not FAILURES
      else "PTT HEADLESS FAILURES: %s" % FAILURES)
sys.exit(1 if FAILURES else 0)
