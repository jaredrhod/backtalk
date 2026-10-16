"""Test the session log: the terminal line, the timestamped file append,
and the fallbacks that keep a broken console or log file from taking the
voice down."""
import re
from pathlib import Path
import sys
import tempfile
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtalk import vlog  # noqa: E402


def check(name, cond, detail=""):
    print("    %s %s%s" % ("ok  " if cond else "FAIL", name,
                           ("  " + str(detail)) if detail else ""))
    if not cond:
        FAILURES.append(name)
    return cond


FAILURES = []


def test_prints_and_appends():
    print("\n--- a line reaches the console and the file ---")
    with tempfile.TemporaryDirectory() as d:
        logfile = Path(d) / "logs" / "backtalk.log"
        printed = []
        old = vlog.LOG_PATH
        vlog.LOG_PATH = logfile
        try:
            with mock.patch("builtins.print",
                            lambda *a, **k: printed.append(a[0])):
                vlog.log("hello world")
            check("printed to the console", printed == ["hello world"], printed)
            check("created the log directory", logfile.parent.is_dir())
            body = logfile.read_text(encoding="utf-8")
            check("wrote a timestamped line",
                  re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} "
                           r"hello world\n$", body) is not None, body)
        finally:
            vlog.LOG_PATH = old


def test_unicode_fallback():
    print("\n--- a console that refuses utf-8 gets ascii, not a crash ---")
    with tempfile.TemporaryDirectory() as d:
        old = vlog.LOG_PATH
        vlog.LOG_PATH = Path(d) / "b.log"
        seen = []
        calls = {"n": 0}

        def picky(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise UnicodeEncodeError("utf-8", "h\u00e9llo", 0, 1, "no")
            seen.append(a[0])

        try:
            with mock.patch("builtins.print", picky):
                vlog.log("h\u00e9llo")
            check("fell back to ascii and kept going",
                  seen == ["h?llo"], seen)
            check("the file still got the real text",
                  "h\u00e9llo" in (Path(d) / "b.log").read_text("utf-8"))
        finally:
            vlog.LOG_PATH = old


def test_broken_log_file_is_swallowed():
    print("\n--- an unwritable log file never raises ---")
    with tempfile.TemporaryDirectory() as d:
        old = vlog.LOG_PATH
        vlog.LOG_PATH = Path(d) / "b.log"
        try:
            with mock.patch.object(Path, "open", side_effect=OSError("locked")), \
                 mock.patch("builtins.print"):
                vlog.log("still fine")
            check("returned instead of raising", True)
        finally:
            vlog.LOG_PATH = old


def test_init_console_off_windows_is_a_noop():
    print("\n--- init_console does nothing off windows ---")
    fake = mock.Mock()
    with mock.patch.object(vlog.sys, "platform", "linux"), \
         mock.patch.object(vlog.sys, "stdout", fake), \
         mock.patch.object(vlog.sys, "stderr", mock.Mock()):
        vlog._init_console()
    check("left the streams untouched", not fake.reconfigure.called)


def test_init_console_survives_a_stubborn_stream():
    print("\n--- init_console swallows a stream that will not reconfigure ---")
    class Boom:
        def reconfigure(self, **kwargs):
            raise OSError("no")

    class WorkingCtypes:
        class windll:
            class kernel32:
                @staticmethod
                def SetConsoleOutputCP(code):
                    return 0

                @staticmethod
                def SetConsoleCP(code):
                    return 0

    with mock.patch.object(vlog.sys, "platform", "win32"), \
         mock.patch.dict(sys.modules, {"ctypes": WorkingCtypes}), \
         mock.patch.object(vlog.sys, "stdout", Boom()), \
         mock.patch.object(vlog.sys, "stderr", Boom()):
        vlog._init_console()
    check("returned instead of raising", True)


def test_init_console_survives_broken_ctypes():
    print("\n--- init_console swallows a ctypes that will not set up ---")

    class FakeCtypes:
        class windll:
            class kernel32:
                @staticmethod
                def SetConsoleOutputCP(code):
                    raise OSError("no")

    fake = mock.Mock()
    with mock.patch.object(vlog.sys, "platform", "win32"), \
         mock.patch.dict(sys.modules, {"ctypes": FakeCtypes}), \
         mock.patch.object(vlog.sys, "stdout", fake), \
         mock.patch.object(vlog.sys, "stderr", mock.Mock()):
        vlog._init_console()
    check("kept going and tuned the streams anyway",
          fake.reconfigure.called)


print("=" * 66)
print("SESSION LOG (vlog)")
print("=" * 66)

test_prints_and_appends()
test_unicode_fallback()
test_broken_log_file_is_swallowed()
test_init_console_off_windows_is_a_noop()
test_init_console_survives_a_stubborn_stream()
test_init_console_survives_broken_ctypes()

print("\n" + "=" * 66)
print("VLOG OK" if not FAILURES else "VLOG FAILURES: %s" % FAILURES)

sys.exit(1 if FAILURES else 0)
