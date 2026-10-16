"""Every text file in this repo must be clean UTF-8: no byte-order mark,
no double-encoded text.

Both failures have shipped here and both were silent. ears.py grew a BOM
at byte 0 (an editor saved it that way), which `ast.parse` refuses to
compile -- so one file in the package could not be imported at all until
it was found by hand. And five of its lines had an em-dash re-read as
Windows-1252 and written back, so three characters of plain punctuation
became six bytes of mojibake, showing up in spoken text and in logs.

This scans instead of trusting, because neither defect is visible in a
normal editor (it renders the mojibake as punctuation or hides the BOM).

The mojibake test works by the round-trip: if a line can be encoded as
cp1252 and those bytes then decode as UTF-8 as something DIFFERENT, the
line is text that was already mangled once. Plain ASCII round-trips to
itself, and any line holding a script cp1252 lacks (CJK, Devanagari,
Tamil) cannot encode at all, so real multilingual text never trips it.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtalk.config import REPO  # noqa: E402

TEXT_EXT = {".py", ".md", ".json", ".toml", ".txt", ".sh", ".bat",
            ".example", ".cfg", ".ini", ".html", ".js", ".css", ".yml"}
SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", ".pytest_cache",
             "dist", "build", "htmlcov", ".ruff_cache"}


def check(name, cond, detail=""):
    print("    %s %s%s" % ("ok  " if cond else "FAIL", name,
                           ("  " + str(detail)) if detail else ""))
    if not cond:
        FAILURES.append(name)
    return cond


def files():
    for path in Path(REPO).rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXT:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def mojibake_line(line):
    try:
        raw = line.encode("cp1252")
    except UnicodeEncodeError:
        return False            # holds scripts cp1252 cannot: not this defect
    try:
        return raw.decode("utf-8") != line
    except UnicodeDecodeError:
        return False            # ordinary UTF-8, as it should be


FAILURES = []


def test_files():
    print("\n--- clean UTF-8, no BOM, no double-encoding ---")
    bom, bad_utf8, mangled = [], [], []
    scanned = 0
    for path in files():
        scanned += 1
        rel = str(path.relative_to(REPO))
        try:
            data = path.read_bytes()
        except OSError as e:
            bad_utf8.append("%s (%s)" % (rel, e))
            continue
        if data.startswith(b"\xef\xbb\xbf") or b"\xef\xbb\xbf" in data[:8]:
            bom.append(rel)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as e:
            bad_utf8.append("%s (%s)" % (rel, e))
            continue
        if "\ufeff" in text:
            bom.append(rel)
        for n, line in enumerate(text.splitlines(), 1):
            if mojibake_line(line):
                mangled.append("%s:%d" % (rel, n))
    check("%d text files are readable UTF-8" % scanned,
          scanned > 0 and not bad_utf8, detail=", ".join(bad_utf8[:4]))
    check("no byte-order mark anywhere", not bom, detail=", ".join(bom[:6]))
    check("no double-encoded text", not mangled, detail=", ".join(mangled[:6]))


def test_known_good_is_really_good():
    print("\n--- the detector itself ---")
    check("plain ASCII is clean",
          not mojibake_line("simple ascii line"))
    check("genuine em-dash is clean",
          not mojibake_line("a real em-dash \u2014 here"))
    check("genuine accented UTF-8 is clean",
          not mojibake_line("Ol\u00e1, voc\u00ea"))
    check("CJK is clean",
          not mojibake_line("\u4f60\u597d\u3002"))
    check("Devanagari is clean",
          not mojibake_line("\u0928\u092e\u0938\u094d\u0924\u0947"))
    check("the double-encoded em-dash is caught",
          mojibake_line("a \u00e2\u20ac\u2014 here"))
    check("the double-encoded curly quote is caught",
          mojibake_line("don\u2019t".encode("utf-8").decode("cp1252")))


print("=" * 66)
print("TEXT ENCODING")
print("=" * 66)

test_files()
test_known_good_is_really_good()

print("\n" + "=" * 66)
print("ENCODING OK" if not FAILURES else "ENCODING FAILURES: %s" % FAILURES)
sys.exit(1 if FAILURES else 0)
