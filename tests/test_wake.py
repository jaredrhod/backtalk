"""Test the wake word: recognising "hey seyon" however whisper spells it,
stripping it without damaging the command, and the listen-until-woken loop.

Wake is English-only by design: the Tamil wake forms and the virama-
tolerant matcher that used to catch whisper's spelling of them are gone,
so the shipped phrases must be Latin."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtalk.config import DEFAULTS  # noqa: E402
from backtalk.ears import Ears, is_wake, strip_wake  # noqa: E402

PHRASES = ["hey seyon", "hey sayon", "hey sean", "hey siyon"]


def check(name, cond, detail=""):
    print("    %s %s%s" % ("ok  " if cond else "FAIL", name,
                           ("  " + str(detail)) if detail else ""))
    if not cond:
        FAILURES.append(name)
    return cond


FAILURES = []


def test_recognition():
    print("\n--- wake phrases are recognised ---")
    check("plain", is_wake("hey seyon", PHRASES))
    check("punctuated", is_wake("Hey, Seyon!", PHRASES))
    check("upper case", is_wake("HEY SEYON", PHRASES))
    check("whisper's spelling", is_wake("hey sayon", PHRASES))
    check("leading a command", is_wake("hey seyon what's the weather",
                                       PHRASES))
    check("command then phrase", is_wake("tell me, hey seyon", PHRASES))


def test_no_false_positives():
    print("\n--- near misses do not wake it ---")
    check("bare name", not is_wake("seyon", PHRASES))
    check("inside a word", not is_wake("the seyonic field", PHRASES))
    check("unrelated", not is_wake("hello there", PHRASES))
    check("empty", not is_wake("", PHRASES))
    check("None", not is_wake(None, PHRASES))


def test_strip_preserves_command():
    print("\n--- stripping leaves the command intact ---")
    check("comma form",
          strip_wake("Hey Seyon, what's the weather", PHRASES)
          == "what's the weather")
    check("bare phrase -> empty",
          strip_wake("hey seyon", PHRASES) == "")
    check("no phrase is a no-op",
          strip_wake("just a question", PHRASES) == "just a question")
    check("casing kept",
          strip_wake("Hey Seyon tell me about Paris", PHRASES)
          == "tell me about Paris")


def test_default_phrases_are_english():
    print("\n--- the shipped wake phrases are Latin, not Tamil ---")
    phrases = DEFAULTS["wake_word_phrases"]
    check("the defaults carry phrases", bool(phrases))
    check("every default phrase is ASCII",
          all(p.isascii() for p in phrases),
          [p for p in phrases if not p.isascii()])
    check("an English default wakes", is_wake("hey seyon", phrases))
    check("every default carries a greeting, not the bare name",
          all(len(p.split()) >= 2 for p in phrases), phrases)
    check("the bare name alone does not wake it",
          not is_wake("seyon", phrases))
    check("the name in a sentence does not wake it",
          not is_wake("...so seyon said hello", phrases))


def test_wait_for_wake_loop():
    print("\n--- wait_for_wake ignores speech until the phrase ---")
    import backtalk.ears as ears_mod
    logged = []
    ears_mod.log = lambda msg: logged.append(msg)
    e = Ears()
    heard = ["what time is it", "no idea", "hey seyon"]
    e.listen_once = lambda gate=None, abort=None, timeout_s=None, on_partial=None, owner_check=None: (
        heard.pop(0) if heard else None)
    woke = e.wait_for_wake(PHRASES)
    check("returns the waking utterance", woke == "hey seyon")
    check("near misses are logged for tuning",
          any("what time is it" in m for m in logged) and
          any("no idea" in m for m in logged),
          logged)

    print("\n--- abort stops it ---")
    e2 = Ears()
    e2.listen_once = lambda gate=None, abort=None, timeout_s=None, on_partial=None, owner_check=None: None
    check("abort -> None", e2.wait_for_wake(PHRASES) is None)


def test_wait_for_wake_command_in_breath():
    print("\n--- a combined breath is used as the command ---")
    e = Ears()
    e.listen_once = lambda gate=None, abort=None, timeout_s=None, on_partial=None, owner_check=None: (
        "Hey Seyon, what's the weather")
    woke = e.wait_for_wake(PHRASES)
    check("combined utterance comes back whole", woke is not None)
    check("strip yields the command",
          strip_wake(woke, PHRASES) == "what's the weather")


def test_blank_phrase_matches_nothing():
    print("\n--- a blank wake phrase is not a wildcard ---")
    check("a blank phrase alone does not wake", not is_wake("hello there", [""]))
    check("blank among real phrases still matches the real one",
          is_wake("hey seyon", ["", "hey seyon"]))
    check("stripping a blank phrase is a no-op",
          strip_wake("hello", [""]) == "hello")


print("=" * 66)
print("WAKE WORD")
print("=" * 66)

test_recognition()
test_no_false_positives()
test_blank_phrase_matches_nothing()
test_default_phrases_are_english()
test_strip_preserves_command()
test_wait_for_wake_loop()
test_wait_for_wake_command_in_breath()

print("\n" + "=" * 66)
print("WAKE OK" if not FAILURES else "WAKE FAILURES: %s" % FAILURES)
sys.exit(1 if FAILURES else 0)
