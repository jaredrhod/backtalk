"""Test the session journal: what it writes, what it refuses to write, and
what it does when the brain is slow or broken."""
from pathlib import Path
import asyncio
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtalk.journal import Journal, summarize, MAX_EVENTS  # noqa: E402


def check(name, cond, detail=""):
    print("    %s %s%s" % ("ok  " if cond else "FAIL", name,
                           ("  " + str(detail)) if detail else ""))
    if not cond:
        FAILURES.append(name)
    return cond


FAILURES = []


def test_off_by_default():
    print("\n--- journaling is off unless configured ---")
    j = Journal(None)
    check("inactive with no directory", not j.active)
    j.say_user("hello")
    j.say_agent("hi")
    check("records nothing while off", j.events == [])
    check("writes nothing while off", j.write("summary") is None)


def test_empty_session_writes_nothing():
    print("\n--- an empty session leaves no file ---")
    with tempfile.TemporaryDirectory() as d:
        j = Journal(d)
        check("active with a directory", j.active)
        check("no file for zero turns", j.write() is None)
        check("directory stayed empty", list(Path(d).iterdir()) == [])


def test_write_contents():
    print("\n--- a real session writes a real entry ---")
    with tempfile.TemporaryDirectory() as d:
        j = Journal(d, agent_name="Seyon")
        j.say_user("  open   notepad  ")
        j.say_agent("Opening it now.")
        j.say_user("thanks")
        j.say_agent("Any time.")
        check("transcript pairs both sides",
              j.transcript()
              == "you: open notepad\nseyon: Opening it now.\n"
                 "you: thanks\nseyon: Any time.")
        path = j.write("We opened Notepad.")
        if not check("wrote a file", path is not None):
            return
        check("file exists", path.exists())
        body = path.read_text(encoding="utf-8")
        check("has the summary", "We opened Notepad." in body)
        check("has the transcript heading", "## Transcript" in body)
        check("has you", "**you:** open notepad" in body)
        check("has the agent name", "**Seyon:** Opening it now." in body)
        check("collapsed the extra spaces", "open   notepad" not in body)
        check("counts the turns", "- turns: 2" in body)
        check("says who wrote it", "written by: backtalk" in body)
        check("names the file by timestamp",
              path.name.startswith("voice-session_")
              and path.name.endswith(".md"), path.name)


def test_blank_utterances_ignored():
    print("\n--- silence is not recorded ---")
    with tempfile.TemporaryDirectory() as d:
        j = Journal(d)
        j.say_user("")
        j.say_user("   ")
        j.say_agent(None)
        check("blank lines dropped", j.events == [])
        j.say_user("real")
        check("real lines kept", j.turns == 1)


def test_cap_on_a_long_session():
    print("\n--- a marathon session cannot produce an unbounded file ---")
    with tempfile.TemporaryDirectory() as d:
        j = Journal(d)
        for i in range(MAX_EVENTS + 500):
            j.say_user("line %d" % i)
        check("capped at MAX_EVENTS", len(j.events) == MAX_EVENTS,
              len(j.events))


def test_two_sessions_do_not_collide():
    print("\n--- two sessions in the same minute both survive ---")
    with tempfile.TemporaryDirectory() as d:
        a = Journal(d)
        a.say_user("first")
        b = Journal(d)
        b.say_user("second")
        pa, pb = a.write(), b.write()
        check("distinct paths", pa != pb, "%s vs %s" % (pa, pb))
        check("both exist", pa.exists() and pb.exists())


def test_unwritable_directory_does_not_raise():
    print("\n--- an unwritable target must not raise ---")
    # A path under a file, not a directory: mkdir cannot succeed.
    with tempfile.TemporaryDirectory() as d:
        blocker = Path(d) / "notadir"
        blocker.write_text("x")
        j = Journal(blocker / "sub")
        j.say_user("hello")
        try:
            out = j.write("s")
            check("returns None instead of raising", out is None, out)
        except Exception as e:
            check("returns None instead of raising", False,
                  "%s: %s" % (type(e).__name__, e))


def test_summary_from_a_fake_brain():
    print("\n--- summarising uses the ask interface ---")
    seen = {}

    async def fake_ask(prompt):
        seen["prompt"] = prompt
        yield "We talked about Notepad. "
        yield "Nothing surprising."

    got = asyncio.run(summarize(fake_ask, "you: open notepad"))
    check("returns the joined summary",
          got == "We talked about Notepad. Nothing surprising.", repr(got))
    check("prompt carries the transcript", "open notepad" in seen["prompt"])


def test_summary_survives_a_broken_brain():
    print("\n--- a broken or slow brain costs the summary, not the entry ---")
    async def boom(prompt):
        raise RuntimeError("brain is down")
        yield ""  # pragma: no cover

    got = asyncio.run(summarize(boom, "you: hello"))
    check("broken brain -> empty summary", got == "", repr(got))

    async def slow(prompt):
        await asyncio.sleep(5)
        yield "too late"  # pragma: no cover

    got = asyncio.run(summarize(slow, "you: hello", timeout=0.2))
    check("slow brain -> empty summary, does not hang", got == "")

    got = asyncio.run(summarize(boom, "   "))
    check("empty transcript -> no brain call at all", got == "")


def test_entry_without_summary_is_still_useful():
    print("\n--- an entry with no summary is still an entry ---")
    with tempfile.TemporaryDirectory() as d:
        j = Journal(d)
        j.say_user("what is two plus two")
        j.say_agent("Four.")
        path = j.write("")            # summariser failed
        body = path.read_text(encoding="utf-8")
        check("still written", path.exists())
        check("has the transcript", "**you:** what is two plus two" in body)
        check("no empty summary section", "## What happened" not in body)


def test_activity_is_its_own_role():
    print("\n--- the activity tracker's notes are their own role ---")
    with tempfile.TemporaryDirectory() as d:
        j = Journal(d, agent_name="Seyon")
        j.say_user("open notepad")
        j.say_agent("Opening it now.")
        j.say_activity("notepad.exe \u2014 notes.txt")
        check("the transcript shows the activity line",
              "activity: notepad.exe" in j.transcript(), j.transcript())
        check("activity is not counted as a turn", j.turns == 1, j.turns)
        body = j.render()
        check("render marks the activity as its own role",
              "**activity:** notepad.exe" in body, body)
        check("render keeps the conversation",
              "**you:** open notepad" in body
              and "**Seyon:** Opening it now." in body, body)


print("=" * 66)
print("SESSION JOURNAL")
print("=" * 66)

test_off_by_default()
test_empty_session_writes_nothing()
test_write_contents()
test_blank_utterances_ignored()
test_cap_on_a_long_session()
test_two_sessions_do_not_collide()
test_unwritable_directory_does_not_raise()
test_summary_from_a_fake_brain()
test_summary_survives_a_broken_brain()
test_entry_without_summary_is_still_useful()
test_activity_is_its_own_role()

print("\n" + "=" * 66)
print("JOURNAL OK" if not FAILURES else "JOURNAL FAILURES: %s" % FAILURES)

# Exit nonzero on failure so a caller reading the exit code is not told a
# broken journal passed.
sys.exit(1 if FAILURES else 0)