"""Recall, isolated: searching the journal by meaning-ish.

recall.py is pure Python over markdown files, so it is driven end to end
against a temp journal. What is checked is the thing the voice line leans
on: the right entry wins a query, the newest is listed first, the date
comes from the filename, the snippet reads clean, and a question with no
answer still returns something to say.
"""
import os
from pathlib import Path
import re
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtalk import recall                                 # noqa: E402

failures = []


def check(name, cond, detail=""):
    print("    %s %s%s" % ("ok  " if cond else "FAIL", name,
                           "" if cond else "   <- " + str(detail)))
    if not cond:
        failures.append(name)


_TMP = tempfile.mkdtemp(prefix="bt_recall_")

_A = ("# Voice session, 2026-10-01 09:00\n\n## Transcript\n\n"
      "**you:** we should use postgres for the billing database\n"
      "**seyon:** agreed, postgres it is\n")
_B = ("# Voice session, 2026-10-10 15:30\n\n## Transcript\n\n"
      "**you:** the voice line latency is all in kokoro tts\n"
      "**seyon:** whisper stt is fast enough\n")
_C = ("# Voice session, 2026-10-05 12:00\n\n## Transcript\n\n"
      "**you:** let us go for a walk\n**seyon:** sounds good\n")
for name, body in (("voice-session_2026-10-01_09-00-00.md", _A),
                   ("voice-session_2026-10-10_15-30-00.md", _B),
                   ("voice-session_2026-10-05_12-00-00.md", _C)):
    with open(os.path.join(_TMP, name), "w", encoding="utf-8") as f:
        f.write(body)

print("=" * 66)
print("JOURNAL RECALL")
print("=" * 66)

toks = recall.tokenize("What did we say about the POSTGRES billing database?")
check("stopwords and short words drop out", "the" not in toks and "we" not in toks,
      toks)
check("content words survive", "postgres" in toks and "billing" in toks, toks)

entries = recall.load_entries(_TMP)
check("every entry is loaded", len(entries) == 3, len(entries))
check("entries come newest first",
      [e["when"] for e in entries]
      == ["2026-10-10 15:30", "2026-10-05 12:00", "2026-10-01 09:00"],
      [e["when"] for e in entries])

check("a missing directory is empty, not an error",
      recall.load_entries(os.path.join(_TMP, "nope")) == [])
check("no directory at all is empty",
      recall.load_entries("") == [])
check("the limit caps the load", len(recall.load_entries(_TMP, limit=2)) == 2)

# A path that is not readable as a file is skipped, not raised. On Windows
# and POSIX alike, a directory named like an entry makes read_text raise.
os.makedirs(os.path.join(_TMP, "voice-session_2020-01-01_00-00-00.md"))
check("an unreadable entry is skipped",
      len(recall.load_entries(_TMP)) == 3, len(recall.load_entries(_TMP)))

hits = recall.search("postgres billing database", _TMP)
check("the matching entry wins",
      hits and hits[0]["when"] == "2026-10-01 09:00", hits)
check("the snippet carries the words",
      hits and "postgres" in hits[0]["snippet"], hits)
check("markdown is stripped from the snippet",
      hits and "**" not in hits[0]["snippet"], hits)

hits2 = recall.search("voice line latency kokoro", _TMP)
check("a different topic finds its own session",
      hits2 and hits2[0]["when"] == "2026-10-10 15:30", hits2)

check("an unheard-of topic finds nothing",
      recall.search("zzzqqq", _TMP) == [])

check("empty query ranks nothing", recall.rank("", entries) == [])
check("no entries ranks nothing", recall.rank("postgres", []) == [])

check("an empty entry has an empty snippet", recall._snippet("", []) == "")
check("a query word picks its own line",
      recall._snippet("alpha\n**the postgres bit**\nbeta", ["postgres"])
      == "the postgres bit")
check("a snippet is trimmed to width",
      len(recall._snippet("x" * 300, ["x"])) == 180)

line = recall.spoken("postgres billing", _TMP)
check("a hit is spoken with its date", line.startswith("On 2026-10-01 09:00"),
      line)

miss = recall.spoken("zzzqqq", _TMP)
check("a miss is still an answer",
      miss == "I don't have anything in my journal about that.", miss)

# A file without the naming pattern falls back to the file's own clock.
stray = os.path.join(_TMP, "notes.md")
with open(stray, "w", encoding="utf-8") as f:
    f.write("a note with no timestamp in its name\n")
from pathlib import Path as _P                                    # noqa: E402
when = recall._when(_P(stray))
check("a stray file falls back to a real time",
      bool(re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", when)), when)

print("\n" + "=" * 66)
print("RECALL OK" if not failures else "RECALL FAILURES: %s" % failures)
sys.exit(1 if failures else 0)
