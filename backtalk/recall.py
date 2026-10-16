# backtalk: talk to your opencode agent out loud.
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
"""Recall — asking your own journal "when did we talk about that?".

journal.py writes one markdown file per session. This reads them back by
meaning, so "when did we talk about the voice latency" can land on the
session where it was actually discussed instead of the newest one.

It is honest about what "meaning" is here: a BM25 ranking over words, not
neural embeddings. That is deliberate. A neural model is hundreds of
megabytes and a startup stall, and backtalk already ships with no such
dependency; a weighted word search finds a past session from a spoken
phrase well enough to be worth having, costs nothing, and runs on the same
CPU the voice line already uses. When it is wrong it is visibly wrong, which
is the honest kind of wrong.

Pure Python, no model, no network, so the whole thing is testable offline.
"""
from __future__ import annotations

import datetime
import math
import os
import re
from pathlib import Path

# The words that carry no topic. A query of "what did we say about the
# database" must rank a database entry on "database", not on "the".
STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can cannot could couldn't
did didn't do does doesn't doing don't down during each few for from further
had hadn't has hasn't have haven't having he her here hers herself him himself
his how i if in into is isn't it its itself just let's me more most mustn't my
myself no nor not of off on once only or other ought our ours ourselves out
over own same shan't she should shouldn't so some such than that the their
theirs them themselves then there these they this those through to too under
until up very was wasn't we were weren't what when where which while who whom
why with won't would wouldn't you your yours yourself yourselves okay ok yeah
yes
""".split())

# Journal filenames carry their own timestamp; prefer it to the clock on
# disk, which a copy or a sync will have changed.
_FNAME = re.compile(r"voice-session[_-](\d{4}-\d{2}-\d{2})[_-]"
                    r"(\d{2}-\d{2}-\d{2})")
_WORD = re.compile(r"[a-z0-9]+")
MAX_ENTRIES = 200


def tokenize(text: str) -> list[str]:
    """The words a query or an entry should be compared on: lowercased
    alphanumerics, stopwords and one-letter words dropped. Pure."""
    return [w for w in _WORD.findall((text or "").lower())
            if len(w) > 1 and w not in STOPWORDS]


def _when(path: Path) -> str:
    """The moment an entry was written, read from its name when it has one,
    else from the file's own clock."""
    m = _FNAME.search(path.name)
    if m:
        date, clock = m.group(1), m.group(2)
        return f"{date} {clock[:5].replace('-', ':')}"
    stamp = datetime.datetime.fromtimestamp(path.stat().st_mtime)
    return stamp.strftime("%Y-%m-%d %H:%M")


def load_entries(directory: str | os.PathLike | None,
                 limit: int = MAX_ENTRIES) -> list[dict]:
    """Every journal entry in a folder, newest first, each with its moment
    and its tokenised text. A missing folder or an unreadable file is
    skipped, never raised: recall is a courtesy, not a critical path."""
    if not directory:
        return []
    root = Path(directory).expanduser()
    if not root.is_dir():
        return []
    out: list[dict] = []
    # Newest first: the name is a sortable timestamp, so a plain reverse
    # sort puts the latest session at the top without touching the clock.
    for path in sorted(root.glob("voice-session*.md"), reverse=True):
        if len(out) >= limit:
            break
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        out.append({"when": _when(path), "text": text,
                    "tokens": tokenize(text)})
    return out


def _snippet(text: str, query: list[str], width: int = 180) -> str:
    """The line of an entry that best answers the query, trimmed to width.

    Falling back to the entry's head when no line stands out keeps the
    answer from ever being empty."""
    best, hits = "", -1
    for line in (text or "").splitlines():
        low = line.lower()
        n = sum(1 for t in query if t in low)
        if n > hits and line.strip():
            best, hits = line.strip(), n
    best = best or (text or "").strip()
    # Strip the transcript's own markdown so the snippet reads aloud clean.
    best = best.replace("**", "").replace("#", "").strip()
    return best[:width].strip()


def rank(query: str, entries: list[dict], k: int = 3) -> list[dict]:
    """The best entries for a query, by BM25 over their tokens.

    k1/b are the usual BM25 constants: k1 how fast term frequency saturates,
    b how hard long entries are penalised. A shared rare word outranks a
    shared common one through the idf. Pure, so the ranking is easy to test."""
    q = tokenize(query)
    if not q or not entries:
        return []
    docs = [e["tokens"] for e in entries]
    n = len(docs)
    avgdl = (sum(len(d) for d in docs) / n) or 1.0
    df: dict[str, int] = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    k1, b = 1.5, 0.75
    scored = []
    for e, d in zip(entries, docs):
        dl = len(d)
        freq: dict[str, int] = {}
        for t in d:
            freq[t] = freq.get(t, 0) + 1
        score = 0.0
        for t in q:
            f = freq.get(t, 0)
            if not f:
                continue
            idf = math.log(1 + (n - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
            score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        if score > 0:
            scored.append((score, e))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [{"when": e["when"], "score": round(score, 4),
             "snippet": _snippet(e["text"], q)} for score, e in scored[:k]]


def search(query: str, directory, k: int = 3) -> list[dict]:
    """Load the journal and rank it. The one call the voice line makes."""
    return rank(query, load_entries(directory), k)


def spoken(query: str, directory) -> str:
    """The spoken answer to a recall question.

    Never empty: a question answered with "I don't have that" is a real
    answer, and the voice line always has something to say back."""
    hits = search(query, directory, k=1)
    if not hits:
        return "I don't have anything in my journal about that."
    top = hits[0]
    return f"On {top['when']} I wrote: {top['snippet']}"
