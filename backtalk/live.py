"""live.py - real-time facts, fetched before the model is allowed to answer.

A reasoning model asked "what happened today" will sometimes answer from
memory, and a memory answer about the present is just a confident
invention. The fix is to not offer it the choice: when a question can only
be answered from the live web, the fetch happens here, in code, and the
result is handed to the model as context. A fetch costs one or two
seconds, so it is far quicker than letting the model discover a search
tool, deliberate, and spawn subagents.

Everything here is best effort. A failed fetch yields no context and the
question is answered normally, so a flaky network never breaks a turn.
"""
from __future__ import annotations

import html
import json
import re
import time
import urllib.parse
import urllib.request

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Jarvis/1.0"
RSS = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"
WIKI = ("https://en.wikipedia.org/w/api.php?action=query&list=search"
        "&srsearch={q}&format=json&srlimit=4")
DDG = "https://lite.duckduckgo.com/lite/?q={q}"

# A question about anything that moves with time, or about a recent year,
# cannot be answered from training data.
TRIGGERS = re.compile(
    r"\b("
    r"today|tonight|now|current|currently|latest|recent|recently|"
    r"right now|up to date|so far|this (?:week|month|year|morning)|"
    r"news|headlines|what(?:'s| is) happening|"
    r"weather|forecast|temperature|"
    r"score|standings|fixture|result[sc]?\b|"
    r"price[sd]?\b|cost|how much (?:is|does|are)|exchange rate|"
    r"stock|share price|market cap|"
    r"who is (?:the )?(?:current|new)|who(?:'s| is) the (?:current|new)|"
    r"president|prime minister|chancellor|ceo|"
    r"released|release date|out now|"
    r"2\d{3}\b"
    r")\b",
    re.IGNORECASE)

NEWSY = re.compile(
    r"\b(news|headlines?|stories|story|happening|events?|today|"
    r"this (?:week|month|year)|latest|breaking)\b", re.IGNORECASE)

MAX_CHARS = 2200
CACHE_TTL = 300.0
CACHE_MAX = 512
TIMEOUT = 6.0
_cache: dict[str, tuple[float, str]] = {}


def _remember(key: str, value: str) -> None:
    """Store a briefing. Keys are whole utterances, so a long-running line
    would otherwise grow the cache without bound; once full, the oldest
    entry goes (dicts keep insertion order, so the first key is oldest)."""
    if len(_cache) >= CACHE_MAX:
        _cache.pop(next(iter(_cache)), None)
    _cache[key] = (time.time(), value)


def wants_live(utterance: str) -> bool:
    return bool(TRIGGERS.search(utterance or ""))


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.read().decode("utf-8", "replace")


def _clean(text: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    # Feeds carry smart quotes and dashes; a stray replacement character
    # would be read out loud as noise.
    for bad, good in (("“", '"'), ("”", '"'), ("‘", "'"),
                      ("’", "'"), ("—", "-"), ("–", "-"),
                      (" ", " "), ("�", "'")):
        text = text.replace(bad, good)
    return text.strip()


def _news() -> str:
    raw = _get(RSS)
    titles = [_clean(t) for t in re.findall(r"<title>(.*?)</title>", raw)]
    # The first two titles are the feed's own name, not stories.
    stories = [t for t in titles[2:] if len(t) > 12][:5]
    if not stories:
        return ""
    return "Top stories right now:\n" + "\n".join(
        f"- {t}" for t in stories)


def _search(url: str, pattern: str) -> str:
    raw = _get(url)
    hits = []
    for h in re.findall(pattern, raw, re.DOTALL):
        # A pattern with an alternation of groups returns a tuple per match,
        # one entry per group and empty for the ones that did not fire. Take
        # the group that actually matched rather than handing the tuple to
        # _clean (which raises on a non-string and silently loses every hit).
        if isinstance(h, tuple):
            h = next((g for g in h if g), "")
        hits.append(_clean(h))
    hits = [h for h in hits if 8 < len(h) < 400]
    if not hits:
        return ""
    body = "\n".join(f"- {h}" for h in hits[:6])
    return f"Search results:\n{body}"


def _wikipedia(query: str) -> str:
    raw = _get(WIKI.format(q=urllib.parse.quote(query[:180])))
    data = json.loads(raw)
    rows = (data.get("query") or {}).get("search") or []
    if not rows:
        return ""
    body = "\n".join(f"- {r['title']}: {_clean(r.get('snippet',''))}"
                     for r in rows)
    return "Encyclopedia results:\n" + body


def fetch(utterance: str) -> str:
    """A short briefing of live facts, or '' if the fetch failed."""
    text = (utterance or "").strip()
    if not text:
        return ""
    key = re.sub(r"\W+", " ", text.lower()).strip()[:120]
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]

    terms = " ".join(re.findall(
        r"[A-Za-z0-9][A-Za-z0-9' \-]{2,}", text)[:14])

    briefing, source = "", ""
    if NEWSY.search(text):
        try:
            briefing = _news()
            source = "Google News"
        except Exception:
            briefing = ""
    if not briefing:
        try:
            briefing = _wikipedia(terms)
            source = "Wikipedia"
        except Exception:
            briefing = ""
    if not briefing:
        try:
            briefing = _search(DDG.format(q=urllib.parse.quote(terms[:150])),
                               r"<a[^>]*class=\"result-link\"[^>]*>(.*?)</a>"
                               r"|result-link[^>]*>(.*?)</a>")
            source = "DuckDuckGo"
        except Exception:
            briefing = ""

    if not briefing:
        _remember(key, "")
        return ""

    stamp = time.strftime("%A, %d %B %Y at %H:%M")
    out = (f"LIVE DATA FETCHED JUST NOW by the assistant, {stamp}, "
           f"from {source}. This is verified current information, not "
           f"memory. Answer ONLY from it, and never claim these facts from "
           f"your own knowledge. If it does not cover what was asked, say "
           f"plainly that it did not show it. Each line below "
           f"is one separate item. Never merge two of them into a single "
           f"story, and never invent detail that is not in the line "
           f"itself. This answer is spoken aloud, so give at most three of "
           f"them unless he asks for more, and never read the whole list.\n"
           f"{briefing}")
    out = out[:MAX_CHARS]
    _remember(key, out)
    return out
