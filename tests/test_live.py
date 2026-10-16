"""live.py offline: the fetch path, with the network faked out.

The live-data path is best effort by design -- a failed fetch yields no
context and the turn is answered normally -- so the interesting behaviour
is the order of the fallbacks, the parsing of each source, and the cache.
Everything here patches `live._get` (the one network call) or the source
functions, so nothing touches a socket.
"""
from pathlib import Path
import sys
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtalk import live  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'ok' if cond else 'FAIL'} {name}"
          + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILURES.append(name)


RSS_XML = """<rss><channel>
<title>Google News</title>
<title>US Edition</title>
<title>Big story number one here</title>
<title>Another longer story headline</title>
<title>tiny</title>
</channel></rss>"""

DDG_HTML = (
    '<a class="result-link" href="https://a.example">'
    "A first pretty long result line</a>\n"
    '<a class="result-link" href="https://b.example">'
    "A second pretty long result line</a>\n"
    "<span>something much too short</span>")

DDG_SQ = ('<a class="result-link" href="#">'
          "A quote-styled result line here</a>")

DDG_PATTERN = (r'<a[^>]*class="result-link"[^>]*>(.*?)</a>'
               r"|result-link[^>]*>(.*?)</a>")

WIKI_JSON = (
    '{"query": {"search": ['
    '{"title": "Bitcoin", "snippet": "<b>Bitcoin</b> is a currency"},'
    '{"title": "Ethereum", "snippet": "Ether &amp; friends"}'
    "]}}")


def triggers():
    print("\n--- wants_live: only time-sensitive questions ---")
    for q in ("what's the latest news", "what is the price of bitcoin",
              "who is the current president", "the 2026 election",
              "what's happening today", "weather in Paris"):
        check(f"live: {q!r}", live.wants_live(q))
    for q in ("explain recursion", "write a haiku", "", "hello there"):
        check(f"not live: {q!r}", not live.wants_live(q))


def cleaning():
    print("\n--- _clean: tags out, entities in ---")
    check("strips tags and unescapes entities",
          live._clean("<b>Tom &amp; Jerry</b>") == "Tom & Jerry")
    check("smart quotes folded to ascii",
          live._clean("“x” — ‘y’") == '"x" - \'y\'')
    check("stray replacement char becomes apostrophe",
          live._clean("a�b") == "a'b")


def news():
    print("\n--- _news: feed name skipped, short titles dropped ---")
    with mock.patch.object(live, "_get", return_value=RSS_XML):
        out = live._news()
    check("stories present", "Big story number one here" in out, out)
    check("second story present",
          "Another longer story headline" in out, out)
    check("feed's own title skipped", "Google News" not in out, out)
    check("too-short title dropped", "tiny" not in out, out)
    check("header present", out.startswith("Top stories right now:"), out)
    with mock.patch.object(live, "_get", return_value="<rss></rss>"):
        check("no stories -> empty", live._news() == "")


def search():
    print("\n--- _search: parses hits, filters by length ---")
    # A single-group pattern avoids the tuple issue in the two-group
    # pattern fetch uses; _search itself is generic on `pattern`.
    with mock.patch.object(live, "_get", return_value=DDG_HTML):
        out = live._search("http://x", r'class="result-link"[^>]*>(.*?)</a>')
    check("both hits present",
          "A first pretty long result line" in out
          and "A second pretty long result line" in out, out)
    check("header present", out.startswith("Search results:"), out)
    with mock.patch.object(live, "_get", return_value="<p>nothing</p>"):
        check("no hits -> empty",
              live._search("http://x", r"<h1>(.*?)</h1>") == "")
    # The pattern fetch actually uses has two groups, so findall yields a
    # tuple per match. This used to hand the tuple to _clean and lose every
    # hit; the flattening in _search is what this pins.
    with mock.patch.object(live, "_get", return_value=DDG_HTML):
        out = live._search("http://x", DDG_PATTERN)
    check("the real two-group pattern keeps its hits",
          "A first pretty long result line" in out, out)
    with mock.patch.object(live, "_get", return_value=DDG_SQ):
        out = live._search("http://x", DDG_PATTERN)
    check("the second alternation group is used when the first misses",
          "A quote-styled result line here" in out, out)


def wikipedia():
    print("\n--- _wikipedia: JSON rows rendered ---")
    with mock.patch.object(live, "_get", return_value=WIKI_JSON):
        out = live._wikipedia("bitcoin")
    check("title rendered", "Bitcoin:" in out, out)
    check("snippet cleaned", "Ether & friends" in out, out)
    check("header present", out.startswith("Encyclopedia results:"), out)
    with mock.patch.object(live, "_get", return_value='{"query": {}}'):
        check("no rows -> empty", live._wikipedia("x") == "")


def network():
    print("\n--- _get: the one real network call ---")
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return "<b>caf\u00e9</b>".encode("utf-8")

    with mock.patch.object(live.urllib.request, "urlopen",
                           return_value=FakeResponse()) as opened:
        out = live._get("http://example.test/feed")
    check("_get reads and decodes the body", out == "<b>caf\u00e9</b>", out)
    check("_get passes a timeout",
          opened.call_args.kwargs.get("timeout") == live.TIMEOUT)
    req = opened.call_args.args[0]
    check("_get sends our user agent",
          req.get_header("User-agent") == live.UA, req.headers)


def fetch_branches():
    print("\n--- fetch: source order and the cache ---")
    live._cache.clear()
    check("empty utterance -> empty", live.fetch("") == "")

    with mock.patch.object(live, "_news", return_value="") as n, \
         mock.patch.object(live, "_wikipedia", return_value="WIKI FACTS") as w, \
         mock.patch.object(live, "_search", return_value="") as s:
        out = live.fetch("what's the latest news today")
    check("news tried first", n.called)
    check("wikipedia fallback after news", w.called)
    check("wikipedia chosen, labeled and stamped",
          "from Wikipedia" in out and "WIKI FACTS" in out
          and "LIVE DATA FETCHED JUST NOW" in out, out[:120])
    check("duckduckgo not reached", not s.called)

    live._cache.clear()
    with mock.patch.object(live, "_news", return_value=""), \
         mock.patch.object(live, "_wikipedia", return_value=""), \
         mock.patch.object(live, "_search", return_value="DDG FACTS"):
        out = live.fetch("current stock price of acme")
    check("duckduckgo as last resort",
          "from DuckDuckGo" in out and "DDG FACTS" in out, out[:120])

    live._cache.clear()
    with mock.patch.object(live, "_news", return_value=""), \
         mock.patch.object(live, "_wikipedia", return_value=""), \
         mock.patch.object(live, "_search", return_value=""):
        check("all sources empty -> empty", live.fetch("what's today") == "")

    live._cache.clear()
    with mock.patch.object(live, "_get", return_value=DDG_HTML):
        out = live.fetch("current share price of acme")
    check("the duckduckgo path works end to end",
          "from DuckDuckGo" in out and "first pretty long result" in out,
          out[:120])

    live._cache.clear()
    with mock.patch.object(live, "_news", return_value=""), \
         mock.patch.object(live, "_wikipedia", return_value=""), \
         mock.patch.object(live, "_search",
                           side_effect=RuntimeError("boom")):
        check("a source raising is swallowed",
              live.fetch("current price") == "")

    live._cache.clear()
    with mock.patch.object(live, "_news",
                           side_effect=RuntimeError("feed down")), \
         mock.patch.object(live, "_wikipedia", return_value="WIKI"), \
         mock.patch.object(live, "_search", return_value=""):
        out = live.fetch("the latest news today")
    check("a news feed that raises falls through to wikipedia",
          "WIKI" in out, out[:80])


def fetch_cache():
    print("\n--- fetch: a warm cache skips the network ---")
    live._cache.clear()
    with mock.patch.object(live, "_wikipedia", return_value="CACHED BRIEFING"):
        first = live.fetch("who is the current president")
    with mock.patch.object(live, "_wikipedia",
                           side_effect=AssertionError("network touched")) as w:
        second = live.fetch("who is the current president")
    check("second call served from cache",
          second == first and "CACHED BRIEFING" in second, second[:60])
    check("network not touched on a cache hit", not w.called)

    live._cache.clear()
    with mock.patch.object(live, "_news", return_value=""), \
         mock.patch.object(live, "_wikipedia", return_value="OLD"), \
         mock.patch.object(live.time, "time", return_value=10_000.0):
        live.fetch("who is the current president")
        # advance beyond the TTL: the entry must be refetched
        with mock.patch.object(live, "_wikipedia", return_value="NEW") as w:
            live._cache["who is the current president"] = (1.0, "OLD")
            out = live.fetch("who is the current president")
        check("stale entry refetched", "NEW" in out and w.called, out[:60])


def fetch_cache_bound():
    print("\n--- the cache is bounded ---")
    live._cache.clear()
    live._remember("keep", "value")
    check("a new key is stored", live._cache["keep"][1] == "value")
    for i in range(live.CACHE_MAX):
        live._remember("k%d" % i, "x")
    check("the cache never exceeds its cap",
          len(live._cache) == live.CACHE_MAX, len(live._cache))
    check("the oldest entry was evicted", "keep" not in live._cache)
    live._cache.clear()


def fetch_truncation():
    print("\n--- fetch: the briefing is bounded ---")
    live._cache.clear()
    with mock.patch.object(live, "_news", return_value="A" * 5000), \
         mock.patch.object(live, "_wikipedia", return_value="x"):
        out = live.fetch("the latest news")
    check("truncated to MAX_CHARS", len(out) == live.MAX_CHARS, len(out))


print("=" * 66)
print("LIVE DATA PATH (offline, network faked)")
print("=" * 66)
triggers()
cleaning()
news()
search()
wikipedia()
network()
fetch_branches()
fetch_cache()
fetch_cache_bound()
fetch_truncation()

print("\n" + "=" * 66)
print("LIVE OK" if not FAILURES else f"LIVE FAILURES: {FAILURES}")
sys.exit(1 if FAILURES else 0)
