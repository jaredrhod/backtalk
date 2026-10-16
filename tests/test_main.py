"""main.py offline: the pure helpers, the reply batching, the spoken
permission gate, and the whole `amain()` live loop driven over fakes --
pinned without a mic, a model, or an opencode server.
"""
from pathlib import Path
import asyncio
import io
import os
import queue
import sys
import tempfile
import threading
import time
import types
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtalk import main as btmain  # noqa: E402
from backtalk import voiceprint as vp  # noqa: E402

import numpy as np  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'ok' if cond else 'FAIL'} {name}"
          + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILURES.append(name)


class FakeMouth:
    def __init__(self):
        self.chunks = []
        self.pending = []
        self.said = []

    def say_chunk(self, s, pending=None):
        self.chunks.append(s)
        self.pending.append(list(pending or []))

    def say(self, s):
        self.said.append(s)


class FakeBrain:
    def __init__(self, items):
        self.items = items
        self.got = None
        self.interrupted = False

    async def ask_stream(self, text):
        self.got = text
        for it in self.items:
            if isinstance(it, BaseException):
                raise it
            yield it

    async def interrupt(self):
        self.interrupted = True


def console():
    print("\n--- console_match: exact phrases only ---")
    check("a verb phrase matches",
          btmain.console_match("clear the session") == "clear")
    check("case and punctuation are ignored",
          btmain.console_match("Compact Context!") == "compact")
    check("hyphens fold to spaces",
          btmain.console_match("hands-free mode") == "micopen")
    check("a spoken look matches",
          btmain.console_match("look at my screen") == "look")
    check("look ignores case and punctuation",
          btmain.console_match("What's on my screen?") == "look")
    check("talking ABOUT a screen is not a look",
          btmain.console_match("the screen is too bright") is None)
    check("an effort level matches",
          btmain.console_match("set effort to high") == "effort:high")
    check("a bare effort word matches",
          btmain.console_match("effort low") == "effort:low")
    check("ordinary speech never triggers a verb",
          btmain.console_match("please clear the table") is None)
    check("unknown phrase -> None", btmain.console_match("hello") is None)


def formatting():
    print("\n--- _fmt_tokens / _spoken_usage ---")
    check("small number", btmain._fmt_tokens(999) == "999 tokens")
    check("thousands", btmain._fmt_tokens(1000) == "about 1 thousand tokens",
          btmain._fmt_tokens(1000))
    check("millions", btmain._fmt_tokens(1_500_000)
          == "about 1.5 million tokens", btmain._fmt_tokens(1_500_000))

    sess = {"turns": 2, "out_tokens": 999, "cost": 2.0}
    usage = {"categories": [
        {"name": "System", "tokens": 1000},
        {"name": "Free space", "tokens": 5000},
        {"name": "Autocompact buffer", "tokens": 3000},
        {"name": "Messages", "tokens": 2000},
    ]}
    out = btmain._spoken_usage(sess, usage)
    check("turn count spoken", "2 turns this session" in out, out)
    check("tokens spoken", "999 tokens spoken out" in out, out)
    check("dollars spoken", "roughly 2 dollars" in out, out)
    check("free space excluded from the occupied total",
          "about 3 thousand tokens sitting in the context window" in out, out)

    one = btmain._spoken_usage({"turns": 1, "out_tokens": 500, "cost": 0.0},
                               {"categories": []})
    check("singular turn", "1 turn this session" in one, one)
    check("no cost -> no money clause", "cents" not in one and "dollars" not in one,
          one)

    weird = btmain._spoken_usage({"turns": 1, "out_tokens": 10, "cost": 0.0},
                                 mock.Mock())
    check("a broken usage object does not raise",
          "1 turn this session" in weird, weird)


def typed_input():
    print("\n--- _clean_typed / _join_paste / _typed_reader_pipe ---")
    check("gutter glyph scrubbed",
          btmain._clean_typed("▎ Hello") == "Hello")
    check("nested gutters scrubbed",
          btmain._clean_typed("│> quoted") == "quoted")
    check("plain text untouched", btmain._clean_typed("  hi  ") == "hi")
    check("paste joined to one line",
          btmain._join_paste("▎ one\n▎ two") == "one two")

    r, w = os.pipe()
    q: queue.Queue = queue.Queue()
    os.write(w, b"hello world\n\x1b[200~pasted\nlines\x1b[201~last line\n")
    os.close(w)
    t = threading.Thread(target=btmain._typed_reader_pipe, args=(q, r),
                         daemon=True)
    t.start()
    t.join(timeout=2)
    os.close(r)
    got = []
    while not q.empty():
        got.append(q.get_nowait())
    check("plain line read", "hello world" in got, got)
    check("paste collapsed into one message", "pasted lines" in got, got)
    check("line after the paste still read", "last line" in got, got)

    # a paste marker with no closing marker yet: hold it, then EOF.
    r2, w2 = os.pipe()
    q2: queue.Queue = queue.Queue()
    os.write(w2, b"\x1b[200~unfinished")
    os.close(w2)
    btmain._typed_reader_pipe(q2, r2)
    os.close(r2)
    check("an unterminated paste is held, not emitted", q2.empty())

    # a closed descriptor: the reader must return, not raise.
    r3, w3 = os.pipe()
    os.close(r3)
    os.close(w3)
    btmain._typed_reader_pipe(queue.Queue(), r3)
    check("a dead descriptor ends the reader quietly", True)

    # blank lines around and between pastes: skipped, not emitted.
    r5, w5 = os.pipe()
    on = btmain._PASTE_ON.encode()
    off = btmain._PASTE_OFF.encode()
    os.write(w5, b"line1\n\n" + on + off + b"\nlast\n" + on + b"body" + off
             + b"\n\n")
    os.close(w5)
    q5: queue.Queue = queue.Queue()
    btmain._typed_reader_pipe(q5, r5)
    os.close(r5)
    mixed = []
    while not q5.empty():
        mixed.append(q5.get_nowait())
    check("blank lines are skipped around pastes",
          mixed == ["line1", "last", "body"], mixed)

    import builtins
    q3: queue.Queue = queue.Queue()
    with mock.patch.object(builtins, "input",
                           side_effect=["  hello  ", "", "│> quoted", EOFError]):
        btmain._typed_reader_simple(q3)
    simple = []
    while not q3.empty():
        simple.append(q3.get_nowait())
    check("the simple reader scrubs and queues",
          simple == ["hello", "quoted"], simple)

    r4, w4 = os.pipe()
    os.write(w4, b"piped line\n")
    os.close(w4)
    q4: queue.Queue = queue.Queue()
    fake_stdin = mock.Mock()
    fake_stdin.fileno.return_value = r4
    with mock.patch.object(btmain.sys, "stdin", fake_stdin):
        btmain._typed_reader(q4)
    os.close(r4)
    piped = []
    while not q4.empty():
        piped.append(q4.get_nowait())
    check("a non-tty stdin is read as lines", piped == ["piped line"], piped)


def human_forms():
    print("\n--- _human_what / _full_detail ---")
    with mock.patch.object(btmain, "CFG", {"extra_dirs": ["C:/vault"]}):
        check("a vault markdown note is named",
              btmain._human_what("Write", {"file_path": "C:/vault/note.md"},
                                 None)
              == "create or change a note in your vault called note",
              btmain._human_what("Write",
                                 {"file_path": "C:/vault/note.md"}, None))
        check("an edit of a vault note says edit",
              btmain._human_what("Edit", {"file_path": "C:/vault/x.md"}, None)
              == "edit a note in your vault called x")
    check("an ordinary file is named, not pathed",
          btmain._human_what("Write", {"file_path": "C:/x/report.txt"}, None)
          == "create or change a file called report.txt")
    check("a bash command names the program",
          btmain._human_what("Bash", {"command": "git status"}, None)
          == "run a git command in the terminal")
    check("a chained command says so",
          btmain._human_what("Bash", {"command": "a && b"}, None)
          .endswith("with several chained parts"))
    check("a fetch names the host",
          btmain._human_what("WebFetch", {"url": "https://example.com/x"},
                             None) == "read a web page at example.com")
    ctx = mock.Mock(display_name="Todo", description="a task list")
    check("an unknown tool uses its display name",
          btmain._human_what("TodoWrite", {}, ctx) == "use the Todo tool")

    long_cmd = "x" * 130
    detail = btmain._full_detail("Bash", {"command": long_cmd}, None)
    check("a long command warns about hidden characters",
          "more characters" in detail and "Check the log" in detail, detail)
    check("a write shows the last two path segments",
          btmain._full_detail("Write", {"file_path": "a/b/report.txt"}, None)
          == "write the file b/report.txt",
          btmain._full_detail("Write", {"file_path": "a/b/report.txt"}, None))
    check("a fetch shows the url",
          btmain._full_detail("WebFetch", {"url": "https://x.example/p"},
                              None) == "fetch a web page: https://x.example/p")
    check("an unknown tool appends its description",
          btmain._full_detail("TodoWrite", {}, ctx)
          == "use Todo, a task list",
          btmain._full_detail("TodoWrite", {}, ctx))


def config_write():
    print("\n--- _write_config_key: session-only on a bad file ---")
    import backtalk.config as cfgmod
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "backtalk.json"
        path.write_text('{"keep": 1}')
        with mock.patch.object(btmain, "CFG", {"keep": 1}), \
             mock.patch.object(cfgmod, "CONFIG_PATH", path), \
             mock.patch.object(btmain, "log"):
            ok = btmain._write_config_key("new", "value")
            written = path.read_text()
        check("write reports success", ok)
        check("the other key survived",
              "keep" in written and "new" in written, written)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "missing.json"
        with mock.patch.object(btmain, "CFG", {}), \
             mock.patch.object(cfgmod, "CONFIG_PATH", path), \
             mock.patch.object(btmain, "log"):
            ok = btmain._write_config_key("a", 1)
        check("a missing file is created", ok and path.exists())

    bad = mock.Mock()
    bad.read_text.return_value = "not json at all"
    cfg = {"a": 0}
    with mock.patch.object(btmain, "CFG", cfg), \
         mock.patch.object(cfgmod, "CONFIG_PATH", bad), \
         mock.patch.object(btmain, "log") as log:
        ok = btmain._write_config_key("a", 1)
    check("an unparsable file is left alone (session-only)",
          ok is False and not bad.write_text.called)
    check("but the in-memory config still updated", cfg["a"] == 1, cfg)
    check("the failure was logged", log.called)

    ro = mock.Mock()
    ro.read_text.return_value = '{"a": 1}'
    ro.write_text.side_effect = OSError("read-only")
    with mock.patch.object(btmain, "CFG", {}), \
         mock.patch.object(cfgmod, "CONFIG_PATH", ro), \
         mock.patch.object(btmain, "log"):
        ok = btmain._write_config_key("a", 2)
    check("a read-only file fails softly", ok is False)


def speak_first_alone():
    print("\n--- speak_reply: first sentence alone, directions, on_reply ---")
    brain = FakeBrain(["Hello <<wave>> there.", "Second.", "Third.",
                       "Fourth."])
    mouth = FakeMouth()
    spoken: list = []
    with mock.patch.object(btmain, "signals") as sig, \
         mock.patch.object(btmain, "log"):
        asyncio.run(btmain.speak_reply(brain, mouth, "hi",
                                       on_reply=spoken.append))
    check("brain got the raw question", brain.got == "hi")
    check("first sentence ships alone",
          mouth.chunks[:1] == ["Hello there."], mouth.chunks)
    check("the direction rode with the first chunk",
          mouth.pending[0] == ["wave"], mouth.pending)
    check("the rest go in a two-sentence breath",
          mouth.chunks[1] == "Second. Third.", mouth.chunks)
    check("the trailing sentence is flushed",
          mouth.chunks[-1] == "Fourth.", mouth.chunks)
    check("on_reply saw each spoken chunk", spoken == mouth.chunks, spoken)
    check("a full reply leaves the bus alone",
          not sig.set_state.called and not sig.static_stop.called)


def speak_empty_and_errors():
    print("\n--- speak_reply: empty and failing turns park the bus ---")
    mouth = FakeMouth()
    with mock.patch.object(btmain, "signals") as sig, \
         mock.patch.object(btmain, "log"):
        asyncio.run(btmain.speak_reply(FakeBrain([]), mouth, "hi"))
    check("nothing spoken on an empty reply", not mouth.chunks and not mouth.said)
    check("an empty turn parks the bus",
          sig.static_stop.called and sig.caption_clear.called)
    states = [c.args[0] for c in sig.set_state.call_args_list]
    check("bus parked to idle", states == ["idle"], states)

    mouth2 = FakeMouth()
    with mock.patch.object(btmain, "signals") as sig2, \
         mock.patch.object(btmain, "log"):
        asyncio.run(btmain.speak_reply(FakeBrain([RuntimeError("boom")]),
                                       mouth, "hi"))
    check("a hard failure says so out loud",
          mouth.said and "could not reach" in mouth.said[0], mouth.said)
    check("and parks the bus", sig.static_stop.called)

    brain = FakeBrain(["One.", "Two.", RuntimeError("mid-stream")])
    mouth2 = FakeMouth()
    with mock.patch.object(btmain, "signals") as sig, \
         mock.patch.object(btmain, "log"):
        asyncio.run(btmain.speak_reply(brain, mouth2, "hi"))
    check("a sentence already in hand is still spoken",
          mouth2.chunks == ["One.", "Two."], mouth2.chunks)
    check("no error line when something was already said",
          mouth2.said == [], mouth2.said)

    brain = FakeBrain(["One.", asyncio.CancelledError()])
    mouth3 = FakeMouth()
    cancelled = False
    try:
        with mock.patch.object(btmain, "signals"), \
             mock.patch.object(btmain, "log"):
            asyncio.run(btmain.speak_reply(brain, mouth3, "hi"))
    except asyncio.CancelledError:
        cancelled = True
    check("a cancelled turn propagates", cancelled)
    check("and the brain is interrupted to stop the model",
          brain.interrupted)

    brain = FakeBrain(["One.", "Two."])
    mouth4 = FakeMouth()

    def boom(_s):
        raise RuntimeError("journal broke")

    with mock.patch.object(btmain, "signals"), \
         mock.patch.object(btmain, "log"):
        asyncio.run(btmain.speak_reply(brain, mouth4, "hi", on_reply=boom))
    check("a raising on_reply callback is swallowed",
          mouth4.chunks == ["One.", "Two."], mouth4.chunks)

    # A speakable-only sentence (all backticks / whitespace) is dropped.
    brain = FakeBrain(["```", "Real words."])
    mouth5 = FakeMouth()
    with mock.patch.object(btmain, "signals"), \
         mock.patch.object(btmain, "log"):
        asyncio.run(btmain.speak_reply(brain, mouth5, "hi"))
    check("an unspeakable sentence is dropped, the next one speaks",
          mouth5.chunks == ["Real words."], mouth5.chunks)

    # A raise from on_reply on the batched (second) chunk is swallowed.
    brain = FakeBrain(["One.", "Two.", "Three."])
    mouth6 = FakeMouth()
    with mock.patch.object(btmain, "signals"), \
         mock.patch.object(btmain, "log"):
        asyncio.run(btmain.speak_reply(brain, mouth6, "hi", on_reply=boom))
    check("a raising on_reply mid-batch is swallowed",
          mouth6.chunks == ["One.", "Two. Three."], mouth6.chunks)

    class DeafBrain(FakeBrain):
        async def interrupt(self):
            raise RuntimeError("interrupt broke")

    brain = DeafBrain(["One.", asyncio.CancelledError()])
    mouth7 = FakeMouth()
    cancelled = False
    try:
        with mock.patch.object(btmain, "signals"), \
             mock.patch.object(btmain, "log"):
            asyncio.run(btmain.speak_reply(brain, mouth7, "hi"))
    except asyncio.CancelledError:
        cancelled = True
    check("a failing interrupt still lets the cancel through", cancelled)

    # A turn that dies with one sentence still batched, and an on_reply
    # that also raises: the tail is spoken, the callback's error is eaten.
    brain = FakeBrain(["One.", "Two.", RuntimeError("mid-stream")])
    mouth8 = FakeMouth()
    with mock.patch.object(btmain, "signals"), \
         mock.patch.object(btmain, "log"):
        asyncio.run(btmain.speak_reply(brain, mouth8, "hi", on_reply=boom))
    check("a failing turn still speaks its batched tail",
          mouth8.chunks == ["One.", "Two."], mouth8.chunks)

    # A clean turn that ends holding exactly one extra sentence.
    brain = FakeBrain(["One.", "Two."])
    mouth9 = FakeMouth()
    with mock.patch.object(btmain, "signals"), \
         mock.patch.object(btmain, "log"):
        asyncio.run(btmain.speak_reply(brain, mouth9, "hi"))
    check("a clean turn flushes its single trailing sentence",
          mouth9.chunks == ["One.", "Two."], mouth9.chunks)


def permission_gate():
    print("\n--- make_permission_gate: speak, wait, decide ---")
    mouth = FakeMouth()
    mouth.speaking = False

    def reset():
        btmain._PERM["fut"] = None
        btmain._PERM["hinted"] = False
        btmain._AUTOAPPROVE["on"] = False

    def run_gate(tool, tool_input, answers, ctx=None, timeout=8.0):
        async def runner():
            gate = btmain.make_permission_gate(mouth)
            task = asyncio.ensure_future(gate(tool, tool_input, ctx))
            last = None
            for ans in answers:
                for _ in range(4000):
                    fut = btmain._PERM["fut"]
                    if fut is not None and fut is not last and not fut.done():
                        break
                    await asyncio.sleep(0.005)
                last = btmain._PERM["fut"]
                if last is not None and not last.done():
                    last.set_result(ans)
            return await asyncio.wait_for(task, timeout)
        return asyncio.run(runner())

    reset()
    with mock.patch.object(btmain, "signals"), \
         mock.patch.object(btmain, "log"):
        btmain._AUTOAPPROVE["on"] = True
        res = run_gate("Bash", {"command": "ls"}, [])
    check("auto-approve returns allow without asking",
          res.behavior == "allow", res)
    reset()

    with mock.patch.object(btmain, "signals"), \
         mock.patch.object(btmain, "log"):
        res = run_gate("Bash", {"command": "ls"}, ["yes please"])
    check("an affirmative answer allows", res.behavior == "allow", res)
    reset()

    with mock.patch.object(btmain, "signals"), \
         mock.patch.object(btmain, "log"):
        res = run_gate("Bash", {"command": "rm -rf /"}, ["absolutely not"])
    check("a non-affirmative answer denies", res.behavior == "deny", res)
    check("the denial quotes the person",
          "absolutely not" in res.message, res.message)
    reset()

    with mock.patch.object(btmain, "signals"), \
         mock.patch.object(btmain, "log"):
        res = run_gate("Write", {"file_path": "x.md"},
                       [btmain._INTERRUPT_ANSWER])
    check("the interrupt sentinel denies silently",
          res.behavior == "deny" and "interrupted" in res.message.lower(),
          res.message)
    reset()

    with mock.patch.object(btmain, "signals"), \
         mock.patch.object(btmain, "log"):
        res = run_gate("WebFetch", {"url": "https://x"}, ["details", "yes"])
    check("asking for details then yes still allows",
          res.behavior == "allow", res)
    check("details were read back out",
          any("details" in s.lower() for s in mouth.said), mouth.said)
    reset()

    with mock.patch.object(btmain, "signals"), \
         mock.patch.object(btmain, "log"), \
         mock.patch.object(btmain, "PERM_TIMEOUT_S", 0.05):
        res = run_gate("Bash", {"command": "ls"}, [], timeout=5.0)
    check("silence times out into a deny", res.behavior == "deny", res)
    check("the timeout is announced", any("No answer" in s
                                          for s in mouth.said), mouth.said)
    reset()

    with mock.patch.object(btmain, "signals"), \
         mock.patch.object(btmain, "log"):
        run_gate("Bash", {"command": "ls"}, ["yes"])
        res = run_gate("Bash", {"command": "ls"}, ["yes"])
    check("a later ask does not re-announce the escape hatch",
          res.behavior == "allow" and btmain._PERM["hinted"], res)
    reset()

    mouth.speaking = False
    real_wait_for = btmain.asyncio.wait_for
    ticks = {"n": 0}

    async def flaky_wait_for(aw, t):
        if not isinstance(aw, asyncio.Task) and ticks["n"] == 0:
            ticks["n"] = 1
            aw.cancel()
            raise asyncio.TimeoutError
        return await real_wait_for(aw, t)

    with mock.patch.object(btmain, "signals") as sig, \
         mock.patch.object(btmain, "log"), \
         mock.patch.object(btmain.asyncio, "wait_for", flaky_wait_for):
        res = run_gate("Bash", {"command": "ls"}, ["yes"])
    check("a quiet wait tick nudges the state to listening",
          sig.set_state.called and res.behavior == "allow", res)
    reset()

    mouth.speaking = True
    ticks["n"] = 0
    with mock.patch.object(btmain, "signals") as sig2, \
         mock.patch.object(btmain, "log"), \
         mock.patch.object(btmain.asyncio, "wait_for", flaky_wait_for):
        res = run_gate("Bash", {"command": "ls"}, ["yes"])
    check("a wait tick while speaking does not touch the state",
          ("listening",) not in [c.args
                                 for c in sig2.set_state.call_args_list]
          and res.behavior == "allow", res)
    mouth.speaking = False
    reset()


def gate_helpers():
    print("\n--- _deny_pending resolves a live ask ---")
    async def runner():
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        btmain._PERM["fut"] = fut
        btmain._deny_pending()
        return await fut

    got = asyncio.run(runner())
    check("a pending future is resolved to the interrupt sentinel",
          got == btmain._INTERRUPT_ANSWER, got)
    btmain._PERM["fut"] = None
    btmain._deny_pending()  # no future: must not raise
    check("no future -> no-op", True)


def journal_flush_case():
    print("\n--- journal_flush: a hangup never hangs or crashes ---")

    class FakeJournal:
        def __init__(self, active=True, events=True):
            self.active = active
            self.events = events
            self.written = []

        def transcript(self):
            return "**you:** hi"

        def write(self, summary):
            self.written.append(summary)
            return None

    class FakeScratch:
        instance = None

        def __init__(self, *a, **k):
            FakeScratch.instance = self
            self.started = False
            self.stopped = False

        async def start(self):
            self.started = True

        async def stop(self):
            self.stopped = True

        async def ask_stream(self, text):
            yield "x"

    j = FakeJournal(active=False)
    asyncio.run(btmain.journal_flush(j, None))
    check("an inactive journal is skipped", j.written == [])

    j = FakeJournal(active=True, events=False)
    asyncio.run(btmain.journal_flush(j, None))
    check("an empty journal is skipped", j.written == [])

    j = FakeJournal()
    with mock.patch.object(btmain, "CFG", {"journal_summary": False}), \
         mock.patch.object(btmain, "log"):
        asyncio.run(btmain.journal_flush(j, None))
    check("no summary -> an entry with no summary", j.written == [""])

    async def fake_summarize(ask_stream, text):
        return "the summary"

    j = FakeJournal()
    with mock.patch.object(btmain, "CFG", {"journal_summary": True}), \
         mock.patch.object(btmain, "WarmBrain", FakeScratch), \
         mock.patch.object(btmain, "journal_summarize", fake_summarize), \
         mock.patch.object(btmain, "log"):
        asyncio.run(btmain.journal_flush(j, None))
    check("a summary is written when asked", j.written == ["the summary"])
    check("the scratch brain is started and stopped",
          FakeScratch.instance.started and FakeScratch.instance.stopped)

    class StubbornScratch(FakeScratch):
        async def stop(self):
            raise RuntimeError("stop broke")

    j = FakeJournal()
    with mock.patch.object(btmain, "CFG", {"journal_summary": True}), \
         mock.patch.object(btmain, "WarmBrain", StubbornScratch), \
         mock.patch.object(btmain, "journal_summarize", fake_summarize), \
         mock.patch.object(btmain, "log"):
        asyncio.run(btmain.journal_flush(j, None))
    check("a scratch brain that will not stop is swallowed",
          j.written == ["the summary"])

    class WrittenJournal(FakeJournal):
        def write(self, summary):
            self.written.append(summary)
            return type("P", (), {"name": "entry.md"})()

    j = WrittenJournal()
    with mock.patch.object(btmain, "CFG", {"journal_summary": False}), \
         mock.patch.object(btmain, "log") as logged:
        asyncio.run(btmain.journal_flush(j, None))
    check("a written entry is logged by name", logged.called)

    class BoomJournal(FakeJournal):
        def write(self, summary):  # noqa: ARG002
            raise RuntimeError("disk full")

    j = BoomJournal()
    with mock.patch.object(btmain, "CFG", {"journal_summary": False}), \
         mock.patch.object(btmain, "log"):
        asyncio.run(btmain.journal_flush(j, None))
    check("a broken write never crashes the hangup", True)


def single_instance():
    print("\n--- _claim_single_instance: one voice line, out loud ---")
    taken = mock.Mock()
    taken.bind.side_effect = OSError("in use")
    with mock.patch.object(btmain.socket, "socket", return_value=taken):
        btmain._instance_lock = None
        ok = btmain._claim_single_instance()
    check("a held port refuses a second voice line", ok is False)
    check("the loser socket is closed", taken.close.called)

    free = mock.Mock()
    with mock.patch.object(btmain.socket, "socket", return_value=free):
        btmain._instance_lock = None
        ok = btmain._claim_single_instance()
    check("a free port is claimed",
          ok is True and btmain._instance_lock is free)
    check("the winner listens but serves nothing", free.listen.called)
    btmain._instance_lock = None

    with mock.patch.object(btmain, "signals") as sig:
        btmain._take_the_bus()
    check("taking the bus parks it and registers the park",
          sig.register_exit_park.called and sig.park.called)


class _LoopMouth:
    def __init__(self):
        self.said = []
        self.chunks = []
        self.speaking = False
        self.shut = 0
        self.shut_down = False
        self.waited = 0
        self.ducker = mock.Mock()

    def say(self, s):
        self.said.append(s)

    def say_chunk(self, s, pending=None):
        self.chunks.append(s)

    def shut_up(self):
        self.shut += 1

    def wait_done(self, timeout=None):
        self.waited += 1

    def shutdown(self):
        self.shut_down = True


class _LoopEars:
    def __init__(self, fail_first=0):
        self.calls = 0
        self.fail_first = fail_first

    def listen_once(self, gate=None, abort=None, on_partial=None):
        self.calls += 1
        time.sleep(0.01)
        if self.calls <= self.fail_first:
            raise RuntimeError("mic gone")
        return None

    def wait_for_wake(self, phrases, gate=None, abort=None, on_partial=None):
        return None


class _LoopJournal:
    instances = []

    def __init__(self, *a, **k):
        self.active = False
        self.events = []
        _LoopJournal.instances.append(self)

    def say_user(self, t):
        pass

    def say_agent(self, t):
        pass

    def say_activity(self, t):
        self.events.append(("activity", t))


class _LoopBrain:
    instances = []

    def __init__(self, *a, **k):
        self.model = k.get("model") or "fake-model"
        self.session = {"turns": 0, "out_tokens": 0,
                        "in_tokens": 0, "cost": 0.0}
        self.started = False
        self.stopped = False
        self.commands = []
        self.got = []
        _LoopBrain.instances.append(self)

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    async def reset_turn(self):
        pass

    async def interrupt(self):
        pass

    async def set_permission_mode(self, m):
        pass

    async def command(self, c):
        self.commands.append(c)
        return f"ok {c}"

    async def context_usage(self):
        return {"categories": []}

    async def ask_stream(self, text):
        if text.startswith("Warmup ping"):
            return
        self.got.append(text)
        yield "Answer one."
        yield "Answer two."


class _LookBrain(_LoopBrain):
    """Accepts the optional screenshot a spoken look sends the brain."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.images = []

    async def ask_stream(self, text, image=None):
        if text.startswith("Warmup ping"):
            return
        self.got.append(text)
        self.images.append(image)
        yield "That is your editor."


class _DeadBrain(_LoopBrain):
    async def start(self):
        raise RuntimeError("no brain, no provider")


class _SlowBrain(_LoopBrain):
    async def ask_stream(self, text):
        if text.startswith("Warmup ping"):
            return
        await asyncio.sleep(0.4)
        yield "Slow answer."


class _SlowRecordBrain(_LoopBrain):
    """Records the turn, then dawdles, so an interrupt catches it mid-reply."""

    async def ask_stream(self, text):
        if text.startswith("Warmup ping"):
            return
        self.got.append(text)
        await asyncio.sleep(0.4)
        yield "Slow answer."


class _PermBrain(_LoopBrain):
    """Mimics the SDK: a turn poses a spoken ask and resumes when the
    NEXT utterance resolves it."""

    async def ask_stream(self, text):
        if text.startswith("Warmup ping"):
            return
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        btmain._PERM["fut"] = fut
        btmain._PERM["asked_at"] = time.monotonic()
        try:
            answer = await asyncio.wait_for(fut, timeout=5)
        finally:
            btmain._PERM["fut"] = None
        yield f"answered {answer}"


class _EdgeBrain(_LoopBrain):
    async def command(self, c):
        self.commands.append(c)
        if c == "/compact":
            raise RuntimeError("compact blew up")
        if c == "/clear":
            return "Error: could not clear the context"
        return f"ok {c}"

    async def set_permission_mode(self, m):
        raise RuntimeError("cannot flip into ask")


class _WarmTokenBrain(_LoopBrain):
    """Yields a token for the hidden warmup ping instead of nothing."""

    async def ask_stream(self, text):
        if text.startswith("Warmup ping"):
            yield "ready"
            return
        self.got.append(text)
        yield "Answer one."


class _WakeEars(_LoopEars):
    """One wake phrase per run, then silence; optional command turn."""

    def __init__(self, wake, then="what is up"):
        super().__init__()
        self._wake = wake
        self._then = then
        self._n = 0

    def wait_for_wake(self, phrases, gate=None, abort=None, on_partial=None,
                      owner_check=None):
        self._n += 1
        return self._wake if self._n == 1 else None

    def listen_once(self, gate=None, abort=None, on_partial=None):
        self.calls += 1
        return self._then if self._n == 1 else None


class _EnrollEars(_LoopEars):
    """Returns a synthetic clip from record_clip for --enroll."""

    def __init__(self, clip=None):
        super().__init__()
        self._clip = clip

    def record_clip(self, seconds=3.0):
        if self._clip is not None:
            return self._clip
        rng = np.random.default_rng(0)
        return (rng.standard_normal(16000) * 3000).astype(np.int16)


class _GateEars(_WakeEars):
    """A wake listener that records the owner check it was handed."""

    def __init__(self, wake):
        super().__init__(wake)
        self.owner_checks = []

    def wait_for_wake(self, phrases, gate=None, abort=None, on_partial=None,
                      owner_check=None):
        self.owner_checks.append(owner_check)
        return super().wait_for_wake(phrases, gate=gate, abort=abort,
                                     on_partial=on_partial,
                                     owner_check=owner_check)


class _QuitEars(_LoopEars):
    """Returns a quit phrase once, then silence."""

    def __init__(self, text="goodbye seyon"):
        super().__init__()
        self._text = text
        self._said = False

    def listen_once(self, gate=None, abort=None, on_partial=None):
        self.calls += 1
        if not self._said:
            self._said = True
            return self._text
        return None


class _BadDeviceEars(_LoopEars):
    """Every capture fails with a device-level error."""

    def listen_once(self, gate=None, abort=None, on_partial=None):
        self.calls += 1
        raise RuntimeError("invalid device")


class _FakePTT:
    def __init__(self):
        self.presses = 0

    def wait_press(self):
        self.presses += 1
        if self.presses > 1:
            time.sleep(0.25)
        return None

    def is_held(self):
        return False


class _TimedPress(_FakePTT):
    """A key that reports a press only after a delay, so it can be made
    to land while other work (a reply, a capture) is still in flight."""

    def __init__(self, delay):
        super().__init__()
        self._delay = delay

    def wait_press(self):
        self.presses += 1
        time.sleep(self._delay)
        return None


class _StaleCaptureEars(_LoopEars):
    """A capture slow enough to resolve only after a live mode switch."""

    def listen_once(self, gate=None, abort=None, on_partial=None):
        self.calls += 1
        time.sleep(0.4)
        return None


class _InstantEars(_LoopEars):
    """A capture that resolves the instant it is asked."""

    def listen_once(self, gate=None, abort=None, on_partial=None):
        self.calls += 1
        return None


class _YieldingBrain(_LoopBrain):
    """A brain whose turn reset yields, opening a window for a press and
    a capture to land mid-command (i.e. under the mode being switched)."""

    async def reset_turn(self):
        await asyncio.sleep(0.1)


class _KbBrain(_LoopBrain):
    """A command that raises KeyboardInterrupt out of the turn loop."""

    async def command(self, c):
        self.commands.append(c)
        raise KeyboardInterrupt


def _noop_record(is_held, on_release=None, on_partial=None):
    if on_release:
        on_release()
    return ""


class _FakeScreen:
    """Stands in for backtalk.screen: a canned capture and a stub watch."""

    def __init__(self, png=b"PNG", raises=None,
                 changes=("code.exe \u2014 main.py",),
                 window=("code.exe", "main.py"), can_see=True):
        self._png = png
        self._raises = raises
        self.changes = list(changes)
        self.window = window
        self.can_see = can_see
        self.watched = []
        self.window_reads = 0

    def available(self):
        return self.can_see

    def capture(self):
        if self._raises is not None:
            raise self._raises
        return self._png

    def active_window(self):
        self.window_reads += 1
        return self.window

    async def watch(self, interval, on_change):
        self.watched.append(interval)
        for change in self.changes:
            on_change(change)
        while True:
            await asyncio.sleep(interval)


class _QuietEars(_LoopEars):
    """A mic that stays silent through a whole check-in window."""

    def listen_once(self, gate=None, abort=None, on_partial=None):
        self.calls += 1
        time.sleep(0.5)
        return None


class _BusyMouth(_LoopMouth):
    """A mouth stuck 'speaking', to hold a check-in back."""

    def __init__(self):
        super().__init__()
        self.speaking = True


class _FollowEars(_LoopEars):
    """Never hears a wake phrase; returns a command on the first DIRECT
    capture (the follow-up window after a reply), then falls silent."""

    def __init__(self, text="goodbye seyon"):
        super().__init__()
        self.text = text
        self.direct = 0

    def wait_for_wake(self, phrases, gate=None, abort=None, on_partial=None,
                      owner_check=None):
        time.sleep(0.05)
        return None

    def listen_once(self, gate=None, abort=None, on_partial=None,
                    owner_check=None):
        self.calls += 1
        self.direct += 1
        time.sleep(0.05)
        return self.text if self.direct == 1 else None


_LAST_SIGNALS = None


def _run_amain(feed, cfg_extra=None, argv=(), brain_cls=_LoopBrain,
               ears=None, mouth=None, ptt=None, record=None,
               feed_delay=0.0, explain=None, screen=None):
    global _LAST_SIGNALS
    cfg = {
        "agent_dir": "C:/agent", "model": "fast-model",
        "deep_model": "deep-model", "greeting": "Hello.",
        "signoff": "Bye.", "ptt_key": "home", "mic_mode": "open",
        "permission_mode": "ask", "effort": "", "journal_dir": None,
        "resume_last_session": False, "wake_word": False,
        "wake_word_phrases": [], "barge_in": False,
        "journal_summary": False,
    }
    cfg.update(cfg_extra or {})
    mouth = mouth or _LoopMouth()
    ears = ears or _LoopEars()

    release = threading.Event()

    def feeder(q):
        if feed_delay:
            time.sleep(feed_delay)
        for line in feed:
            if isinstance(line, tuple):
                time.sleep(line[0])
                line = line[1]
            q.put(line)
        # The loop often returns while its last `typed_q.get` is still
        # parked in a pool thread. That thread is non-daemon, so leaving
        # it blocked hangs interpreter shutdown. Once the run is over,
        # hand the abandoned getters a few empty lines to release them.
        release.wait(30)
        for _ in range(8):
            q.put("")

    ptt_patch = (mock.patch.object(btmain, "PTTListener",
                                   side_effect=RuntimeError("no key hook"))
                 if ptt is None
                 else mock.patch.object(btmain, "PTTListener",
                                        lambda key: ptt))
    record_patch = (mock.patch.object(btmain, "record_held")
                    if record is None else
                    mock.patch.object(btmain, "record_held", record))

    _LoopBrain.instances.clear()
    btmain._PERM.update(fut=None, hinted=False, asked_at=0.0)
    btmain._CONFIRM.update(verb=None, at=0.0)
    btmain._AUTOAPPROVE["on"] = False
    btmain._MIC.update(mode="ptt", gen=0, btn=False)

    async def runner():
        try:
            async with asyncio.timeout(20):
                await btmain.amain()
        except SystemExit as e:
            result["exit"] = e.code
        except asyncio.TimeoutError:
            result["timeout"] = True

    result = {"exit": None, "timeout": False}
    screen = screen or _FakeScreen()
    sig = mock.Mock()
    _LAST_SIGNALS = sig
    with mock.patch.object(btmain, "CFG", cfg), \
         mock.patch.object(btmain, "Mouth", lambda: mouth), \
         mock.patch.object(btmain, "Ears", lambda: ears), \
         mock.patch.object(btmain, "WarmBrain", brain_cls), \
         mock.patch.object(btmain, "Journal", _LoopJournal), \
         mock.patch.object(btmain, "make_permission_gate",
                           lambda m: (lambda *a, **k: None)), \
         mock.patch.object(btmain, "_typed_reader", feeder), \
         mock.patch.object(btmain, "set_turn_language"), \
         mock.patch.object(btmain, "warm_ears"), \
         mock.patch.object(btmain, "explain_audio_failure",
                           explain or (lambda e: False)), \
         mock.patch.object(btmain, "_write_config_key",
                           lambda k, v: True), \
         mock.patch.object(btmain, "QUIT_PHRASES", {"goodbye seyon"}), \
         mock.patch.object(btmain, "signals", sig), \
         mock.patch.object(btmain, "screen", screen), \
         mock.patch.object(btmain, "log"), \
         ptt_patch, record_patch, \
         mock.patch.object(btmain.sys, "argv", ["prog", *argv]):
        asyncio.run(runner())
    release.set()
    time.sleep(0.1)
    return mouth, ears, result


def amain_loop():
    print("\n--- amain: the live loop over fakes ---")
    feed = [
        "what is the capital of France",
        "clear the session",
        "compact the session",
        "switch to the deep model",
        "back to the fast model",
        "set effort to low",
        "usage report",
        "go hands free",
        "push to talk mode",
        "stop asking for permission",
        "confirm",
        "start asking again",
        "goodbye seyon",
    ]
    mouth, _, res = _run_amain(feed, argv=["--model"])
    brain = _LoopBrain.instances[-1]
    check("the loop ran to a clean hangup",
          res["exit"] is None and not res["timeout"], res)
    check("the greeting is spoken", "Hello." in mouth.said, mouth.said[:2])
    check("a typed question is answered aloud",
          "Answer one." in mouth.chunks, mouth.chunks)
    check("the signoff is spoken on quit", "Bye." in mouth.said)
    check("the mouth is shut down on the way out", mouth.shut_down)
    check("the brain is stopped on the way out", brain.stopped)
    check("console clear reached the brain",
          "/clear" in brain.commands, brain.commands)
    check("console compact reached the brain", "/compact" in brain.commands)
    check("console deep selected the deep model",
          any("deep-model" in c for c in brain.commands), brain.commands)
    check("console fast selected the fast model",
          any("fast-model" in c for c in brain.commands), brain.commands)
    check("console effort set a level",
          any("effort low" in c for c in brain.commands), brain.commands)
    check("usage was spoken",
          any("this session" in s for s in mouth.said), mouth.said)
    check("the mic was switched to push-to-talk",
          btmain._MIC["mode"] == "ptt", btmain._MIC["mode"])

    mouth2, _, res2 = _run_amain([], brain_cls=_DeadBrain)
    check("a brain that will not connect exits the line",
          res2["exit"] == 1 and not res2["timeout"], res2)
    check("and says so out loud",
          any("brain" in s.lower() for s in mouth2.said), mouth2.said)

    import backtalk.brain as btbrain
    missing = str(Path(tempfile.gettempdir()) / "backtalk-no-such-session")
    with mock.patch.object(btbrain, "SESSION_FILE", missing):
        mouth3, _, res3 = _run_amain(
            ["goodbye seyon"],
            cfg_extra={"wake_word": True, "wake_word_phrases": [],
                       "resume_last_session": True},
            argv=["--wake-word"])
    check("wake word with no phrases falls back quietly",
          res3["exit"] is None and not res3["timeout"], res3)
    check("the fallback still hangs up on the phrase",
          "Bye." in mouth3.said, mouth3.said)

    print("\n--- amain: console edges, interrupts, the mic, and the key ---")
    edge_feed = [
        "clear the session",
        "compact the session",
        "stop asking for permission",
        "later on",
        "stop asking for permission",
        "confirm",
        "push to talk mode",
        "go hands free",
        "start asking again",
        "goodbye seyon",
    ]
    mouth4, _, res4 = _run_amain(
        edge_feed, argv=["--model", "alt-model"],
        cfg_extra={"permission_mode": "bypassPermissions",
                   "mic_mode": "ptt", "effort": "bogus"},
        brain_cls=_EdgeBrain)
    check("the edge console run hangs up cleanly",
          res4["exit"] is None and not res4["timeout"], res4)
    check("a console verb that errors is reported, not fatal",
          any("That command hit an error" in s for s in mouth4.said),
          mouth4.said)
    check("a slash-command error is read back verbatim",
          any(s.startswith("Error:") for s in mouth4.said), mouth4.said)
    check("an unanswered confirm stays put",
          any("Staying as we are" in s for s in mouth4.said), mouth4.said)
    check("with no key hook, the boot ptt falls back to the open mic",
          any("Push to talk. Hold the home key" in s for s in mouth4.said),
          mouth4.said)
    check("hands-free can be turned on live",
          btmain._MIC["mode"] == "open", btmain._MIC["mode"])
    check("a failed live flip into 'ask' is admitted",
          any("couldn't switch over" in s for s in mouth4.said), mouth4.said)

    mouth5, _, res5 = _run_amain(["ask one", "ask two", "goodbye seyon"],
                                 brain_cls=_SlowBrain)
    check("a mid-reply interruption hangs up cleanly",
          res5["exit"] is None and not res5["timeout"], res5)
    check("the running reply was cut at least once", mouth5.shut >= 2,
          mouth5.shut)

    mouth6, _, res6 = _run_amain(
        ["do thing one", "yes please", "do thing two", "goodbye seyon"],
        brain_cls=_PermBrain)
    check("the permission run hangs up cleanly",
          res5["exit"] is None and not res5["timeout"], res5)
    check("a spoken answer resumes the paused turn",
          any("answered yes please" in c for c in mouth6.chunks),
          mouth5.chunks)

    def make_rec():
        state = {"n": 0}

        def rec(is_held, on_release=None, on_partial=None):
            state["n"] += 1
            if on_release:
                on_release()
            return "hello via ptt" if state["n"] == 1 else ""
        return rec

    mouth6b, _, res6b = _run_amain(
        ["goodbye seyon"], cfg_extra={"mic_mode": "ptt"},
        ptt=_FakePTT(), record=make_rec(), feed_delay=0.5)
    check("the push-to-talk run hangs up cleanly",
          res6b["exit"] is None and not res6b["timeout"], res6b)
    check("a held key sends the recorded words to the brain",
          any("hello via ptt" in b.got for b in _LoopBrain.instances),
          [b.got for b in _LoopBrain.instances])

    mouth6c, _, res6c = _run_amain(
        ["push to talk mode", "goodbye seyon"],
        cfg_extra={"mic_mode": "ptt"}, ptt=_FakePTT(),
        record=_noop_record)
    check("push-to-talk already on says so",
          any("Already on push to talk" in s for s in mouth6c.said),
          mouth6c.said)

    mouth7, _, res7 = _run_amain(
        ["goodbye seyon"], cfg_extra={"mic_mode": "open"},
        ears=_LoopEars(), feed_delay=0.3)
    check("the open-mic run hangs up cleanly (silent captures)",
          res7["exit"] is None and not res7["timeout"], res7)
    check("the open mic was actually polled",
          mouth7 is not None, mouth7)

    mouth8, _, res8 = _run_amain(
        ["goodbye seyon"], cfg_extra={"mic_mode": "open"},
        ears=_LoopEars(fail_first=3), feed_delay=0.5)
    check("the open-mic failure run hangs up cleanly",
          res8["exit"] is None and not res8["timeout"], res8)
    check("three mic failures fall back to push-to-talk",
          any("open microphone keeps failing" in s for s in mouth8.said),
          mouth8.said)

    print("\n--- amain: startup config and the capture edges ---")
    mouth9, _, res9 = _run_amain(["goodbye seyon"],
                                 brain_cls=_WarmTokenBrain)
    check("a warmup reply is consumed silently",
          res9["exit"] is None and not res9["timeout"], res9)

    mouth10, _, res10 = _run_amain(["goodbye seyon"],
                                   cfg_extra={"effort": "high"})
    check("a configured effort is applied at boot",
          res9["exit"] is None and not res9["timeout"], res9)
    check("the boot effort reached the brain",
          any("effort high" in c for c in _LoopBrain.instances[-1].commands),
          _LoopBrain.instances[-1].commands)

    with tempfile.TemporaryDirectory() as td:
        sess = Path(td) / "session.json"
        sess.write_text("saved-session-id\n", encoding="utf-8")
        import backtalk.brain as btbrain
        with mock.patch.object(btbrain, "SESSION_FILE", sess):
            _, _, res10 = _run_amain(
                ["goodbye seyon"],
                cfg_extra={"resume_last_session": True})
    check("a saved session file is resumed",
          res9["exit"] is None and not res9["timeout"], res9)

    _, _, res_w1 = _run_amain(
        ["goodbye seyon"],
        cfg_extra={"mic_mode": "open", "wake_word": True,
                   "wake_word_phrases": ["vijay seyon"]},
        ears=_WakeEars(wake="vijay seyon hello"), feed_delay=0.3)
    check("a wake phrase that carried the command is used",
          res_w1["exit"] is None and not res_w1["timeout"], res_w1)

    _, _, res_w2 = _run_amain(
        ["goodbye seyon"],
        cfg_extra={"mic_mode": "open", "wake_word": True,
                   "wake_word_phrases": ["vijay seyon"]},
        ears=_WakeEars(wake="vijay seyon"), feed_delay=0.3)
    check("a bare wake phrase then listens for the command",
          res_w2["exit"] is None and not res_w2["timeout"], res_w2)

    _, _, res_bd = _run_amain(
        ["goodbye seyon"], cfg_extra={"mic_mode": "open"},
        ears=_BadDeviceEars(), feed_delay=0.4,
        explain=lambda e: True)
    check("a recognized device failure falls back quiet but clean",
          res_bd["exit"] is None and not res_bd["timeout"], res_bd)

    _, _, res_q = _run_amain(
        ["goodbye seyon"], cfg_extra={"mic_mode": "open"}, ears=_QuitEars(),
        feed_delay=0.3)
    check("a quit phrase captured by the open mic hangs up",
          res_q["exit"] is None and not res_q["timeout"], res_q)

    def rec_boom(is_held, on_release=None, on_partial=None):
        if on_release:
            on_release()
        raise RuntimeError("invalid device")

    _, _, res_rb = _run_amain(
        ["goodbye seyon"], cfg_extra={"mic_mode": "ptt"},
        ptt=_FakePTT(), record=rec_boom, feed_delay=0.5,
        explain=lambda e: True)
    check("a device failure during a held key is spoken plainly",
          res_rb["exit"] is None and not res_rb["timeout"], res_rb)

    def rec_boom_hard(is_held, on_release=None, on_partial=None):
        if on_release:
            on_release()
        raise RuntimeError("strange")

    _, _, res_rbh = _run_amain(
        ["goodbye seyon"], cfg_extra={"mic_mode": "ptt"},
        ptt=_FakePTT(), record=rec_boom_hard, feed_delay=0.5)
    check("an unrecognized record failure is only logged",
          res_rbh["exit"] is None and not res_rbh["timeout"], res_rbh)

    mouth_cf, _, res_cf = _run_amain(
        ["stop asking for permission", "goodbye seyon"])
    check("a quit phrase cancels a pending confirm and hangs up",
          res_cf["exit"] is None and not res_cf["timeout"], res_cf)
    check("the signoff still plays",
          "Bye." in mouth_cf.said, mouth_cf.said)

    def rec_quit(is_held, on_release=None, on_partial=None):
        if on_release:
            on_release()
        return "goodbye seyon"

    _, _, res_rq = _run_amain(
        ["goodbye seyon"], cfg_extra={"mic_mode": "ptt"},
        ptt=_FakePTT(), record=rec_quit, feed_delay=0.5)
    check("a quit phrase spoken into the key hangs up",
          res_rq["exit"] is None and not res_rq["timeout"], res_rq)

    check("the signoff still plays",
          "Bye." in mouth_cf.said, mouth_cf.said)

    def rec_hangup(is_held, on_release=None, on_partial=None):
        if on_release:
            on_release()
        return "goodbye seyon"

    mouth_kb, _, res_kb = _run_amain(
        ["clear the session"], brain_cls=_KbBrain)
    check("a KeyboardInterrupt out of a turn unwinds the loop cleanly",
          res_kb["exit"] is None and not res_kb["timeout"], res_kb)
    check("an interrupt does not play the signoff",
          "Bye." not in mouth_kb.said, mouth_kb.said)

    _, _, res_pi = _run_amain(
        ["please answer"], cfg_extra={"mic_mode": "ptt"},
        brain_cls=_SlowBrain, ptt=_TimedPress(0.15), record=rec_quit)
    check("a key press while a reply streams interrupts and hangs up",
          res_pi["exit"] is None and not res_pi["timeout"], res_pi)

    _, _, res_st = _run_amain(
        ["push to talk mode", "go hands free"],
        cfg_extra={"mic_mode": "open"}, ears=_StaleCaptureEars(),
        ptt=_TimedPress(0.6), record=rec_quit)
    check("a capture born before a mode switch is discarded unprocessed",
          res_st["exit"] is None and not res_st["timeout"], res_st)

    _, _, res_mg = _run_amain(
        ["push to talk mode", "goodbye seyon"],
        cfg_extra={"mic_mode": "open"}, ears=_InstantEars(),
        ptt=_TimedPress(0.02), record=_noop_record,
        brain_cls=_YieldingBrain)
    check("a press and a capture that finished under the old mode are eaten",
          res_mg["exit"] is None and not res_mg["timeout"], res_mg)


def screen_look():
    print("\n--- amain: the spoken look sends one screenshot ---")
    mouth, _, res = _run_amain(
        ["look at my screen", "goodbye seyon"], brain_cls=_LookBrain)
    check("the look run hangs up cleanly",
          res["exit"] is None and not res["timeout"], res)
    brain = _LookBrain.instances[-1]
    check("the screenshot reached the brain",
          brain.images == [b"PNG"], brain.images)
    check("the answer about the screen is spoken",
          any("That is your editor." in c for c in mouth.chunks), mouth.chunks)

    mouth2, _, res2 = _run_amain(
        ["look at my screen", "goodbye seyon"], brain_cls=_LookBrain,
        screen=_FakeScreen(raises=RuntimeError("no display")))
    check("a look with no display fails cleanly",
          res2["exit"] is None and not res2["timeout"], res2)
    check("and says it cannot see",
          any("can't see your screen" in s for s in mouth2.said), mouth2.said)


def activity_tracker():
    print("\n--- amain: the activity tracker notes the focused window ---")
    _LoopJournal.instances.clear()
    mouth, _, res = _run_amain(
        ["goodbye seyon"],
        cfg_extra={"screen_enabled": True, "screen_track_interval": 0.02},
        screen=_FakeScreen(changes=("code.exe \u2014 main.py",)))
    check("the tracker run hangs up cleanly",
          res["exit"] is None and not res["timeout"], res)
    journal = _LoopJournal.instances[-1]
    check("the focused window was journaled",
          ("activity", "code.exe \u2014 main.py") in journal.events,
          journal.events)
    check("the sampling interval came from config",
          _LAST_SIGNALS.active_window.called, _LAST_SIGNALS.mock_calls)
    check("the label also went on the signal bus",
          mock.call("code.exe \u2014 main.py")
          in _LAST_SIGNALS.active_window.call_args_list,
          _LAST_SIGNALS.active_window.call_args_list)


def check_in_decision():
    print("\n--- check_in_due: who may speak first ---")
    d = btmain.check_in_due
    check("an idle, switched-on line may start talking",
          d(True, 30, 60, False, False, False) is True)
    check("switched off means never",
          d(False, 30, 60, False, False, False) is False)
    check("a zero interval means never",
          d(True, 0, 60, False, False, False) is False)
    check("too short a quiet is not yet",
          d(True, 30, 10, False, False, False) is False)
    check("a speaking mouth keeps it quiet",
          d(True, 30, 60, True, False, False) is False)
    check("a reply in flight keeps it quiet",
          d(True, 30, 60, False, True, False) is False)
    check("a pending permission ask keeps it quiet",
          d(True, 30, 60, False, False, True) is False)
    check("a spoken pause keeps it quiet",
          d(True, 30, 60, False, False, False, paused=True) is False)
    check("quiet hours keep it quiet",
          d(True, 30, 60, False, False, False, quiet=True) is False)
    check("a held app keeps it quiet",
          d(True, 30, 60, False, False, False, held=True) is False)


def checkin_control_text():
    print("\n--- checkin_control: the spoken quiet phrases ---")
    ctl = btmain.checkin_control
    check("a plain stay quiet uses the default",
          ctl("stay quiet for a while", 3600) == ("pause", 3600.0))
    check("a bare be quiet uses the default",
          ctl("be quiet", 3600) == ("pause", 3600.0))
    check("let me work uses the default",
          ctl("let me work for a bit", 3600) == ("pause", 3600.0))
    check("minutes are read as digits",
          ctl("stay quiet for 20 minutes", 3600) == ("pause", 1200.0))
    check("hours are read as digits",
          ctl("quiet for 2 hours", 3600) == ("pause", 7200.0))
    check("minutes are read as words",
          ctl("quiet for twenty minutes", 3600) == ("pause", 1200.0))
    check("an hour maps to 3600",
          ctl("quiet for an hour", 3600) == ("pause", 3600.0))
    check("half an hour maps to 1800",
          ctl("quiet for half an hour", 3600) == ("pause", 1800.0))
    check("seconds are read too",
          ctl("quiet for 30 seconds", 3600) == ("pause", 30.0))
    check("an unreadable number falls back to the default",
          ctl("quiet for umpteen minutes", 900) == ("pause", 900.0))
    check("resume phrases bring it back",
          ctl("you can check in again", 3600) == ("resume", 0))
    check("check in again brings it back",
          ctl("check on me again please", 3600) == ("resume", 0))
    check("anything else is not a command",
          ctl("what is the weather", 3600) is None)


def checkin_quiet_helpers():
    print("\n--- quiet-hours, held apps, and the spoken pause text ---")
    q = btmain.in_quiet_hours
    check("a daytime hour is not quiet", q(12, 23, 7) is False)
    check("late night is quiet", q(23, 23, 7) is True)
    check("early morning is quiet", q(3, 23, 7) is True)
    check("the end hour is not quiet", q(7, 23, 7) is False)
    check("an empty window is never quiet", q(12, 5, 5) is False)
    check("a same-day window works", q(12, 9, 17) is True)
    check("outside a same-day window is not quiet", q(20, 9, 17) is False)
    h = btmain.held_by_app
    check("a matching hold holds", h("zoom.exe Weekly sync", ["zoom"]) is True)
    check("no hold terms means never held",
          btmain.held_by_app("zoom.exe Meeting", []) is False)
    check("a non-matching window is not held",
          btmain.held_by_app("code.exe", ["zoom"]) is False)
    check("an empty label is not held",
          btmain.held_by_app("", ["zoom"]) is False)
    s = btmain._spoken_quiet
    check("hours are spoken in hours", "1 hour" in s(3600), s(3600))
    check("many hours are plural", "2 hours" in s(7200), s(7200))
    check("minutes are spoken as minutes",
          "30 minutes" in s(1800), s(1800))
    check("a single minute is singular", "1 minute" in s(60), s(60))
    check("seconds speak as seconds", "30 seconds" in s(30), s(30))
    check("nothing reads as no check-ins", s(0) == "Okay, no check-ins for now.")


def reminder_and_stop_helpers():
    print("\n--- reminders and stop words: the pure wording ---")
    r = btmain.reminder_request
    check("a plain remind-me with minutes is read",
          r("remind me in 20 minutes to stretch") == (1200.0, "stretch"))
    check("a timer with no label is read",
          r("set a timer for 5 minutes") == (300.0, ""))
    check("an hour looks past the hour word",
          r("remind me in an hour to call mom") == (3600.0, "call mom"))
    check("half an hour is understood",
          r("remind me in half an hour to rest") == (1800.0, "rest"))
    check("no length named reads as zero seconds",
          r("remind me to water the plants") == (0, "water the plants"))
    check("a non-reminder is not a reminder",
          r("what is the weather") is None)
    check("hours are spoken as hours",
          btmain._human_span(3600) == "1 hour")
    check("minutes are spoken as minutes",
          btmain._human_span(1800) == "30 minutes")
    check("seconds are spoken as seconds",
          btmain._human_span(30) == "30 seconds")
    check("a single second stays singular",
          btmain._human_span(1) == "1 second")
    check("the reminder acknowledgement names the span",
          btmain._spoken_reminder(1200, "stretch")
          == "Okay, I'll remind you in 20 minutes to stretch.")
    check("the acknowledgement drops an empty label",
          btmain._spoken_reminder(300, "")
          == "Okay, I'll remind you in 5 minutes.")
    check("a reminder with a label is spoken",
          btmain._reminder_line("stretch") == "Reminder: stretch.")
    check("a label-less reminder is a timer",
          btmain._reminder_line("") == "Your timer is up.")
    s = btmain.stop_command
    check("a bare stop is a stop", s("stop") is True)
    check("stop it is a stop", s("stop it") is True)
    check("cancel is a stop", s("cancel") is True)
    check("never mind is a stop", s("never mind") is True)
    check("a sentence with stop is not a stop",
          s("stop the docker container") is False)
    check("an unrelated phrase is not a stop", s("what time is it") is False)


def reminders():
    print("\n--- amain: a spoken reminder fires on time ---")
    mouth, _, res = _run_amain(
        ["remind me in 1 second to stretch", (1.4, "goodbye seyon")])
    check("the reminder run hangs up cleanly",
          res["exit"] is None and not res["timeout"], res)
    check("the reminder was acknowledged",
          any("I'll remind you in 1 second to stretch" in s
              for s in mouth.said), mouth.said)
    check("the reminder fired when its moment came",
          "Reminder: stretch." in mouth.said, mouth.said)

    mouth2, _, res2 = _run_amain(
        ["set a timer for 1 second", (1.4, "goodbye seyon")])
    check("a label-less timer hangs up cleanly",
          res2["exit"] is None and not res2["timeout"], res2)
    check("a label-less timer says time is up",
          "Your timer is up." in mouth2.said, mouth2.said)

    mouth3, _, res3 = _run_amain(
        ["remind me to water the plants", "goodbye seyon"])
    check("a timeless reminder asks for a time",
          any("How long from now" in s for s in mouth3.said), mouth3.said)


def stop_words():
    print("\n--- amain: a spoken stop cuts the reply ---")
    mouth, _, res = _run_amain(
        ["ask one", "stop", "goodbye seyon"], brain_cls=_SlowRecordBrain)
    check("the stop run hangs up cleanly",
          res["exit"] is None and not res["timeout"], res)
    brain = _LoopBrain.instances[-1]
    check("the stop word never reached the brain",
          brain.got == ["ask one"], brain.got)
    check("the stop is acknowledged mid-reply",
          "Okay." in mouth.said, mouth.said)

    mouth2, _, res2 = _run_amain(["stop", "goodbye seyon"])
    check("a stop with nothing playing still hangs up",
          res2["exit"] is None and not res2["timeout"], res2)
    brain2 = _LoopBrain.instances[-1]
    check("a silent stop is not sent to the brain",
          "stop" not in brain2.got, brain2.got)
    check("a silent stop is not acknowledged",
          "Okay." not in mouth2.said, mouth2.said)


def board_control_text():
    print("\n--- board_command: the spoken board ---")
    b = btmain.board_command
    check("clear the board is a clear", b("clear the board") == {"a": "clear"})
    check("sweep it is a clear", b("sweep it") == {"a": "clear"})
    check("wipe that now is a clear", b("wipe that now") == {"a": "clear"})
    check("reset the board is a reset",
          b("reset the board") == {"a": "reset"})
    check("explode it is an explode", b("explode it") == {"a": "explode"})
    check("assemble is an assemble", b("assemble") == {"a": "assemble"})
    check("put X on the board presents it",
          b("put the plan on the board") == {"a": "present",
                                             "title": "the plan"})
    check("show me X on the board drops the me",
          b("show me the numbers on the board")
          == {"a": "present", "title": "the numbers"})
    check("present X is a present",
          b("present the roadmap") == {"a": "present", "title": "the roadmap"})
    check("on the glass lands too",
          b("throw the budget on the glass")
          == {"a": "present", "title": "the budget"})
    check("a bare title after the tail is a present",
          b("the roadmap on the board")
          == {"a": "present", "title": "the roadmap"})
    check("ordinary speech is not a board command",
          b("what is on the board") is None)
    check("a question about a screen is not a command",
          b("show me how to fix the screen") is None)
    check("an empty present is nothing", b("present") is None)
    check("an unrelated sentence is not a board command",
          b("let's go for a walk") is None)


def board_control():
    print("\n--- amain: a spoken board command reaches the glass ---")
    sent = []
    with mock.patch.object(btmain.board, "_URL", "http://127.0.0.1:8794"), \
         mock.patch.object(btmain.board, "send",
                           side_effect=lambda c: sent.append(c) or True):
        mouth, _, res = _run_amain(
            ["clear the board", "put the plan on the board", "goodbye seyon"])
    check("the board run hangs up cleanly",
          res["exit"] is None and not res["timeout"], res)
    check("both commands were sent",
          sent == [{"a": "clear"}, {"a": "present", "title": "the plan"}],
          sent)
    check("the clear is acknowledged", "Board's clear." in mouth.said,
          mouth.said)
    check("the present is acknowledged",
          "It's up on the board." in mouth.said, mouth.said)

    with mock.patch.object(btmain.board, "_URL", "http://127.0.0.1:8794"), \
         mock.patch.object(btmain.board, "send", return_value=False):
        mouth2, _, res2 = _run_amain(["explode it", "goodbye seyon"])
    check("an unreachable board still hangs up cleanly",
          res2["exit"] is None and not res2["timeout"], res2)
    check("an unreachable board is admitted",
          "I couldn't reach the board." in mouth2.said, mouth2.said)

    with mock.patch.object(btmain.board, "_URL", ""):
        mouth3, _, res3 = _run_amain(["clear the board", "goodbye seyon"])
    check("with no board, a board phrase still hangs up cleanly",
          res3["exit"] is None and not res3["timeout"], res3)
    brain3 = _LoopBrain.instances[-1]
    check("with no board, a board phrase is ordinary talk",
          "clear the board" in brain3.got, brain3.got)


def recall_text():
    print("\n--- recall_request: the spoken recall ---")
    r = btmain.recall_request
    check("a remember-about question yields the topic",
          r("what do you remember about the database") == "the database")
    check("when-did-we-talk yields the topic",
          r("when did we talk about kokoro") == "kokoro")
    check("a last-talk question yields the topic",
          r("when did we last talk about the billing plan")
          == "the billing plan")
    check("do-you-remember yields the topic",
          r("do you remember postgres") == "postgres")
    check("recall X yields the topic",
          r("recall the billing plan") == "the billing plan")
    check("a bare recall is nothing", r("recall") is None)
    check("an ordinary question is not a recall",
          r("how are you doing today") is None)


def recall():
    print("\n--- amain: a spoken recall reads the journal ---")
    d = tempfile.mkdtemp(prefix="bt_main_recall_")
    with open(os.path.join(d, "voice-session_2026-10-01_09-00-00.md"), "w",
              encoding="utf-8") as f:
        f.write("**you:** we should use postgres for billing\n")
    mouth, _, res = _run_amain(
        ["what do you remember about postgres", "hello there",
         "goodbye seyon"], cfg_extra={"journal_dir": d})
    check("the recall run hangs up cleanly",
          res["exit"] is None and not res["timeout"], res)
    check("the journal answer is spoken",
          any(s.startswith("On 2026-10-01 09:00") for s in mouth.said),
          mouth.said)
    brain = _LoopBrain.instances[-1]
    check("the recall never reached the brain",
          "what do you remember about postgres" not in brain.got, brain.got)
    check("ordinary talk still reaches the brain",
          "hello there" in brain.got, brain.got)

    mouth2, _, res2 = _run_amain(
        ["what do you remember about postgres", "goodbye seyon"])
    brain2 = _LoopBrain.instances[-1]
    check("with no journal, a recall question is ordinary talk",
          "what do you remember about postgres" in brain2.got, brain2.got)
    check("the no-journal run hangs up cleanly",
          res2["exit"] is None and not res2["timeout"], res2)


def partial_caption():
    print("\n--- amain: partial_stt wires the heard line, then clears it ---")
    mouth, _, res = _run_amain(
        ["hello there", "goodbye seyon"],
        cfg_extra={"partial_stt": True})
    check("the partial-stt run hangs up cleanly",
          res["exit"] is None and not res["timeout"], res)
    check("each taken turn clears the heard line",
          _LAST_SIGNALS.heard_clear.called)

    mouth2, _, res2 = _run_amain(
        ["hello there", "goodbye seyon"],
        cfg_extra={"partial_stt": False})
    check("with partial_stt off, the heard line is left alone",
          res2["exit"] is None and not res2["timeout"]
          and not _LAST_SIGNALS.heard_clear.called, res2)


def check_in_quiet_and_pause():
    print("\n--- amain: quiet hours and a spoken pause hold check-ins ---")
    # A spoken "stay quiet" must stop a check-in that would otherwise fire.
    mouth, _, res = _run_amain(
        ["stay quiet for a while", "goodbye seyon"],
        cfg_extra={"check_in_enabled": True, "check_in_interval": 0.02},
        ears=_QuietEars(), feed_delay=0.05)
    check("the paused run hangs up cleanly",
          res["exit"] is None and not res["timeout"], res)
    brain = _LoopBrain.instances[-1]
    check("a spoken pause holds the check-in back",
          btmain.CHECK_IN_PROMPT not in brain.got, brain.got)
    check("and it says it will be quiet",
          any("Quiet for" in s for s in mouth.said), mouth.said)

    # Resume brings them back, and the control is not sent to the brain.
    mouth2, _, res2 = _run_amain(
        ["you can check in again", "goodbye seyon"],
        cfg_extra={"check_in_enabled": True, "check_in_interval": 0.02},
        ears=_QuietEars(), feed_delay=0.05)
    check("the resumed run hangs up cleanly",
          res2["exit"] is None and not res2["timeout"], res2)
    check("resume is acknowledged",
          any("back on" in s for s in mouth2.said), mouth2.said)


def check_in_quiet_hours():
    print("\n--- amain: quiet hours hold check-ins by the clock ---")
    hour = types.SimpleNamespace(tm_hour=23)
    with mock.patch.object(btmain.time, "localtime", lambda *a: hour):
        _, _, res = _run_amain(
            ["goodbye seyon"],
            cfg_extra={"check_in_enabled": True, "check_in_interval": 0.02,
                       "check_in_quiet_hours": [23, 7]},
            ears=_QuietEars(), feed_delay=0.35)
    check("the quiet-hours run hangs up cleanly",
          res["exit"] is None and not res["timeout"], res)
    brain = _LoopBrain.instances[-1]
    check("an overnight hour holds the check-in back",
          btmain.CHECK_IN_PROMPT not in brain.got, brain.got)


def check_in_held_app():
    print("\n--- amain: a focused meeting/game holds check-ins ---")
    screen = _FakeScreen(window=("zoom.exe", "Weekly sync - Zoom"))
    _, _, res = _run_amain(
        ["goodbye seyon"],
        cfg_extra={"check_in_enabled": True, "check_in_interval": 0.02,
                   "check_in_hold_apps": ["zoom"]},
        ears=_QuietEars(), screen=screen, feed_delay=0.35)
    check("the held-app run hangs up cleanly",
          res["exit"] is None and not res["timeout"], res)
    brain = _LoopBrain.instances[-1]
    check("a held app keeps the line quiet",
          btmain.CHECK_IN_PROMPT not in brain.got, brain.got)
    check("the focused window was actually read",
          screen.window_reads > 0, screen.window_reads)


def check_in_followup():
    print("\n--- amain: the follow-up window answers without the wake word ---")
    mouth, _, res = _run_amain(
        ["hello"],
        cfg_extra={"mic_mode": "open", "wake_word": True,
                   "wake_word_phrases": ["vijay seyon"],
                   "followup_window": 3.0},
        ears=_FollowEars(text="goodbye seyon"), feed_delay=0.02)
    check("the follow-up run hangs up cleanly",
          res["exit"] is None and not res["timeout"], res)
    check("the reply was spoken",
          any("Answer one." in c for c in mouth.chunks), mouth.chunks)


def voice_enrollment():
    print("\n--- amain: --enroll learns a voice, and the gate reads it ---")
    with tempfile.TemporaryDirectory() as td:
        prof = os.path.join(td, "vp.json")
        mouth, _, res = _run_amain([], argv=["--enroll"],
                                   ears=_EnrollEars(),
                                   cfg_extra={"speaker_profile": prof})
        check("the enroll run exits cleanly",
              res["exit"] is None and not res["timeout"], res)
        check("a voiceprint was written", os.path.exists(prof))
        check("enrollment is spoken back",
              any("Enrolled" in s for s in mouth.said), mouth.said)

    with tempfile.TemporaryDirectory() as td:
        prof = os.path.join(td, "vp.json")
        mouth, _, res = _run_amain(
            [], argv=["--enroll"], ears=_EnrollEars(clip=np.zeros(100, np.int16)),
            cfg_extra={"speaker_profile": prof})
        check("a failed enroll still exits cleanly",
              res["exit"] is None and not res["timeout"], res)
        check("no print is written when the clips are unusable",
              not os.path.exists(prof))
        check("the line admits it did not catch enough",
              any("enough of your voice" in s for s in mouth.said), mouth.said)

    with tempfile.TemporaryDirectory() as td:
        prof = os.path.join(td, "vp.json")
        clip = (np.random.default_rng(1).standard_normal(16000)
                * 3000).astype(np.int16)
        vp.save_profile(prof, vp.feature(clip))
        ears_fake = _GateEars("vijay seyon")
        _, _, res = _run_amain(
            ["goodbye seyon"],
            cfg_extra={"mic_mode": "open", "wake_word": True,
                       "wake_word_phrases": ["vijay seyon"],
                       "speaker_id": True, "speaker_profile": prof},
            ears=ears_fake, feed_delay=0.02)
        check("the gated run hangs up cleanly",
              res["exit"] is None and not res["timeout"], res)
        check("the owner check reached the wake gate",
              any(c is not None for c in ears_fake.owner_checks),
              ears_fake.owner_checks)

    _, _, res = _run_amain(
        ["goodbye seyon"],
        cfg_extra={"speaker_id": True, "speaker_profile":
                   os.path.join(tempfile.gettempdir(), "no-such-vp.json")})
    check("speaker_id with no print runs ungated and clean",
          res["exit"] is None and not res["timeout"], res)


def voice_whoami():
    print("\n--- amain: --whoami reports the score without arming the gate ---")
    clip = (np.random.default_rng(3).standard_normal(16000)
            * 3000).astype(np.int16)
    with tempfile.TemporaryDirectory() as td:
        prof = os.path.join(td, "vp.json")
        mouth, _, res = _run_amain([], argv=["--whoami"],
                                   ears=_EnrollEars(),
                                   cfg_extra={"speaker_profile": prof})
        check("whoami with no print exits cleanly",
              res["exit"] is None and not res["timeout"], res)
        check("whoami asks for an enrollment first",
              any("not learned your voice" in s for s in mouth.said), mouth.said)

    with tempfile.TemporaryDirectory() as td:
        prof = os.path.join(td, "vp.json")
        vp.save_profile(prof, vp.feature(clip), threshold=0.5)
        mouth, _, res = _run_amain([], argv=["--whoami"],
                                   ears=_EnrollEars(clip=clip),
                                   cfg_extra={"speaker_profile": prof})
        check("whoami with a print exits cleanly",
              res["exit"] is None and not res["timeout"], res)
        check("whoami says a matching voice is the owner",
              any("scored" in s and "that's you" in s for s in mouth.said),
              mouth.said)

    with tempfile.TemporaryDirectory() as td:
        prof = os.path.join(td, "vp.json")
        vp.save_profile(prof, vp.feature(clip), threshold=2.0)
        mouth, _, res = _run_amain([], argv=["--whoami"],
                                   ears=_EnrollEars(clip=clip),
                                   cfg_extra={"speaker_profile": prof})
        check("whoami says a below-floor voice is not the owner",
              any("not quite you" in s for s in mouth.said), mouth.said)

    with tempfile.TemporaryDirectory() as td:
        prof = os.path.join(td, "vp.json")
        vp.save_profile(prof, vp.feature(clip), threshold=0.5)
        mouth, _, res = _run_amain(
            [], argv=["--whoami"],
            ears=_EnrollEars(clip=np.zeros(100, np.int16)),
            cfg_extra={"speaker_profile": prof})
        check("whoami with an unusable clip exits cleanly",
              res["exit"] is None and not res["timeout"], res)
        check("whoami admits it could not judge",
              any("enough of your voice to judge" in s for s in mouth.said),
              mouth.said)


def check_in():
    print("\n--- amain: the assistant can speak first ---")
    mouth, _, res = _run_amain(
        ["goodbye seyon"],
        cfg_extra={"check_in_enabled": True, "check_in_interval": 0.02},
        ears=_QuietEars(), feed_delay=0.35)
    check("the check-in run hangs up cleanly",
          res["exit"] is None and not res["timeout"], res)
    brain = _LoopBrain.instances[-1]
    check("a check-in reached the brain unprompted",
          btmain.CHECK_IN_PROMPT in brain.got, brain.got)
    check("the check-in reply was spoken",
          any("Answer one." in c for c in mouth.chunks), mouth.chunks)


def check_in_prompt_activity():
    print("\n--- amain: a check-in folds in what they were doing ---")
    screen = _FakeScreen(changes=("code.exe \u2014 main.py",))
    _, _, res = _run_amain(
        ["goodbye seyon"],
        cfg_extra={"check_in_enabled": True, "check_in_interval": 0.02,
                   "screen_enabled": True, "screen_track_interval": 0.02},
        ears=_QuietEars(), screen=screen, feed_delay=0.35)
    check("the specific check-in run hangs up cleanly",
          res["exit"] is None and not res["timeout"], res)
    brain = _LoopBrain.instances[-1]
    check("the check-in named the last activity",
          any("code.exe \u2014 main.py" in got for got in brain.got),
          brain.got)


def check_in_defer():
    print("\n--- amain: a busy line does not speak first ---")
    mouth, _, res = _run_amain(
        ["goodbye seyon"],
        cfg_extra={"check_in_enabled": True, "check_in_interval": 0.02},
        ears=_QuietEars(), mouth=_BusyMouth(), feed_delay=0.35)
    check("the held-back run hangs up cleanly",
          res["exit"] is None and not res["timeout"], res)
    brain = _LoopBrain.instances[-1]
    check("a speaking mouth holds the check-in back",
          btmain.CHECK_IN_PROMPT not in brain.got, brain.got)


def tty_reader():
    print("\n--- _typed_reader: the POSIX line editor ---")
    reads = iter([
        b"Hel",
        b"lo\x7f\x08p",              # backspace twice, then "p" -> "Help"
        b"\x1b[20",                  # partial paste-on marker, held back
        b"0~" + b"X" * 70,           # marker completes, 70-char paste body
        b"\x1b[20",                  # partial paste-off marker, held back
        b"1~",                       # paste closes (70 chars -> a count)
        b"\r",                       # Enter sends the composed line
        b"",                         # EOF ends the reader
    ])

    def fake_read(fd, n):
        return next(reads, b"")

    termios = types.ModuleType("termios")
    termios.TCSADRAIN = 1
    termios.tcgetattr = lambda fd: ["saved"]
    termios.tcsetattr = lambda fd, when, attrs: None
    tty = types.ModuleType("tty")
    tty.setcbreak = lambda fd: None

    out = io.StringIO()
    fake_sys = types.SimpleNamespace(
        stdin=types.SimpleNamespace(fileno=lambda: 0),
        stdout=out)
    q = queue.Queue()
    with mock.patch.dict(sys.modules, {"termios": termios, "tty": tty}), \
         mock.patch.object(btmain.os, "isatty", lambda fd: True), \
         mock.patch.object(btmain.os, "read", fake_read), \
         mock.patch.object(btmain, "sys", fake_sys):
        btmain._typed_reader(q)

    got = []
    while not q.empty():
        got.append(q.get_nowait())
    line = "Help " + "X" * 70
    check("the tty editor assembled the line across reads",
          got == [line], got)
    check("a long paste collapsed to a count",
          "[pasted 70 chars]" in out.getvalue(), out.getvalue()[:80])
    check("backspace erased characters in the echo",
          "\b \b" in out.getvalue(), out.getvalue()[:80])
    check("the reader restored the terminal on the way out",
          "\x1b[?2004l" in out.getvalue(), out.getvalue()[:80])


class _Stop(Exception):
    pass


def tty_reader_branches():
    print("\n--- _typed_reader: the odd branches ---")

    def make_termios(boom=False):
        m = types.ModuleType("termios")
        m.TCSADRAIN = 1
        m.tcgetattr = lambda fd: ["saved"]

        def setter(fd, when, attrs):
            if boom:
                raise OSError("gone")

        m.tcsetattr = setter
        return m

    def make_sys(out):
        return types.SimpleNamespace(
            stdin=types.SimpleNamespace(fileno=lambda: 0),
            stdout=out)

    with mock.patch.dict(sys.modules, {"termios": None, "tty": None}), \
         mock.patch.object(btmain.os, "isatty", lambda fd: True), \
         mock.patch.object(btmain, "_typed_reader_simple") as simple, \
         mock.patch.object(btmain, "sys", make_sys(io.StringIO())):
        btmain._typed_reader(queue.Queue())
    check("a platform without termios uses the simple reader", simple.called)

    out = io.StringIO()
    tty_ok = types.ModuleType("tty")
    tty_ok.setcbreak = lambda fd: None
    with mock.patch.dict(sys.modules, {"termios": make_termios(boom=True),
                                       "tty": tty_ok}), \
         mock.patch.object(btmain.os, "isatty", lambda fd: True), \
         mock.patch.object(btmain.os, "read", side_effect=OSError("gone")), \
         mock.patch.object(btmain, "sys", make_sys(out)):
        btmain._typed_reader(queue.Queue())
    check("a read error restores the terminal and stops",
          "\x1b[?2004l" in out.getvalue(), out.getvalue()[:40])

    reads = iter([
        b"\x01",                    # a control char: ignored
        b"\x7f",                    # backspace with an empty buffer
        b"\r",                      # Enter on an empty buffer: nothing queued
        b"hi ",                     # a line ending in a space
        b"\x1b[200~\x1b[201~",      # an empty paste: nothing to insert
        b"\x1b[200~body\x1b[201~",  # a real paste onto the "hi " buffer
        b"\r",                      # send "hi body"
        b"",                        # EOF
    ])

    def fake_read(fd, n):
        return next(reads, b"")

    out2 = io.StringIO()
    tty = types.ModuleType("tty")
    tty.setcbreak = lambda fd: None
    q = queue.Queue()
    with mock.patch.dict(sys.modules, {"termios": make_termios(), "tty": tty}), \
         mock.patch.object(btmain.os, "isatty", lambda fd: True), \
         mock.patch.object(btmain.os, "read", fake_read), \
         mock.patch.object(btmain, "sys", make_sys(out2)):
        btmain._typed_reader(q)
    got = []
    while not q.empty():
        got.append(q.get_nowait())
    check("the odd branches still compose the right line",
          got == ["hi body"], got)


def remember_and_focus():
    print("\n--- remember/focus: the pure wording ---")
    r = btmain.remember_request
    check("a remember-that is kept whole",
          r("remember that the wifi password is hunter2")
          == "the wifi password is hunter2")
    check("a make-a-note drops the opener",
          r("make a note of the spare key code") == "the spare key code")
    check("write this down keeps the case and words",
          r("write this down: Call Sara at 4") == "Call Sara at 4")
    check("a bare remember is not a save",
          r("remember the time we went to Rome") is None)
    check("an empty remember body is nothing",
          r("remember that ...") is None)
    check("ordinary talk is not a save",
          r("how are you today") is None)

    with tempfile.TemporaryDirectory() as td:
        path = btmain.remember_note(td, "the door code is 4321")
        with open(path, encoding="utf-8") as f:
            body = f.read()
        check("a kept line lands in a dated note",
              "the door code is 4321" in body and path.endswith(".md"), body)
        btmain.remember_note(td, "and the gate is 9")
        with open(path, encoding="utf-8") as f:
            body2 = f.read()
        check("a second keep appends, it does not replace",
              body2.count("- ") == 2, body2)

    f = btmain.focus_request
    check("focus for twenty minutes is twenty",
          f("focus for twenty minutes", 25) == 20)
    check("focus with no length is the default",
          f("focus mode", 25) == 25)
    check("a pomodoro uses the default",
          f("start a pomodoro", 50) == 50)
    check("do not disturb for half an hour is thirty",
          f("do not disturb for half an hour", 25) == 30)
    check("a zero default still lands at least a minute",
          f("focus mode", 0) == 25)
    check("focus the window is not a focus stretch",
          f("focus the window", 25) is None)
    check("ordinary talk is not a focus",
          f("let us take a walk", 25) is None)

    print("\n--- amain: remember and focus are handled in-line ---")
    with tempfile.TemporaryDirectory() as td:
        mouth, _, res = _run_amain(
            ["remember that the router is in the hall", "goodbye seyon"],
            cfg_extra={"remember_dir": td})
        check("the remember run hangs up cleanly",
              res["exit"] is None and not res["timeout"], res)
        brain = _LoopBrain.instances[-1]
        check("a kept line never reaches the brain",
              "remember that the router is in the hall" not in brain.got,
              brain.got)
        check("the keep is acknowledged aloud",
              any("written that down" in s for s in mouth.said), mouth.said)
        wrote = [p for p in os.listdir(td) if p.startswith("Remembered")]
        check("the keep landed on disk", len(wrote) == 1, wrote)

    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "blocker"), "w", encoding="utf-8") as f:
            f.write("x")
        mouth, _, res = _run_amain(
            ["remember that the router is in the hall", "goodbye seyon"],
            cfg_extra={"remember_dir": os.path.join(td, "blocker", "sub")})
        check("an unwritable notes folder still hangs up cleanly",
              res["exit"] is None and not res["timeout"], res)
        check("an unwritable notes folder is admitted",
              any("couldn't write that down" in s for s in mouth.said),
              mouth.said)

    mouth, _, res = _run_amain(["focus for five minutes", "goodbye seyon"])
    check("the focus run hangs up cleanly",
          res["exit"] is None and not res["timeout"], res)
    brain = _LoopBrain.instances[-1]
    check("a focus stretch never reaches the brain",
          "focus for five minutes" not in brain.got, brain.got)
    check("the focus stretch is spoken",
          any("focusing for 5 minutes" in s for s in mouth.said), mouth.said)

    mouth2, _, res2 = _run_amain(["focus mode", "goodbye seyon"])
    check("a default-length focus uses the configured minutes",
          any("focusing for 25 minutes" in s for s in mouth2.said),
          mouth2.said)


def remembered_readback():
    print("\n--- read-back: the pure bits ---")
    q = btmain.remembered_request
    check("what did i ask you to remember is a read-back",
          q("what did i ask you to remember") is True)
    check("read me my notes is a read-back", q("read me my notes") is True)
    check("what's in my notes is a read-back",
          q("what's in my notes") is True)
    check("remember that X is not a read-back",
          q("remember that the door is locked") is False)
    check("ordinary talk is not a read-back", q("how was your day") is False)

    s = btmain.spoken_remembered
    check("nothing kept is admitted plainly",
          s([]) == "You haven't asked me to remember anything yet.")
    check("one kept line is read alone",
          s(["call the dentist"]) == "You asked me to remember this: "
          "call the dentist.")
    check("several kept lines are read in one breath",
          s(["call the dentist", "buy milk"])
          == "Here's what you asked me to remember: call the dentist, and "
          "buy milk.")

    with tempfile.TemporaryDirectory() as td:
        check("an empty folder reads back nothing",
              btmain.read_remembered(td) == [])
        check("no folder at all reads back nothing",
              btmain.read_remembered("") == [])
        older = os.path.join(td, "Remembered 2026-10-12.md")
        newer = os.path.join(td, "Remembered 2026-10-13.md")
        with open(newer, "w", encoding="utf-8") as f:
            f.write("- 09:00 \u2014 the morning thing\n"
                    "- 17:00 \u2014 the newer thing\n"
                    "\n")
        with open(older, "w", encoding="utf-8") as f:
            f.write("- 09:00 \u2014 the older thing\n"
                    "- plain line with no separator\n")
        got = btmain.read_remembered(td, limit=10)
        check("the newest note and its newest line come first",
              got[0] == "the newer thing", got)
        check("the blank line is skipped, the separator is dropped",
              got == ["the newer thing", "the morning thing",
                      "plain line with no separator", "the older thing"], got)
        check("the limit is honoured",
              btmain.read_remembered(td, limit=1) == ["the newer thing"])
        os.mkdir(os.path.join(td, "Remembered 2026-01-01.md"))
        check("a note that will not open is skipped, not fatal",
              btmain.read_remembered(td, limit=10) == got)

    print("\n--- amain: a spoken read-back reads the kept notes ---")
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "Remembered 2026-10-13.md"), "w",
                  encoding="utf-8") as f:
            f.write("- 08:00 \u2014 call the dentist at four\n")
        mouth, _, res = _run_amain(
            ["what did i ask you to remember", "goodbye seyon"],
            cfg_extra={"remember_dir": td})
        check("the read-back run hangs up cleanly",
              res["exit"] is None and not res["timeout"], res)
        brain = _LoopBrain.instances[-1]
        check("a read-back never reaches the brain",
              "what did i ask you to remember" not in brain.got, brain.got)
        check("the kept line is read aloud",
              any("call the dentist at four" in s for s in mouth.said),
              mouth.said)

    with tempfile.TemporaryDirectory() as td:
        mouth, _, res = _run_amain(
            ["what did i ask you to remember", "goodbye seyon"],
            cfg_extra={"remember_dir": td})
        check("an empty notes folder is admitted",
              any("haven't asked me to remember" in s for s in mouth.said),
              mouth.said)


def briefing_cases():
    print("\n--- briefing: the pure bits ---")
    b = btmain.briefing_request
    check("brief me is a briefing", b("brief me") is True)
    check("a brief-me tail is a briefing",
          b("brief me on my day") is True)
    check("what is on my plate is a briefing",
          b("what's on my plate today") is True)
    check("ordinary talk is not a briefing",
          b("what time is it") is False)

    check("a new day is due", btmain.briefing_due("2026-10-12", "2026-10-13")
          is True)
    check("the same day is not due",
          btmain.briefing_due("2026-10-13", "2026-10-13") is False)
    check("no marker is due",
          btmain.briefing_due("", "2026-10-13") is True)

    with tempfile.TemporaryDirectory() as td:
        path, today = btmain.briefing_state_path(td)
        check("the marker is named for the day and sits in the folder",
              path.startswith(td) and path.endswith(".briefing_day"), path)
        check("a missing marker reads as empty",
              btmain.read_briefing_day(path) == "")
        btmain.write_briefing_day(path, today)
        check("a written marker reads back", btmain.read_briefing_day(path)
              == today)
        with open(os.path.join(td, "blocker"), "w", encoding="utf-8") as f:
            f.write("x")
        btmain.write_briefing_day(os.path.join(td, "blocker", "x"), today)
        check("an unwritable marker stays quiet",
              btmain.read_briefing_day(
                  os.path.join(td, "blocker", "x")) == "")

    print("\n--- amain: the spoken and the automatic briefing ---")
    mouth, _, res = _run_amain(["brief me", "goodbye seyon"])
    check("the briefing run hangs up cleanly",
          res["exit"] is None and not res["timeout"], res)
    brain = _LoopBrain.instances[-1]
    check("the briefing is framed for the brain",
          btmain._BRIEFING_PROMPT in brain.got, brain.got)
    check("the raw phrase is not sent verbatim",
          "brief me" not in brain.got, brain.got)

    with tempfile.TemporaryDirectory() as td:
        _, _, res2 = _run_amain(
            ["goodbye seyon"],
            cfg_extra={"daily_briefing": True, "signals_dir": td})
        check("the automatic briefing run hangs up cleanly",
              res["exit"] is None and not res["timeout"], res)
        brain = _LoopBrain.instances[-1]
        check("the first launch of the day briefs",
              btmain._BRIEFING_PROMPT in brain.got, brain.got)
        _, _, res3 = _run_amain(
            ["goodbye seyon"],
            cfg_extra={"daily_briefing": True, "signals_dir": td})
        brain3 = _LoopBrain.instances[-1]
        check("a second launch the same day does not",
              btmain._BRIEFING_PROMPT not in brain3.got, brain3.got)


def care_cases():
    print("\n--- care: the pure decision ---")
    c = btmain.care
    check("a low battery off the charger is a nudge",
          "battery is at 12 percent" in c.battery_nudge(12, False),
          c.battery_nudge(12, False))
    check("a charging battery is not a nudge",
          c.battery_nudge(12, True) is None)
    check("an unknown battery is not a nudge",
          c.battery_nudge(None, False) is None)
    check("a topped-up battery is not a nudge",
          c.battery_nudge(80, False) is None)
    check("a nearly-full disk is a nudge",
          "nearly full" in c.disk_nudge(2.0), c.disk_nudge(2.0))
    check("a roomy disk is not a nudge", c.disk_nudge(200.0) is None)
    check("an unknown disk is not a nudge", c.disk_nudge(None) is None)
    check("a long session is a nudge",
          "about 2 hours" in c.session_nudge(150.0), c.session_nudge(150.0))
    check("a short session is not a nudge", c.session_nudge(10.0) is None)
    check("an unknown session is not a nudge", c.session_nudge(None) is None)
    check("the first worthy fact wins",
          c.pick(c.facts(pct=10, on_ac=False)) == c.battery_nudge(10, False))
    check("no worthy fact is nothing",
          c.pick(c.facts(pct=90, on_ac=True, free_gb=500, minutes=5)) is None)
    check("enough quiet is due", c.due(1000.0, 0.0, 500.0) is True)
    check("too soon is not due", c.due(1000.0, 900.0, 500.0) is False)

    print("\n--- amain: a due care nudge colours the check-in ---")
    with mock.patch.object(btmain.care, "current_nudge",
                           return_value=("the battery is at 12 percent and "
                                         "the charger is not plugged in",
                                         999.0)):
        _, _, res = _run_amain(
            ["goodbye seyon"],
            cfg_extra={"check_in_enabled": True, "check_in_interval": 0.02,
                       "care_enabled": True},
            ears=_QuietEars(), feed_delay=0.35)
    check("the care check-in run hangs up cleanly",
          res["exit"] is None and not res["timeout"], res)
    brain = _LoopBrain.instances[-1]
    check("a due care nudge is folded into the check-in",
          any("battery is at 12 percent" in got for got in brain.got),
          brain.got)

    with mock.patch.object(btmain.care, "current_nudge",
                           return_value=(None, 0.0)):
        _, _, res2 = _run_amain(
            ["goodbye seyon"],
            cfg_extra={"check_in_enabled": True, "check_in_interval": 0.02,
                       "care_enabled": True},
            ears=_QuietEars(), feed_delay=0.35)
    brain2 = _LoopBrain.instances[-1]
    check("no care fact leaves the check-in plain",
          not any("you might also mention" in got for got in brain2.got),
          brain2.got)


def main_entry():
    print("\n--- main(): the process boundary ---")

    def run(claim=True, run_exc=None, log_boom=False, park_boom=False):
        rec = {"exit": [], "os_exit": [], "park": 0, "bus": 0, "log": []}

        def fake_run(coro):
            coro.close()
            if run_exc is not None:
                raise run_exc

        def fake_exit(code=0):
            rec["exit"].append(code)
            raise SystemExit(code)

        def fake_os_exit(code):
            rec["os_exit"].append(code)
            raise _Stop

        def fake_log(s):
            if log_boom:
                raise BaseException("the log itself died")
            rec["log"].append(s)

        def fake_park():
            if park_boom:
                raise BaseException("park died")
            rec["park"] += 1

        sig = types.SimpleNamespace(park=fake_park)
        with mock.patch.object(btmain, "_claim_single_instance",
                               lambda: claim), \
             mock.patch.object(btmain, "_take_the_bus",
                               lambda: rec.__setitem__("bus",
                                                       rec["bus"] + 1)), \
             mock.patch.object(btmain.asyncio, "run", fake_run), \
             mock.patch.object(btmain, "signals", sig), \
             mock.patch.object(btmain, "log", fake_log), \
             mock.patch.object(btmain.sys, "exit", fake_exit), \
             mock.patch.object(btmain.os, "_exit", fake_os_exit):
            try:
                btmain.main()
            except (SystemExit, _Stop):
                pass
        return rec

    rec = run(claim=False)
    check("a second voice line refuses to start",
          rec["exit"] == [1] and rec["bus"] == 0 and rec["park"] == 0, rec)

    rec = run()
    check("the clean path parks the bus and hard-exits 0",
          rec["os_exit"] == [0] and rec["park"] == 1 and rec["bus"] == 1,
          rec)

    rec = run(run_exc=KeyboardInterrupt())
    check("an interrupt still parks and hard-exits 0",
          rec["os_exit"] == [0] and rec["park"] == 1, rec)

    rec = run(run_exc=RuntimeError("kaboom"))
    check("a crash is logged and hard-exits 1",
          rec["os_exit"] == [1]
          and any("CRASH" in s for s in rec["log"]), rec)

    rec = run(run_exc=RuntimeError("kaboom"), log_boom=True)
    check("a crash path survives the log itself failing",
          rec["os_exit"] == [1], rec)

    rec = run(park_boom=True)
    check("the clean path survives a failing park",
          rec["os_exit"] == [0], rec)


print("=" * 66)
print("MAIN HELPERS, REPLY BATCHING, PERMISSION GATE (offline)")
print("=" * 66)
console()
formatting()
typed_input()
human_forms()
config_write()
speak_first_alone()
speak_empty_and_errors()
permission_gate()
gate_helpers()
journal_flush_case()
single_instance()
amain_loop()
screen_look()
activity_tracker()
check_in_decision()
checkin_control_text()
checkin_quiet_helpers()
reminder_and_stop_helpers()
reminders()
stop_words()
board_control_text()
board_control()
recall_text()
recall()
remember_and_focus()
remembered_readback()
briefing_cases()
care_cases()
voice_enrollment()
voice_whoami()
partial_caption()
check_in()
check_in_prompt_activity()
check_in_defer()
check_in_quiet_and_pause()
check_in_quiet_hours()
check_in_held_app()
check_in_followup()
tty_reader()
tty_reader_branches()
main_entry()

print("\n" + "=" * 66)
print("MAIN OK" if not FAILURES else f"MAIN FAILURES: {FAILURES}")
sys.exit(1 if FAILURES else 0)
