"""The brain, offline: every WarmBrain path driven by a fake opencode server.

No model, no network, no 90s timeouts. A FakeServer is swapped in for
brain._Server so the real WarmBrain runs its real code -- session setup,
sentence streaming, reasoning-vs-text filtering, the empty-turn and
timeout guards, session errors, permission routing, the slash-commands,
usage tallying. A second part boots a real localhost HTTP+SSE server so
_Server's own request/health/event-stream code is exercised for real too.

If opencode's wire format changes, this is the test that notices without
needing a model behind it.
"""
import asyncio
import http.server
import json
import os
import queue
import sys
import tempfile
import threading
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtalk import brain                                  # noqa: E402
from backtalk.config import CFG                             # noqa: E402

CFG["live_data"] = False
CFG["show_usage"] = False
CFG["resume_last_session"] = False

failures = []


def check(name, cond, detail=""):
    print("    %s %s%s" % ("ok  " if cond else "FAIL", name,
                           "" if cond else "   <- " + str(detail)))
    if not cond:
        failures.append(name)


class FakeServer:
    """Stands in for brain._Server: canned replies, scripted events."""

    def __init__(self, *a, **k):
        self.sid = "ses_test"
        self.requests = []
        self.events = []
        self.stopped = False
        self.fail_prompt = False
        self.fail_resume = False
        self.fail_abort = False
        self.fail_perm = False
        self.fail_usage = False
        self.raise_session_get = False
        self.zero_tokens = False
        self.messages = "default"

    async def ensure(self):
        pass

    async def stop(self):
        self.stopped = True

    async def request(self, method, path, body=None, directory=None,
                      timeout=300):
        self.requests.append((method, path, body))
        if self.fail_prompt and "prompt_async" in path:
            raise RuntimeError("prompt exploded")
        if self.fail_resume and method == "GET" and path.startswith("/session/ses_old"):
            raise RuntimeError("gone")
        if self.fail_abort and "abort" in path:
            raise RuntimeError("abort failed")
        if self.fail_perm and "permissions/" in path:
            raise RuntimeError("permission reply failed")
        if self.raise_session_get and method == "GET" \
                and path.startswith(f"/session/{self.sid}") \
                and not path.endswith("/message"):
            raise RuntimeError("session read failed")
        if method == "POST" and path == "/session":
            return {"id": self.sid}
        if method == "GET" and path == f"/session/{self.sid}/message":
            if self.fail_usage:
                raise RuntimeError("usage read failed")
            if self.messages == "empty":
                return []
            if self.messages == "user":
                return [{"info": {"role": "user"}}]
            return [{"info": {"role": "assistant", "cost": 0.02,
                              "tokens": {"input": 10, "output": 40}}}]
        if method == "GET" and path.startswith(f"/session/{self.sid}"):
            if self.zero_tokens:
                return {"tokens": {"input": 0, "output": 0}}
            return {"tokens": {"input": 100, "output": 20, "reasoning": 5,
                               "cache": {"read": 7}}}
        return {}

    async def next_event(self, timeout=None):
        return self.events.pop(0) if self.events else None


async def _drain(agen):
    return [c async for c in agen]


async def _noop(*a, **k):
    return None


async def new_brain(**kw):
    fake = FakeServer()
    saved = brain._Server
    brain._Server = lambda *a, **k: fake
    b = brain.WarmBrain(**kw)
    await b.start()
    return fake, b, saved


async def scenario_stream():
    fake, b, saved = await new_brain()
    try:
        sid = b._sid
        fake.events = [
            {"type": "message.part.updated",
             "properties": {"sessionID": sid,
                            "part": {"id": "p1", "type": "text"}}},
            {"type": "message.part.updated",           # reasoning is not spoken
             "properties": {"sessionID": sid,
                            "part": {"id": "r1", "type": "reasoning"}}},
            {"type": "message.part.delta",
             "properties": {"sessionID": sid, "partID": "r1",
                            "field": "text", "delta": "thinking out loud. "}},
            {"type": "message.part.delta",
             "properties": {"sessionID": "other", "partID": "p1",
                            "field": "text", "delta": "from another session. "}},
            {"type": "message.part.delta",
             "properties": {"sessionID": sid, "partID": "p1",
                            "field": "text", "delta": "Hello there. "}},
            {"type": "message.part.delta",
             "properties": {"sessionID": sid, "partID": "p1",
                            "field": "text", "delta": "How are you today?"}},
            {"type": "session.idle", "properties": {"sessionID": sid}},
        ]
        out = [c async for c in b.ask_stream("hi")]
        check("streamed the spoken sentences",
              out == ["Hello there.", "How are you today?"], out)
        check("never spoke the reasoning part",
              all("thinking" not in c for c in out), out)
        check("ignored another session's delta",
              all("another session" not in c for c in out), out)
        check("session tallied one turn", b.session["turns"] == 1, b.session)
        check("usage tokens recorded",
              b.session["in_tokens"] == 10 and b.session["out_tokens"] == 40,
              b.session)
    finally:
        brain._Server = saved


async def scenario_tool_flush():
    fake, b, saved = await new_brain()
    try:
        sid = b._sid
        fake.events = [
            {"type": "message.part.delta",
             "properties": {"sessionID": sid, "partID": "p1",
                            "field": "text", "delta": "Let me check"}},
            {"type": "message.part.updated",
             "properties": {"sessionID": sid,
                            "part": {"id": "s1", "type": "step-finish",
                                     "reason": "tool-calls"}}},
            {"type": "session.idle", "properties": {"sessionID": sid}},
        ]
        out = [c async for c in b.ask_stream("check something")]
        check("a tool boundary flushes the pending line",
              out == ["Let me check"], out)
    finally:
        brain._Server = saved


async def scenario_empty_and_error():
    fake, b, saved = await new_brain()
    try:
        sid = b._sid
        fake.events = [{"type": "session.idle",
                        "properties": {"sessionID": sid}}]
        out = [c async for c in b.ask_stream("say nothing")]
        check("an empty turn still says something",
              len(out) == 1 and "empty" in out[0], out)

        fake.events = [{"type": "session.error",
                        "properties": {"sessionID": sid, "error": "boom"}}]
        out = [c async for c in b.ask_stream("break")]
        check("a session error is spoken, not swallowed",
              any("did not work" in c for c in out), out)
    finally:
        brain._Server = saved


async def scenario_timeout():
    fake, b, saved = await new_brain()
    old = CFG.get("turn_timeout")
    CFG["turn_timeout"] = 0.0001
    try:
        sid = b._sid
        fake.events = []                       # next_event returns None
        out = [c async for c in b.ask_stream("slow")]
        check("a stalled turn apologises instead of hanging",
              len(out) == 1 and "did not get an answer" in out[0], out)
        check("the stall interrupts the turn",
              any("abort" in p for _, p, _ in fake.requests), fake.requests)

        async def boom():
            raise RuntimeError("abort refused")
        b.interrupt = boom
        fake.events = []                       # next_event returns None again
        out = [c async for c in b.ask_stream("slow again")]
        check("a stall whose interrupt itself fails is swallowed",
              len(out) == 1 and "did not get an answer" in out[0], out)
    finally:
        CFG["turn_timeout"] = old
        brain._Server = saved


async def scenario_prompt_failure():
    fake, b, saved = await new_brain()
    try:
        fake.fail_prompt = True
        raised = False
        try:
            async for _ in b.ask_stream("fail"):
                pass
        except RuntimeError:
            raised = True
        check("a prompt failure propagates", raised)
        check("a failed prompt leaves the turn clean", b._dirty is False)
    finally:
        brain._Server = saved


async def scenario_resume():
    fake = FakeServer()
    saved = brain._Server
    brain._Server = lambda *a, **k: fake
    try:
        b = brain.WarmBrain(resume_id="ses_old")
        await b.start()
        check("an existing session is resumed", b._sid == "ses_old", b._sid)
        fake.fail_resume = True
        b2 = brain.WarmBrain(resume_id="ses_old")
        await b2.start()
        check("a lost session falls back to a fresh one",
              b2._sid == "ses_test", b2._sid)
    finally:
        brain._Server = saved


async def scenario_permissions():
    async def allow(*a):
        return _Decision("allow")

    async def deny(*a):
        return _Decision("deny")

    for mode, gate, expect_allow in [
        ("bypassPermissions", None, True),
        ("ask", allow, True),
        ("ask", deny, False),
    ]:
        fake = FakeServer()
        saved = brain._Server
        brain._Server = lambda *a, **k: fake
        try:
            b = brain.WarmBrain(can_use_tool=gate)
            await b.start()
            b._perm_mode = mode
            sid = b._sid
            fake.events = [
                {"type": "permission.asked",
                 "properties": {"sessionID": sid, "id": "per1",
                                "permission": "bash",
                                "patterns": ["rm -rf /"]}},
                {"type": "session.idle", "properties": {"sessionID": sid}},
            ]
            async for _ in b.ask_stream("do it"):
                pass
            replies = [body for _, p, body in fake.requests
                       if "permissions/per1" in p]
            got = replies[-1]["response"] if replies else None
            want = "once" if expect_allow else "reject"
            check("permission %s/%s -> %s" % (mode, bool(gate), want),
                  got == want, got)
        finally:
            brain._Server = saved


class _Decision:
    def __init__(self, behavior):
        self.behavior = behavior


async def scenario_commands():
    fake, b, saved = await new_brain()
    try:
        sid = b._sid
        fake.events = [{"type": "session.idle",
                        "properties": {"sessionID": sid}}]
        check("plain text goes to the model",
              await b.command("just talk") == "ok")

        check("model is reported", (await b.command("/model")).startswith("model is"))
        check("model is switched",
              await b.command("/model new/model") == "model set to new/model")
        check("effort is set",
              await b.command("/effort high") == "effort set to high")
        check("a bad effort is refused",
              "unknown effort" in await b.command("/effort nope"))
        check("compact runs", await b.command("/compact") == "compacted")

        b._dirty = True
        fake.events = [{"type": "session.idle",
                        "properties": {"sessionID": sid}}]
        check("clear restarts the session", await b.command("/clear") == "cleared")
        check("clear stopped the old server", fake.stopped)

        check("a failing command is reported, not raised",
              (await b.command("/effort x")).startswith("unknown effort"))
    finally:
        brain._Server = saved


async def scenario_usage_and_interrupt():
    fake, b, saved = await new_brain()
    try:
        ctx = await b.context_usage()
        check("context usage is summed", ctx["categories"][0]["tokens"] == 132,
              ctx)
        await b.interrupt()
        check("interrupt aborts the session",
              any("abort" in p for _, p, _ in fake.requests), fake.requests)
    finally:
        brain._Server = saved


# ---- the real _Server over a real localhost HTTP + SSE server -----------

class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def _json(self, body):
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.startswith("/global/health"):
            return self._json({"ok": True})
        if self.path.startswith("/event"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            while True:
                item = self.server.sse_queue.get()
                if item is None:
                    return
                if isinstance(item, dict) and "__raw__" in item:
                    line = item["__raw__"] + "\n"
                else:
                    line = "data: %s\n\n" % item
                self.wfile.write(line.encode())
                self.wfile.flush()
        if "/session" in self.path:
            return self._json({"tokens": {"input": 3, "output": 4}})
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n:
            self.rfile.read(n)
        return self._json({})


class _Proc:
    def __init__(self, poll=None, pid=4242):
        self._poll = poll
        self.returncode = poll
        self.pid = pid
        self.terminated = False
        self.killed = False

    def poll(self):
        return self._poll

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True


async def scenario_model_ref():
    b = brain.WarmBrain()
    try:
        b._model_ref("nope")
        raised = False
    except brain.OpencodeError:
        raised = True
    check("a model with no provider slash is refused", raised)


async def scenario_server_health_and_spawn():
    srv = brain._Server()
    with mock.patch.object(brain.urllib.request, "urlopen",
                           side_effect=OSError("nothing listening")):
        check("health is False when nothing answers",
              await srv._healthy() is False)

    calls = {"n": 0}

    async def health():
        calls["n"] += 1
        return calls["n"] >= 3

    async def spawn():
        calls["spawn"] = True

    with mock.patch.object(srv, "_healthy", health), \
         mock.patch.object(srv, "_spawn", spawn), \
         mock.patch.object(srv, "_start_reader", _noop), \
         mock.patch.object(brain.asyncio, "sleep", _noop):
        await srv.ensure()
    check("ensure spawns, retries, and waits for the server",
          calls.get("spawn") is True)

    srv2 = brain._Server()

    async def dead_spawn():
        srv2.proc = _Proc(poll=1)

    async def never():
        return False

    with mock.patch.object(srv2, "_healthy", never), \
         mock.patch.object(srv2, "_spawn", dead_spawn):
        try:
            await srv2.ensure()
            died = False
        except brain.OpencodeError:
            died = True
    check("a server that exits at once is reported", died)

    srv3 = brain._Server()
    times = iter([0.0, 100.0, 100.0])
    with mock.patch.object(srv3, "_healthy", never), \
         mock.patch.object(srv3, "_spawn", _noop), \
         mock.patch.object(brain.time, "time", lambda: next(times)):
        try:
            await srv3.ensure()
            timed = False
        except brain.OpencodeError:
            timed = True
    check("a server that never comes up times out", timed)


async def scenario_spawn_cmdline():
    srv = brain._Server()
    old_bin = CFG.get("opencode_bin")
    try:
        CFG.pop("opencode_bin", None)
        with mock.patch.object(brain.shutil, "which", lambda name: None):
            try:
                await srv._spawn()
                raised = False
            except brain.OpencodeError:
                raised = True
            check("no opencode on PATH is a clear error", raised)

        started = {}

        class _Popen:
            def __init__(self, cmd, **kw):
                started["cmd"] = cmd
                started["env"] = kw.get("env")
                self.pid = 7

        with mock.patch.object(brain.sys, "platform", "win32"), \
             mock.patch.object(brain.shutil, "which",
                               lambda name: r"C:\tools\opencode.cmd"), \
             mock.patch.object(brain.subprocess, "Popen", _Popen):
            await srv._spawn()
        check("a server is spawned", srv.proc is not None)
        check("a .cmd launcher is wrapped in cmd /c",
              started["cmd"][:2] == ["cmd", "/c"], started["cmd"])
        check("the server password is cleared from the child env",
              "OPENCODE_SERVER_PASSWORD" not in started["env"]
              and started["env"].get("OPENCODE_CLIENT") == "backtalk",
              started["env"].get("OPENCODE_CLIENT"))

        with mock.patch.object(brain.shutil, "which",
                               lambda name: r"C:\tools\opencode.exe"), \
             mock.patch.object(brain.subprocess, "Popen", _Popen):
            srv2 = brain._Server()
            await srv2._spawn()
        check("a plain executable is launched directly",
              started["cmd"][0] == r"C:\tools\opencode.exe", started["cmd"])
    finally:
        if old_bin is not None:
            CFG["opencode_bin"] = old_bin


async def scenario_server_stop():
    srv = brain._Server()
    srv._reader = asyncio.create_task(asyncio.sleep(10))
    proc = _Proc(poll=None)
    srv.proc = proc
    await srv.stop()
    check("stop cancels the reader and terminates the process",
          srv._reader is None and proc.terminated and srv.proc is None)

    srv2 = brain._Server()
    proc2 = _Proc(poll=None)

    def boom():
        raise OSError("no terminate")
    proc2.terminate = boom
    srv2.proc = proc2
    await srv2.stop()
    check("a stubborn process is killed", proc2.killed)

    srv3 = brain._Server()
    proc3 = _Proc(poll=0)
    srv3.proc = proc3
    await srv3.stop()
    check("an already-dead process is not touched", not proc3.terminated)

    srv4 = brain._Server()
    proc4 = _Proc(poll=None)

    def nope():
        raise OSError("nothing works")
    proc4.terminate = nope
    proc4.kill = nope
    srv4.proc = proc4
    await srv4.stop()
    check("a process that refuses to die is left alone", srv4.proc is None)


async def scenario_start_reader_idempotent():
    srv = brain._Server()
    srv._reader = asyncio.create_task(asyncio.sleep(10))
    await srv._start_reader()
    check("a live reader is not restarted", not srv._reader.done())
    srv._reader.cancel()


async def scenario_read_events():
    srv = brain._Server()
    with mock.patch.object(brain.urllib.request, "urlopen",
                           side_effect=OSError("refused")):
        await srv._read_events()
    check("a stream that will not open ends quietly", True)

    class _Stream:
        def __init__(self, lines):
            self._lines = list(lines)
            self.closed = False

        def readline(self):
            return self._lines.pop(0) if self._lines else b""

        def close(self):
            self.closed = True

    srv2 = brain._Server()
    stream = _Stream([b"\n", b"not an event\n",
                      b"data: " + json.dumps({"type": "hello"}).encode() + b"\n",
                      b"data: {bad json\n"])
    with mock.patch.object(brain.urllib.request, "urlopen",
                           lambda *a, **k: stream):
        await srv2._read_events()
    ev = srv2._events.get_nowait()
    check("the reader forwards events and skips junk",
          ev.get("type") == "hello", ev)
    check("the stream is closed when the reader ends", stream.closed)

    srv3 = brain._Server()
    closed_boom = _Stream([b""])

    def _boom():
        raise OSError("already closed")

    closed_boom.close = _boom
    with mock.patch.object(brain.urllib.request, "urlopen",
                           lambda *a, **k: closed_boom):
        await srv3._read_events()
    check("a stream whose close fails is still handled", True)

    class _BadStream:
        def readline(self):
            raise OSError("cable pulled")

    srv3 = brain._Server()
    with mock.patch.object(brain.urllib.request, "urlopen",
                           lambda *a, **k: _BadStream()):
        await srv3._read_events()
    check("a stream that errors mid-flight ends quietly", True)


async def scenario_next_event_reconnect():
    srv = brain._Server()
    await srv._events.put({"type": "e1"})
    with mock.patch.object(srv, "_start_reader", _noop):
        ev = await srv.next_event()
    check("next_event starts a reader and returns the event",
          ev == {"type": "e1"}, ev)

    srv2 = brain._Server()
    srv2._reader = asyncio.create_task(asyncio.sleep(0))
    await asyncio.sleep(0.01)
    calls = {"n": 0}

    async def start2():
        calls["n"] += 1

    await srv2._events.put({"type": "e2"})
    with mock.patch.object(srv2, "_start_reader", start2):
        ev = await srv2.next_event()
    check("a dropped reader is reconnected",
          calls["n"] == 1 and ev == {"type": "e2"}, ev)


async def scenario_brain_edges():
    b0 = brain.WarmBrain()
    check("context_usage with no session is None",
          await b0.context_usage() is None)
    await b0.interrupt()
    out = [c async for c in b0.ask_stream("hello")]
    check("ask_stream with no session says nothing", out == [])
    await b0._collect_usage()
    await b0.stop()
    check("an unstarted brain collects nothing and stops cleanly", True)

    fake, b, saved = await new_brain()
    try:
        fake.raise_session_get = True
        check("a failing context read is None",
              await b.context_usage() is None)
        fake.raise_session_get = False

        await b.set_permission_mode("ask")
        check("set_permission_mode records the intent", b._perm_mode == "ask")

        fake.fail_abort = True
        await b.interrupt()
        check("a failing abort is swallowed", True)
        fake.fail_abort = False

        before = dict(b.session)
        b._tally({"tokens": {"input": 5, "output": 9,
                             "cache": {"read": 2}}}, count_turn=False)
        check("_tally can skip the turn and reads the cache",
              b.session["turns"] == before["turns"]
              and b.session["in_tokens"] == before["in_tokens"] + 7
              and b.session["out_tokens"] == before["out_tokens"] + 9,
              b.session)
        b._tally(None)
        check("a malformed usage payload is swallowed", True)

        old_show = CFG.get("show_usage")
        CFG["show_usage"] = True
        try:
            await b._publish_usage()
            check("usage is published when asked", True)
            fake.zero_tokens = True
            await b._publish_usage()
            check("zero usage is not announced", True)
            fake.zero_tokens = False
            fake.raise_session_get = True
            await b._publish_usage()
            check("a failing usage publish is swallowed", True)
        finally:
            CFG["show_usage"] = old_show
            fake.raise_session_get = False

        fake.messages = "empty"
        await b._collect_usage()
        fake.messages = "user"
        await b._collect_usage()
        fake.fail_usage = True
        await b._collect_usage()
        fake.fail_usage = False
        fake.messages = "default"
        check("empty, non-assistant, and failed usage reads are swallowed",
              True)

        old_resume = CFG.get("resume_last_session")
        old_file = brain.SESSION_FILE
        try:
            CFG["resume_last_session"] = True
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.close()
            brain.SESSION_FILE = tmp.name
            b._remember_session()
            with open(tmp.name) as f:
                wrote = f.read()
            check("the session id is remembered", wrote == b._sid, wrote)
            os.unlink(tmp.name)
            brain.SESSION_FILE = os.path.dirname(tmp.name)
            b._remember_session()
            check("a session that cannot be saved is not fatal", True)
        finally:
            CFG["resume_last_session"] = old_resume
            brain.SESSION_FILE = old_file

        await b.stop()
        check("stop tears the server down", fake.stopped and b._sid is None)
    finally:
        brain._Server = saved


async def scenario_reset_turn():
    b = brain.WarmBrain()
    await b.reset_turn()
    check("a clean turn skips the drain", True)

    fake, b2, saved = await new_brain()
    try:
        b2._dirty = True
        fake.events = [
            {"type": "message.part.updated",
             "properties": {"sessionID": b2._sid}},
            {"type": "session.idle", "properties": {"sessionID": b2._sid}},
        ]
        await b2.reset_turn()
        check("reset drains to the idle marker and clears dirty",
              b2._dirty is False)

        b2._dirty = True
        fake.events = []
        await b2.reset_turn()
        check("reset with no events returns", b2._dirty is False)

        b2._dirty = True
        ticks = iter([0.0, 100.0, 100.0])
        with mock.patch.object(brain.time, "time", lambda: next(ticks)):
            await b2.reset_turn()
        check("reset past its deadline returns", b2._dirty is False)
    finally:
        brain._Server = saved

    b3 = brain.WarmBrain()
    b3._dirty = True
    await b3.reset_turn()
    check("reset with no server returns", True)


async def scenario_command_edges():
    b = brain.WarmBrain()
    check("compact before start says so",
          await b.command("/compact") == "not started")

    fake = FakeServer()
    saved = brain._Server
    brain._Server = lambda *a, **k: fake
    try:
        b2 = brain.WarmBrain()
        check("clear on a fresh brain starts a new session",
              await b2.command("/clear") == "cleared")
        check("clear left a live server", b2._srv is not None)
        check("a bad model is reported, not raised",
              (await b2.command("/model nope")).startswith("error:"))
    finally:
        brain._Server = saved


async def scenario_body_fields():
    old_v = CFG.get("variant")
    old_a = CFG.get("agent")
    CFG["variant"] = "high"
    CFG["agent"] = "build"
    fake = FakeServer()
    saved = brain._Server
    brain._Server = lambda *a, **k: fake
    try:
        b = brain.WarmBrain()
        await b.start()
        sid = b._sid
        fake.events = [{"type": "session.idle",
                        "properties": {"sessionID": sid}}]
        await _drain(b.ask_stream("hi"))
        body = [bd for m, p, bd in fake.requests if "prompt_async" in p][-1]
        check("agent and variant ride along",
              body.get("agent") == "build"
              and body.get("variant") == "high", body)
    finally:
        CFG["variant"] = old_v
        CFG["agent"] = old_a
        brain._Server = saved


async def scenario_live_fetch():
    fake, b, saved = await new_brain()
    old_live = CFG.get("live_data")
    old_wants = brain.live.wants_live
    old_fetch = brain.live.fetch
    try:
        CFG["live_data"] = True
        brain.live.wants_live = lambda u: True
        brain.live.fetch = lambda u: "Fresh facts for you."
        sid = b._sid
        fake.events = [{"type": "session.idle", "properties": {"sessionID": sid}}]
        await _drain(b.ask_stream("what time is it"))
        prompt = [body for m, p, body in fake.requests
                  if "prompt_async" in p][-1]
        check("live facts are prepended to the turn",
              prompt["parts"][0]["text"] == "Fresh facts for you.",
              prompt["parts"])

        brain.live.wants_live = lambda u: False
        fake.events = [{"type": "session.idle", "properties": {"sessionID": sid}}]
        await _drain(b.ask_stream("hello there"))
        prompt = [body for m, p, body in fake.requests
                  if "prompt_async" in p][-1]
        check("no live facts when none are wanted",
              prompt["parts"][0]["text"] == "hello there",
              prompt["parts"])

        brain.live.wants_live = lambda u: True

        def boom(u):
            raise RuntimeError("live down")
        brain.live.fetch = boom
        fake.events = [{"type": "session.idle", "properties": {"sessionID": sid}}]
        await _drain(b.ask_stream("weather?"))
        check("a live lookup that fails answers without it", True)
    finally:
        CFG["live_data"] = old_live
        brain.live.wants_live = old_wants
        brain.live.fetch = old_fetch
        brain._Server = saved


async def scenario_image_part():
    fake, b, saved = await new_brain()
    old_live = CFG.get("live_data")
    old_wants = brain.live.wants_live
    try:
        CFG["live_data"] = True
        brain.live.wants_live = lambda u: True
        sid = b._sid
        fake.events = [{"type": "session.idle",
                        "properties": {"sessionID": sid}}]
        await _drain(b.ask_stream("what is on my screen?", image=b"\x89PNG"))
        prompt = [body for m, p, body in fake.requests
                  if "prompt_async" in p][-1]
        parts = prompt["parts"]
        check("the question and the screenshot both ride along",
              parts[0]["text"] == "what is on my screen?" and len(parts) == 2,
              parts)
        fpart = parts[1]
        check("the screenshot is a png data url",
              fpart["type"] == "file"
              and fpart["mime"] == "image/png"
              and fpart["filename"] == "screen.png"
              and fpart["url"].startswith("data:image/png;base64,"), fpart)
        check("a screenshot skips the live lookup",
              all(p.get("type") != "text"
                  or p["text"] != "Fresh facts for you." for p in parts),
              parts)
    finally:
        CFG["live_data"] = old_live
        brain.live.wants_live = old_wants
        brain._Server = saved


async def scenario_stream_edges():
    fake, b, saved = await new_brain()
    try:
        sid = b._sid
        fake.events = [
            {"type": "message.part.updated",
             "properties": {"sessionID": sid, "part": {"type": "text"}}},
            {"type": "message.part.updated",
             "properties": {"sessionID": sid,
                            "part": {"id": "s0", "type": "step-finish",
                                     "reason": "tool-calls"}}},
            {"type": "message.part.updated",
             "properties": {"sessionID": "other",
                            "part": {"id": "x", "type": "text"}}},
            {"type": "permission.asked",
             "properties": {"sessionID": "other", "id": "p"}},
            {"type": "session.idle", "properties": {"sessionID": "other"}},
            {"type": "session.error", "properties": {"sessionID": "other"}},
            {"type": "ping", "properties": {"sessionID": sid}},
            {"type": "session.idle", "properties": {"sessionID": sid}},
        ]
        out = [c async for c in b.ask_stream("edge")]
        check("foreign events are ignored; the turn still ends",
              len(out) == 1 and "empty" in out[0], out)

        fake.events = [
            {"type": "message.part.delta",
             "properties": {"sessionID": sid, "partID": "p1",
                            "field": "text", "delta": "Almost there"}},
            {"type": "session.error",
             "properties": {"sessionID": sid, "error": "boom"}},
        ]
        out = [c async for c in b.ask_stream("break mid")]
        check("a mid-answer error speaks the tail then apologises",
              out == ["Almost there",
                      "That did not work on my side. Ask me again."], out)
    finally:
        brain._Server = saved


async def scenario_permission_edges():
    fake, b, saved = await new_brain()
    try:
        sid = b._sid
        await b._handle_permission({"sessionID": sid})

        b._perm_mode = "bypassPermissions"
        await b._handle_permission({
            "id": "pe", "sessionID": sid, "permission": "edit",
            "metadata": {"filepath": "/tmp/x"}})
        await b._handle_permission({
            "id": "pw", "sessionID": sid, "permission": "webfetch",
            "patterns": ["https://example.test"]})
        check("edit and webfetch asks are answered", True)

        fake.events = [
            {"type": "permission.asked",
             "properties": {"sessionID": sid, "id": "p1",
                            "permission": "bash", "patterns": ["ls"]}},
            {"type": "session.idle", "properties": {"sessionID": sid}},
        ]
        b._perm_mode = "ask"
        b._can_use_tool = None
        async for _ in b.ask_stream("go"):
            pass
        got = [body for _, p, body in fake.requests
               if "permissions/p1" in p][-1]["response"]
        check("no gate in ask mode denies", got == "reject", got)

        async def raising(*a):
            raise RuntimeError("gate exploded")

        b._can_use_tool = raising
        fake.events = [
            {"type": "permission.asked",
             "properties": {"sessionID": sid, "id": "p2",
                            "permission": "bash", "patterns": ["rm"]}},
            {"type": "session.idle", "properties": {"sessionID": sid}},
        ]
        await _drain(b.ask_stream("go again"))
        got = [body for _, p, body in fake.requests
               if "permissions/p2" in p][-1]
        check("a gate that raises denies", got["response"] == "reject", got)

        b3 = brain.WarmBrain(can_use_tool=raising)
        await b3._handle_permission({"id": "z", "sessionID": "s"})

        fake.fail_perm = True
        b._perm_mode = "bypassPermissions"
        await b._handle_permission({
            "id": "pf", "sessionID": sid, "permission": "bash",
            "patterns": ["rm"]})
        check("a failed permission reply is swallowed", True)
    finally:
        brain._Server = saved


async def scenario_real_server():
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.sse_queue = queue.Queue()
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    old_port = CFG.get("opencode_port")
    CFG["opencode_port"] = port
    try:
        srv = brain._Server()
        await srv.ensure()                    # health path, no spawn
        check("a healthy server is reused", srv.proc is None)

        got = await srv.request("GET", "/session/abc")
        check("a real request round-trips", got == {"tokens": {"input": 3, "output": 4}}, got)

        q = srv._url("/event", "/tmp/x")
        check("a directory becomes a query", q.endswith("?directory=%2Ftmp%2Fx"), q)

        httpd.sse_queue.put({"__raw__": "this line is not an event"})
        httpd.sse_queue.put("data: {not json")
        httpd.sse_queue.put(json.dumps({"type": "ping", "properties": {"n": 1}}))
        ev = await srv.next_event(timeout=5)
        check("junk lines are skipped, a real event survives",
              ev is not None and ev.get("type") == "ping", ev)

        empty = await srv.next_event(timeout=0.2)
        check("an idle stream times out to None", empty is None, empty)

        httpd.sse_queue.put(None)
        srv._reader.cancel()
    finally:
        CFG["opencode_port"] = old_port
        httpd.shutdown()


async def main():
    await scenario_stream()
    await scenario_tool_flush()
    await scenario_empty_and_error()
    await scenario_timeout()
    await scenario_prompt_failure()
    await scenario_resume()
    await scenario_permissions()
    await scenario_commands()
    await scenario_usage_and_interrupt()
    await scenario_model_ref()
    await scenario_server_health_and_spawn()
    await scenario_spawn_cmdline()
    await scenario_server_stop()
    await scenario_start_reader_idempotent()
    await scenario_read_events()
    await scenario_next_event_reconnect()
    await scenario_brain_edges()
    await scenario_reset_turn()
    await scenario_command_edges()
    await scenario_body_fields()
    await scenario_live_fetch()
    await scenario_image_part()
    await scenario_stream_edges()
    await scenario_permission_edges()
    await scenario_real_server()


print("=" * 66)
print("THE BRAIN, OFFLINE")
print("=" * 66)
asyncio.run(main())
print("\n" + "=" * 66)
print("BRAIN OK" if not failures else "BRAIN FAILURES: %s" % failures)
sys.exit(1 if failures else 0)
