# backtalk: talk to your opencode agent out loud.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The warm brain — a persistent opencode session, streaming.

One opencode session lives for the whole voice session: no per-turn process
spawn, no per-turn context reload. opencode streams `message.part.delta`
events as the model writes, so sentences are yielded the moment they're
complete and the mouth starts speaking while the rest of the thought is
still forming.

The session's directory is YOUR agent's folder (agent_dir in
backtalk.json) — whatever AGENTS.md lives there defines who is speaking.
backtalk adds only the spoken-delivery discipline (config.DISCIPLINE): the
medium, never the character.

Transport: opencode's local HTTP server (`opencode serve`). The server is
started on demand if it isn't already running, and reused if it is.
"""
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from typing import Any

from backtalk import live, signals
from backtalk.config import CFG, DISCIPLINE, refresh_clock
from backtalk.vlog import log

_SENTENCE_END = re.compile(r"(?<=[.!?])\s")

# One bounded pool for every blocking call the brain makes into the stdlib
# (HTTP requests, the SSE socket, live-data fetches). The default executor
# is per-process and shared with anything else that calls
# run_in_executor(_IO, ...), so a slow or stuck call could quietly eat
# threads the rest of the process needs. 8 matches main.py's _BLOCKERS.
_IO = ThreadPoolExecutor(max_workers=8, thread_name_prefix="brain-io")


SESSION_FILE = os.path.join(CFG["signals_dir"], ".backtalk_session")


class OpencodeError(RuntimeError):
    pass


class _Server:
    """The opencode HTTP server: start it if needed, talk to it over
    JSON + SSE. One process per voice session, shared by every request."""

    def __init__(self, port: int | None = None, host: str = "127.0.0.1"):
        self.port = int(port or CFG.get("opencode_port") or 4599)
        self.host = host
        self.base = f"http://{host}:{self.port}"
        self.proc: subprocess.Popen | None = None
        self._events: "asyncio.Queue[dict]" = asyncio.Queue()
        self._reader: asyncio.Task | None = None
        self._replies: dict[str, asyncio.Future] = {}

    # ---- lifecycle -----------------------------------------------------
    async def ensure(self):
        if await self._healthy():
            log(f"[brain] reusing opencode server on {self.base}")
            # The reader is started on BOTH paths. opencode's event stream
            # is per-connection, not per-process: a server that was already
            # running still owes this client its own stream, and without one
            # every turn would block forever waiting for an idle that can
            # never arrive.
            await self._start_reader()
            return
        await self._spawn()
        deadline = time.time() + 90
        while time.time() < deadline:
            if await self._healthy():
                log(f"[brain] opencode server up on {self.base}")
                await self._start_reader()
                return
            if self.proc and self.proc.poll() is not None:
                raise OpencodeError(
                    f"opencode serve exited immediately (code "
                    f"{self.proc.returncode}). Is opencode installed and on "
                    f"PATH?")
            await asyncio.sleep(1.0)
        raise OpencodeError(
            f"opencode server did not come up on {self.base} within 90s")

    async def _healthy(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base}/global/health")
            with urllib.request.urlopen(req, timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    async def _spawn(self):
        exe = CFG.get("opencode_bin") or shutil.which("opencode") \
            or shutil.which("opencode.cmd") or shutil.which("opencode.exe")
        if not exe:
            raise OpencodeError(
                "opencode was not found on PATH. Install it with "
                "`npm install -g opencode-ai`, or set opencode_bin in "
                "backtalk.json.")
        env = dict(os.environ)
        # A server password in the environment would make every request
        # 401. backtalk starts its OWN server and speaks to it over
        # loopback only, so the password is cleared deliberately.
        for k in ("OPENCODE_SERVER_PASSWORD", "OPENCODE_SERVER_USERNAME"):
            env.pop(k, None)
        env["OPENCODE_CLIENT"] = "backtalk"
        cmd = [exe, "serve", "--port", str(self.port),
               "--hostname", self.host]
        if sys.platform == "win32" and exe.lower().endswith(".cmd"):
            cmd = ["cmd", "/c", exe, "serve", "--port", str(self.port),
                   "--hostname", self.host]
        self.proc = subprocess.Popen(
            cmd, env=env, cwd=CFG["agent_dir"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if sys.platform == "win32" else 0)
        log(f"[brain] started opencode server (pid {self.proc.pid})")

    async def stop(self):
        if self._reader:
            self._reader.cancel()
            self._reader = None
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=10)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None

    # ---- HTTP ----------------------------------------------------------
    def _url(self, path: str, directory: str | None) -> str:
        q = {}
        if directory:
            q["directory"] = directory
        url = f"{self.base}{path}"
        return f"{url}?{urllib.parse.urlencode(q)}" if q else url

    async def request(self, method: str, path: str, body=None,
                      directory: str | None = None, timeout: int = 300) -> Any:
        def call():
            data = json.dumps(body).encode() if body is not None else None
            req = urllib.request.Request(
                self._url(path, directory), data=data, method=method,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                return json.loads(raw) if raw else None
        return await asyncio.get_running_loop().run_in_executor(_IO, call)

    # ---- SSE event stream ---------------------------------------------
    async def _start_reader(self):
        if self._reader and not self._reader.done():
            return
        self._reader = asyncio.create_task(self._read_events())
        # Give the connection a moment to open, so a server that is up but
        # still refusing streams fails here (where the error is visible)
        # rather than as a mysterious stall on the first turn.
        await asyncio.sleep(1.0)

    async def _read_events(self):
        directory = CFG["agent_dir"]
        url = self._url("/event", directory)
        loop = asyncio.get_running_loop()

        def open_stream():
            return urllib.request.urlopen(url, timeout=600)

        try:
            stream = await loop.run_in_executor(_IO, open_stream)
        except Exception as e:
            log(f"[brain] event stream failed to open: {e!r}")
            return
        try:
            while True:
                line = await loop.run_in_executor(_IO, stream.readline)
                if not line:
                    break
                text = line.decode("utf-8", "replace").strip()
                if not text.startswith("data:"):
                    continue
                try:
                    ev = json.loads(text[5:].strip())
                except Exception:
                    continue
                await self._events.put(ev)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log(f"[brain] event stream ended: {e!r}")
        finally:
            # Close the connection on every exit, cancel included: without
            # this a stop/clear leaks a socket that holds an _IO worker for
            # the life of its 600s read timeout, and enough of those starve
            # the pool the readers share.
            try:
                stream.close()
            except Exception:
                pass

    async def next_event(self, timeout: float | None = None) -> dict | None:
        # A dropped connection is silent: the reader task ends and the queue
        # simply stops filling, which from the caller's side looks exactly
        # like a model that is thinking forever. So liveness is checked on
        # every wait, and a dead reader is reconnected rather than waited on.
        if self._reader is None or self._reader.done():
            if self._reader is not None:
                log("[brain] event stream dropped, reconnecting")
            await self._start_reader()
        if timeout is None:
            return await self._events.get()
        try:
            return await asyncio.wait_for(self._events.get(), timeout)
        except asyncio.TimeoutError:
            return None


class WarmBrain:
    """Same surface the voice loop already uses, backed by opencode."""

    def __init__(self, model: str | None = None, can_use_tool=None,
                 resume_id: str | None = None):
        self.model = model or CFG["model"]
        self.deep_model = CFG.get("deep_model") or self.model
        self._can_use_tool = can_use_tool
        self._resume_id = resume_id
        self.session = {"turns": 0, "out_tokens": 0, "in_tokens": 0,
                        "cost": 0.0}
        self._srv: _Server | None = None
        self._sid: str | None = None
        self._agent = CFG.get("agent") or "build"
        self._variant: str | None = CFG.get("variant") or None
        self._dirty = False
        self._perm_mode = CFG["permission_mode"]
        self._message_id: str | None = None

    # ---- helpers -------------------------------------------------------
    @property
    def _dir(self) -> str:
        return CFG["agent_dir"]

    def _model_ref(self, model: str | None = None):
        """'provider/model-id' -> ({providerID, modelID}, full string).

        opencode's model ids are namespaced by the provider AND often carry
        their own prefix (nvidia/nemotron-... under provider nvidia), so the
        split is on the FIRST slash only and the remainder is the model id
        exactly as opencode lists it."""
        ref = (model or self.model).strip()
        if "/" not in ref:
            raise OpencodeError(
                f"model {ref!r} is not in provider/model form "
                f"(for example nvidia/nvidia/nemotron-3-super-120b-a12b)")
        provider, model_id = ref.split("/", 1)
        return {"providerID": provider, "modelID": model_id}, ref

    def _system(self) -> str:
        """The spoken-delivery discipline. opencode has no 'preset system
        prompt' concept, so it goes in as an explicit system string on
        every turn — the AGENTS.md in agent_dir is loaded on top of it.

        The clock is refreshed here, per turn, because DISCIPLINE bakes in
        the time at import. A process left running past midnight otherwise
        greets the next morning as yesterday, and the model's only stated
        source of truth for the date is the one string it was given.
        """
        return refresh_clock(DISCIPLINE)

    # ---- lifecycle -----------------------------------------------------
    async def start(self):
        self._srv = _Server()
        await self._srv.ensure()
        if self._resume_id:
            try:
                await self._srv.request(
                    "GET", f"/session/{self._resume_id}", directory=self._dir,
                    timeout=15)
                self._sid = self._resume_id
                log(f"[brain] resumed session {self._sid[:8]}")
                return
            except Exception as e:
                log(f"[brain] resume failed ({str(e)[:80]}), starting fresh")
        sess = await self._srv.request(
            "POST", "/session", {"title": f"voice - {CFG['name']}"},
            directory=self._dir, timeout=30)
        self._sid = sess["id"]
        log(f"[brain] new session {str(self._sid)[:8]}")

    async def stop(self):
        if self._srv:
            await self._srv.stop()
            self._srv = None
        self._sid = None

    async def set_permission_mode(self, backtalk_mode: str):
        """Live flip. opencode decides per-call from the server's config,
        so this records the intent and the gate honours it from here on."""
        self._perm_mode = backtalk_mode
        log(f"[brain] permission mode -> {backtalk_mode}")

    async def context_usage(self):
        if not self._srv or not self._sid:
            return None
        try:
            info = await self._srv.request(
                "GET", f"/session/{self._sid}", directory=self._dir,
                timeout=15)
        except Exception:
            return None
        t = info.get("tokens") or {}
        total = int(t.get("input") or 0) + int(t.get("output") or 0) \
            + int(t.get("reasoning") or 0) \
            + int((t.get("cache") or {}).get("read") or 0)
        return {"categories": [{"name": "context", "tokens": total}]}

    # ---- usage ---------------------------------------------------------
    async def _publish_usage(self):
        """opencode has no subscription rate-limit window to draw, so the
        face gets the session's own numbers instead. Best effort."""
        if not CFG.get("show_usage") or not self._srv or not self._sid:
            return
        try:
            info = await self._srv.request(
                "GET", f"/session/{self._sid}", directory=self._dir,
                timeout=10)
            t = info.get("tokens") or {}
            used = int(t.get("input") or 0) + int(t.get("output") or 0)
            signals.set_rate_limit("five_hour", None, None)
            signals.set_rate_limit("seven_day", None, None)
            if used:
                signals.set_rate_limit("session", None, None)
        except Exception:
            pass

    def _tally(self, info, count_turn=True):
        try:
            s = self.session
            if count_turn:
                s["turns"] += 1
            t = info.get("tokens") or {}
            s["out_tokens"] += int(t.get("output") or 0)
            s["in_tokens"] += int(t.get("input") or 0) \
                + int((t.get("cache") or {}).get("read") or 0)
            s["cost"] += float(info.get("cost") or 0.0)
        except Exception:
            pass

    def _remember_session(self):
        if not CFG.get("resume_last_session") or not self._sid:
            return
        try:
            with open(SESSION_FILE, "w") as f:
                f.write(self._sid)
        except OSError:
            pass

    # ---- turn control --------------------------------------------------
    async def interrupt(self):
        if self._srv and self._sid:
            try:
                await self._srv.request(
                    "POST", f"/session/{self._sid}/abort", {},
                    directory=self._dir, timeout=15)
            except Exception:
                pass

    async def reset_turn(self, timeout: float = 8.0):
        """Drop a cancelled turn's leftovers so the next question cannot
        answer the previous one. opencode streams per-session events and
        tags every message, so the drain is a message-id filter rather
        than a shared-pipe resync — but a turn that died mid-flight can
        still leave an unconsumed message, so the pipe is drained to the
        next idle marker either way."""
        if not self._dirty:
            return
        self._dirty = False
        await self.interrupt()
        deadline = time.time() + timeout
        srv = self._srv
        if srv is None:
            return
        while time.time() < deadline:
            ev = await srv.next_event(timeout=max(0.1, deadline - time.time()))
            if ev is None:
                return
            if ev.get("type") in ("session.idle", "session.error"):
                return

    async def command(self, cmd: str) -> str:
        """The voice console's session verbs, mapped onto opencode."""
        self._dirty = True
        text = (cmd or "").strip()
        try:
            if text.startswith("/clear"):
                await self.reset_turn()
                if self._srv:
                    await self._srv.stop()
                    self._srv = None
                await self.start()
                return "cleared"
            if text.startswith("/compact"):
                if self._srv is None:
                    return "not started"
                model, _ = self._model_ref()
                await self._srv.request(
                    "POST", f"/session/{self._sid}/summarize", {"model": model},
                    directory=self._dir, timeout=180)
                return "compacted"
            if text.startswith("/model"):
                rest = text[len("/model"):].strip()
                if rest and rest != self.model:
                    self.model = rest
                    self._model_ref()
                    return f"model set to {rest}"
                return f"model is {self.model}"
            if text.startswith("/effort"):
                rest = text[len("/effort"):].strip().lower()
                mapping = {"low": "low", "medium": "medium", "high": "high",
                           "xhigh": "high", "max": "max"}
                if rest in mapping:
                    self._variant = mapping[rest]
                    return f"effort set to {rest}"
                return f"unknown effort {rest!r}"
            # Anything else goes to the model as a plain instruction.
            async for _ in self.ask_stream(text.lstrip("/")):
                pass
            return "ok"
        except Exception as e:
            log(f"[brain] command {cmd!r} failed: {e!r}")
            return f"error: {e}"
        finally:
            self._dirty = False

    # ---- the turn ------------------------------------------------------
    async def _live(self, utterance: str) -> str:
        """Live facts for questions memory cannot answer.

        The fetch is done here, in code, instead of being left to the
        model's judgement, because a model asked about the present will
        sometimes answer from training data and a wrong answer about
        today sounds exactly as confident as a right one. Doing it here
        also removes the slow path: left to itself the model delegates
        webfetch to subagents, which costs a minute of silence.
        """
        if not CFG.get("live_data", True):
            return ""
        loop = asyncio.get_running_loop()
        try:
            if not await loop.run_in_executor(_IO, live.wants_live, utterance):
                return ""
            return await loop.run_in_executor(_IO, live.fetch, utterance)
        except Exception as e:
            log(f"[live] lookup failed, answering without it: {e!r}")
            return ""

    async def ask_stream(self, utterance: str, image: bytes | None = None):
        """Yield complete sentences as opencode streams them.

        image, when given, is a PNG of the user's screen: it rides along
        as a file part so a vision model can answer about what is really
        on the display. A current screenshot is already the freshest fact
        there is, so the live-data lookup is skipped when one is sent.
        """
        srv, sid = self._srv, self._sid
        if not srv or not sid:
            log("[brain] ask_stream with no session")
            return
        model, _ = self._model_ref()
        body: dict[str, Any] = {
            "model": model,
            "system": self._system(),
            "parts": [{"type": "text", "text": utterance}],
        }
        if self._agent:  # pragma: no branch -- _agent defaults to "build", never falsy
            body["agent"] = self._agent
        if self._variant:
            body["variant"] = self._variant

        if image is not None:
            body["parts"].append({
                "type": "file",
                "mime": "image/png",
                "filename": "screen.png",
                "url": "data:image/png;base64,"
                       + base64.b64encode(image).decode("ascii"),
            })
        else:
            live = await self._live(utterance)
            if live:
                body["parts"].insert(0, {"type": "text", "text": live})
                log(f"[live] fetched fresh facts for {utterance[:48]!r}")

        self._dirty = True
        try:
            await srv.request("POST", f"/session/{sid}/prompt_async", body,
                              directory=self._dir, timeout=60)
        except Exception as e:
            log(f"[brain] prompt failed: {e!r}")
            self._dirty = False
            raise

        buf = ""
        # opencode streams a model's thinking as `reasoning` parts and its
        # answer as `text` parts, on the same event stream. Only the text
        # is spoken: a reasoning model that narrates its plan aloud would
        # otherwise have the mouth reading its own scratchpad.
        kinds: dict[str, str] = {}
        spoken = 0
        turn_limit = float(CFG.get("turn_timeout") or 150)
        while True:
            ev = await srv.next_event(timeout=turn_limit)
            if ev is None:
                log(f"[brain] no answer within {turn_limit:.0f}s, "
                    "abandoning the turn and resetting")
                self._dirty = False
                try:
                    await self.interrupt()
                except Exception:
                    pass
                yield ("I did not get an answer back in time. "
                       "Ask me again and I will try once more.")
                spoken += 1
                return
            kind = ev.get("type")
            props = ev.get("properties") or {}

            if kind == "message.part.delta" and props.get("field") == "text":
                if props.get("sessionID") != sid:
                    continue
                pid = str(props.get("partID") or "")
                # An unknown part id is a text part in practice: the
                # delta can beat its own `updated` event across the wire.
                if kinds.get(pid, "text") != "text":
                    continue
                buf += props.get("delta") or ""
                while True:
                    m = _SENTENCE_END.search(buf)
                    if not m:
                        break
                    sentence, buf = buf[:m.end()].strip(), buf[m.end():]
                    if sentence:  # pragma: no branch -- a match keeps its punctuation
                        yield sentence
                        spoken += 1

            elif kind == "message.part.updated":
                if props.get("sessionID") != sid:
                    continue
                part = props.get("part") or {}
                pid = part.get("id")
                if pid:
                    kinds[pid] = part.get("type") or "text"
                # A step that ends in tool calls is a speech boundary:
                # flush now, so "On it, let me grab that" doesn't sit
                # silent through the whole tool run and then play glued
                # to the answer.
                if part.get("type") == "step-finish" \
                        and part.get("reason") == "tool-calls":
                    tail = buf.strip()
                    buf = ""
                    kinds = {}
                    if tail:
                        yield tail
                        spoken += 1

            elif kind == "permission.asked":
                if props.get("sessionID") != sid:
                    continue
                await self._handle_permission(props)

            elif kind in ("session.idle",):
                if props.get("sessionID") != sid:
                    continue
                self._dirty = False
                break

            elif kind == "session.error":
                if props.get("sessionID") != sid:
                    continue
                self._dirty = False
                log(f"[brain] session error: {json.dumps(props)[:300]}")
                tail = buf.strip()
                if tail:
                    yield tail
                    spoken += 1
                yield "That did not work on my side. Ask me again."
                return

        tail = buf.strip()
        if tail:
            yield tail
            spoken += 1

        # A turn can end with nothing to say: the session goes idle (or the
        # provider drops the stream) before a single text delta arrives.
        # Staying quiet is the worst possible answer -- you cannot tell an
        # empty reply from a broken microphone, so you just wait forever.
        # Say something, and write down why, so the log explains the gap.
        if not spoken:
            log("[brain] turn produced no text at all -- the session went "
                f"idle empty (session {str(sid)[:8]})")
            yield ("I did not hear myself there -- that one came back "
                   "empty. Ask me again.")
            self._dirty = False

        await self._collect_usage()
        self._remember_session()

    async def _collect_usage(self):
        """opencode reports usage on the finished message, not on the
        event stream. One cheap read so 'usage report' has real numbers."""
        srv, sid = self._srv, self._sid
        if not srv or not sid:
            return
        try:
            msgs = await srv.request(
                "GET", f"/session/{sid}/message", directory=self._dir,
                timeout=20)
        except Exception:
            msgs = None
        if isinstance(msgs, list) and msgs:
            info = msgs[-1].get("info") or {}
            if info.get("role") == "assistant":
                self._tally(info)
        await self._publish_usage()

    async def _handle_permission(self, props):
        """Route opencode's permission ask through the spoken gate."""
        pid = props.get("id")
        sid = props.get("sessionID")
        if not pid or not sid:
            return
        permission = props.get("permission") or "unknown"
        meta = props.get("metadata") or {}
        tool_input = {}
        if permission == "edit":
            tool_input = {"file_path": meta.get("filepath")}
        elif permission == "bash":
            tool_input = {"command": (props.get("patterns") or [""])[0]}
        elif permission == "webfetch":
            tool_input = {"url": (props.get("patterns") or [""])[0]}

        # The opencode tool names differ from the SDK's; the gate speaks
        # the SDK's vocabulary, so map onto it here.
        sdk_name = {"edit": "Edit", "bash": "Bash",
                    "webfetch": "WebFetch"}.get(permission, permission)

        allow = False
        if self._perm_mode == "bypassPermissions":
            allow = True
        elif self._can_use_tool:
            class _Ctx:
                display_name = sdk_name
                description = ""
            try:
                res = await self._can_use_tool(sdk_name, tool_input, _Ctx)
                allow = getattr(res, "behavior", "deny") == "allow"
            except Exception as e:
                log(f"[brain] permission gate raised: {e!r}")
                allow = False

        srv = self._srv
        if srv is None:
            return
        try:
            await srv.request(
                "POST", f"/session/{sid}/permissions/{pid}",
                {"response": "once" if allow else "reject"},
                directory=self._dir, timeout=30)
            log(f"[brain] permission {permission} -> "
                f"{'allow' if allow else 'reject'}")
        except Exception as e:
            log(f"[brain] permission reply failed: {e!r}")


if __name__ == "__main__":
    async def demo():
        b = WarmBrain()
        await b.start()
        for prompt in ("Voice check: greet me in one sentence.",
                       "And what's two plus two, spoken like yourself?"):
            t0 = time.time()
            async for s in b.ask_stream(prompt):
                print(f"  ({time.time()-t0:4.1f}s) {s}", flush=True)
        await b.stop()

    asyncio.run(demo())
