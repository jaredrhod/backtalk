# backtalk: talk to your local AI agent out loud.
# Copyright (C) 2026 Jared Rhodenizer, AnZym contributors
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
"""The warm local brain — connects to a local OpenAI-compatible inference server
(such as llama-server, Ollama, vLLM, or Aphrodite), executing multi-step
filesystem and web tools and streaming spoken output to the mouth.
"""
import asyncio
import json
import os
import re
import subprocess
from pathlib import Path

import httpx

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

from backtalk import signals
from backtalk.config import CFG, DISCIPLINE
from backtalk.vlog import log

_SENTENCE_END = re.compile(r"(?<=[.!?])\s")
_SPECIAL_TOKENS = re.compile(r"<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>|<\|[^|]+?\|>")
_THINK_BLOCK = re.compile(r"<think>[\s\S]*?</think>|\[THINK\][\s\S]*?\[/THINK\]")
_TOOL_TAGS = re.compile(r"<tool_call>[\s\S]*?</tool_call>|<function=[^>]+>[\s\S]*?</function>|<parameter=[^>]+>[\s\S]*?</parameter>")
_CODE_BLOCK = re.compile(r"```[\s\S]*?```", re.DOTALL)
_INLINE_SCRIPT = re.compile(r"<<\s*['\"]?[A-Za-z0-9_]+['\"]?[\s\S]*", re.DOTALL)
_RAW_CODE_LINE = re.compile(r"^(?:import |from |def |class |with open|cat >|\s*return |\s*#|\s*if __name__).*$", re.MULTILINE)

SESSION_FILE = os.path.join(CFG.get("signals_dir", "/tmp/signals"), ".backtalk_session")


def _clean_text(text: str) -> str:
    """Strip special tokens, think blocks, code blocks, and tool tags for speech output."""
    t = _THINK_BLOCK.sub("", text)
    t = _TOOL_TAGS.sub("", t)
    t = _CODE_BLOCK.sub("", t)
    t = _INLINE_SCRIPT.sub("", t)
    t = _SPECIAL_TOKENS.sub("", t)
    t = _RAW_CODE_LINE.sub("", t)
    t = re.sub(r"^#+\s*", "", t, flags=re.MULTILINE)
    t = t.replace("**", "").replace("*", "").replace("`", "")
    lines = [
        line.strip()
        for line in t.split("\n")
        if line.strip() and not line.strip().startswith(("#", "//", "/*", "*", "def ", "import "))
    ]
    return " ".join(lines).strip()


def parse_tool_call(text: str) -> tuple[str, dict] | None:
    """Extract tool name and arguments from JSON or model XML format."""
    # Format 1: <tool_call>JSON</tool_call>
    m1 = re.search(r"<tool_call>([\s\S]*?)</tool_call>", text)
    if m1:
        raw = m1.group(1).strip()
        try:
            d = json.loads(raw)
            return d.get("tool") or d.get("name"), d.get("parameters") or d.get("arguments") or d
        except Exception:
            pass

    # Format 2: <function=NAME><parameter=KEY>VALUE</parameter></function>
    m2 = re.search(r"<function=([a-zA-Z0-9_-]+)>([\s\S]*?)</function>", text)
    if m2:
        tool_name = m2.group(1).strip()
        params = {}
        for pm in re.finditer(r"<parameter=([a-zA-Z0-9_-]+)>([\s\S]*?)</parameter>", m2.group(2)):
            params[pm.group(1).strip()] = pm.group(2).strip()
        return tool_name, params

    # Format 3: JSON markdown code block
    m3 = re.search(r'```(?:json)?\s*(\{\s*"(?:tool|name)"[\s\S]*?\})\s*```', text)
    if m3:
        try:
            d = json.loads(m3.group(1))
            return d.get("tool") or d.get("name"), d.get("parameters") or d.get("arguments") or d
        except Exception:
            pass

    return None


def resolve_project_path(target: str, current_cwd: str) -> str:
    """Map alias or relative name to absolute filesystem directory."""
    aliases = CFG.get("project_aliases", {})
    cleaned = target.strip().lower().replace("_", " ").replace("-", " ")
    for alias, p in aliases.items():
        if alias.lower() in cleaned or cleaned == alias.lower():
            return os.path.expanduser(p)

    for extra in CFG.get("extra_dirs", []):
        expanded_extra = os.path.expanduser(extra)
        if os.path.basename(expanded_extra).lower() == cleaned:
            return expanded_extra

    expanded = os.path.expanduser(target.strip())
    if os.path.isabs(expanded) and os.path.exists(expanded):
        return expanded
    rel = os.path.join(current_cwd, target.strip())
    if os.path.exists(rel):
        return rel
    return expanded


def execute_tool(tool_name: str, args: dict, brain_ref=None) -> str:
    """Execute a local, web, or workspace tool safely."""
    default_ws = CFG.get("agent_dir", os.getcwd())
    cwd = brain_ref.active_project_dir if brain_ref else default_ws
    try:
        signals.set_state("thinking")

        if tool_name == "switch_workspace":
            target = args.get("path") or args.get("project") or args.get("name", "")
            resolved = resolve_project_path(target, cwd)
            log(f"[brain-local] switching workspace to: {resolved}")
            if not os.path.exists(resolved):
                return f"Error: Workspace path {resolved} does not exist."
            if brain_ref:
                brain_ref.active_project_dir = resolved

            tree_lines = []
            for root, dirs, files in os.walk(resolved):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules", "build")]
                rel = os.path.relpath(root, resolved)
                depth = rel.count(os.sep)
                if depth > 2:
                    continue
                indent = "  " * depth
                if rel != ".":
                    tree_lines.append(f"{indent}📂 {os.path.basename(root)}/")
                for f in files[:8]:
                    if not f.startswith("."):
                        tree_lines.append(f"{indent}  📄 {f}")
                if len(tree_lines) > 30:
                    break

            readme_text = ""
            readme_p = os.path.join(resolved, "README.md")
            if os.path.exists(readme_p):
                try:
                    with open(readme_p, "r", encoding="utf-8", errors="ignore") as f:
                        readme_text = f.read(1500)
                except Exception:
                    pass

            return (
                f"--- Switched Active Workspace to: {resolved} ---\n"
                f"Workspace File Tree:\n" + "\n".join(tree_lines[:25]) + "\n"
                + (f"README Overview:\n{readme_text}\n" if readme_text else "")
            )

        elif tool_name == "search_web":
            if not DDGS:
                return "Error: Web search library (ddgs / duckduckgo_search) is not installed."
            query = args.get("query", "")
            if not query:
                return "Error: No search query provided."
            log(f"[brain-local] searching web for: {query}")
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=int(args.get("max_results", 4))):
                    title = r.get("title", "")
                    snippet = r.get("body", "")
                    href = r.get("href", "")
                    results.append(f"Title: {title}\nSnippet: {snippet}\nURL: {href}\n")
            return f"--- Web Search Results for '{query}' ---\n" + ("\n".join(results) if results else "No results found.")

        elif tool_name == "read_web_page":
            if not BeautifulSoup:
                return "Error: BeautifulSoup (bs4) is not installed."
            url = args.get("url", "")
            if not url:
                return "Error: No URL provided."
            log(f"[brain-local] fetching web page: {url}")
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            resp = httpx.get(url, headers=headers, timeout=12.0, follow_redirects=True)
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                tag.decompose()
            text = " ".join(soup.stripped_strings)
            max_chars = int(args.get("max_chars", 3000))
            return f"--- Web Page Content ({url}) ---\n{text[:max_chars]}"

        elif tool_name == "list_dir":
            raw_p = args.get("path", ".")
            p = raw_p if os.path.isabs(raw_p) else os.path.join(cwd, raw_p)
            p = os.path.expanduser(p)
            log(f"[brain-local] inspecting directory: {p}")
            if not os.path.exists(p):
                return f"Error: Directory {p} does not exist."

            tree_items = []
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "build", "install", "node_modules", ".git")]
                rel = os.path.relpath(root, p)
                depth = rel.count(os.sep)
                if depth > 2:
                    continue
                prefix = "  " * depth
                if rel != ".":
                    tree_items.append(f"{prefix}📂 {os.path.basename(root)}/")
                for f in files:
                    if not f.startswith("."):
                        full_rel = os.path.join(rel, f) if rel != "." else f
                        tree_items.append(f"{prefix}  📄 {full_rel}")
                if len(tree_items) > 50:
                    break

            return f"Recursive File Tree of {p}:\n" + "\n".join(tree_items[:45])

        elif tool_name == "read_file":
            raw_p = args.get("path", "")
            p = raw_p if os.path.isabs(raw_p) else os.path.join(cwd, raw_p)
            p = os.path.expanduser(p)
            log(f"[brain-local] reading file: {p}")
            if not os.path.exists(p):
                found = list(Path(cwd).glob(f"**/{os.path.basename(raw_p)}"))
                if found:
                    p = str(found[0])
                else:
                    return f"Error: File {raw_p} does not exist in {cwd}."
            max_lines = int(args.get("max_lines", 120))
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                lines = [f.readline() for _ in range(max_lines)]
            return f"--- File Content: {p} ---\n" + "".join(lines)

        elif tool_name == "search_files":
            raw_p = args.get("path", ".")
            base = raw_p if os.path.isabs(raw_p) else os.path.join(cwd, raw_p)
            base = os.path.expanduser(base)
            query = args.get("query", "")
            log(f"[brain-local] searching files for: {query} in {base}")
            matches = []
            for root, _, filenames in os.walk(base):
                for fn in filenames:
                    if query.lower() in fn.lower():
                        matches.append(os.path.join(root, fn))
                        if len(matches) >= 20:
                            break
                if len(matches) >= 20:
                    break
            return f"Found matching files for '{query}':\n" + "\n".join(matches)

        elif tool_name == "run_command":
            cmd = args.get("cmd", "")
            target_cwd = args.get("cwd", cwd)
            target_cwd = target_cwd if os.path.isabs(target_cwd) else os.path.join(cwd, target_cwd)
            target_cwd = os.path.expanduser(target_cwd)
            log(f"[brain-local] running command: {cmd} (in {target_cwd})")
            res = subprocess.run(cmd, shell=True, cwd=target_cwd, capture_output=True, text=True, timeout=15)
            out = res.stdout if res.stdout else res.stderr
            return f"Command output ($ {cmd} in {target_cwd}):\n{out[:2000]}"

        return f"Unknown tool: {tool_name}"
    except Exception as e:
        return f"Tool execution failed: {e}"


TOOL_PROMPT = """
### OPERATIONAL DIRECTIVE: MULTI-STEP TOOLS & SPOKEN ANSWERS
You are a voice-interactive assistant connected directly to audio input and speech synthesis.

RULES:
1. When you need to inspect directories, read code files, or search the web, invoke tools using <tool_call>{"tool": "name", ...}</tool_call>.
2. `list_dir` returns the recursive file tree, so you can locate and read target files immediately.
3. As soon as you have inspected the necessary code or files, STOP calling tools and deliver your spoken answer directly.
4. Keep spoken responses clear, concise, and natural for text-to-speech audio. Avoid markdown tables, URLs, or long unpronounceable code blocks.

Available Tools:
- switch_workspace(path)
- list_dir(path)
- read_file(path, max_lines)
- search_files(path, query)
- run_command(cmd, cwd)
- search_web(query)
- read_web_page(url)
"""


class LocalWarmBrain:
    """A persistent local LLM session via an OpenAI-compatible endpoint (llama-server, Ollama, etc.)."""

    def __init__(self, model: str | None = None, can_use_tool=None, resume_id: str | None = None):
        self.api_base = CFG.get("api_base", "http://127.0.0.1:8080/v1").rstrip("/")
        self.model = model or CFG.get("model", "default")
        self.active_project_dir = CFG.get("agent_dir", os.getcwd())
        self._can_use_tool = can_use_tool
        self.session = {"turns": 0, "out_tokens": 0, "in_tokens": 0, "cost": 0.0}
        self.messages = []
        self._dirty = False
        self._interrupted = False
        self.system_prompt = self._load_system_prompt()
        self.permission_mode = CFG.get("permission_mode", "ask")

    def _get_workspace_snapshot(self) -> str:
        try:
            cwd = self.active_project_dir
            if not os.path.exists(cwd):
                return ""
            entries = os.listdir(cwd)
            dirs = [f"{e}/" for e in sorted(entries) if os.path.isdir(os.path.join(cwd, e)) and not e.startswith(".")]
            files = [e for e in sorted(entries) if os.path.isfile(os.path.join(cwd, e)) and not e.startswith(".")]
            return (
                f"\n[Active Workspace: {cwd}]\n"
                f"Directories: {', '.join(dirs[:15])}\n"
                f"Files: {', '.join(files[:15])}"
            )
        except Exception:
            return ""

    def _load_system_prompt(self) -> str:
        agent_dir = Path(os.path.expanduser(CFG.get("agent_dir", os.getcwd())))
        prompt_parts = [DISCIPLINE, TOOL_PROMPT]

        for filename in ("AGENT.md", "CLAUDE.md", "SYSTEM.md"):
            p = agent_dir / filename
            if p.exists():
                try:
                    prompt_parts.append(p.read_text(encoding="utf-8"))
                    log(f"[brain-local] loaded persona from {p}")
                    break
                except Exception as e:
                    log(f"[brain-local] error reading {p}: {e}")

        for extra in CFG.get("extra_dirs", []):
            extra_path = Path(os.path.expanduser(extra))
            idx = extra_path / "VAULT-INDEX.md"
            if idx.exists():
                try:
                    prompt_parts.append(f"## Context ({extra})\n{idx.read_text(encoding='utf-8')[:3000]}")
                    log(f"[brain-local] loaded index from {idx}")
                except Exception:
                    pass
            readme = extra_path / "README.md"
            if readme.exists():
                try:
                    prompt_parts.append(f"## Workspace Overview ({extra})\n{readme.read_text(encoding='utf-8')[:4000]}")
                    log(f"[brain-local] loaded workspace summary from {readme}")
                except Exception:
                    pass

        return "\n\n".join(prompt_parts)

    async def start(self):
        full_prompt = self.system_prompt + self._get_workspace_snapshot()
        self.messages = [{"role": "system", "content": full_prompt}]
        self._dirty = False
        self._interrupted = False
        log(f"[brain-local] connected to {self.api_base} (model={self.model}, cwd={self.active_project_dir})")

    async def stop(self):
        self.messages.clear()
        self._dirty = False

    async def interrupt(self):
        self._interrupted = True
        self._dirty = False

    async def reset_turn(self, timeout: float = 8.0):
        self._dirty = False
        self._interrupted = False

    async def set_permission_mode(self, mode: str):
        self.permission_mode = mode
        log(f"[brain-local] permission mode set to: {mode}")

    async def context_usage(self):
        return {
            "total_tokens": self.session["in_tokens"] + self.session["out_tokens"],
            "turns": self.session["turns"],
        }

    async def command(self, cmd: str) -> str:
        parts = cmd.strip().split()
        if not parts:
            return ""
        verb = parts[0].lower()
        if verb == "/clear":
            full_prompt = self.system_prompt + self._get_workspace_snapshot()
            self.messages = [{"role": "system", "content": full_prompt}]
            self.session = {"turns": 0, "out_tokens": 0, "in_tokens": 0, "cost": 0.0}
            return "Conversation history cleared."
        elif verb == "/compact":
            if len(self.messages) > 9:
                self.messages = [self.messages[0]] + self.messages[-8:]
            return "Session compacted."
        elif verb == "/model" and len(parts) > 1:
            self.model = parts[1]
            return f"Switched model to {self.model}."
        elif verb == "/effort":
            return "Effort level updated."
        elif verb in ("/cd", "/workspace") and len(parts) > 1:
            res = execute_tool("switch_workspace", {"path": parts[1]}, self)
            return res
        return f"Command acknowledged: {cmd}"

    async def _query_llm(self, messages: list) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": 0.3,
            "max_tokens": 2048,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{self.api_base}/chat/completions", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            log(f"[brain-local] LLM returned status {resp.status_code}: {resp.text[:200]}")
            return ""

    async def ask_stream(self, utterance: str):
        self._dirty = True
        self._interrupted = False

        snapshot = self._get_workspace_snapshot()
        user_msg = utterance
        if snapshot:
            user_msg = f"{utterance}\n\n[Active Workspace Telemetry]:{snapshot}"
        self.messages.append({"role": "user", "content": user_msg})

        max_tool_turns = 5
        for turn_idx in range(max_tool_turns):
            reply = await self._query_llm(self.messages)
            parsed = parse_tool_call(reply)
            if not parsed:
                break

            tool_name, args = parsed
            log(f"[brain-local] tool call ({turn_idx+1}/{max_tool_turns}): {tool_name} with args {args}")

            tool_result = execute_tool(tool_name, args, self)
            log(f"[brain-local] tool output ({len(tool_result)} chars):\n{tool_result[:300]}...")

            self.messages.append({"role": "assistant", "content": reply})

            if turn_idx == max_tool_turns - 1:
                self.messages.append({
                    "role": "user",
                    "content": f"[Tool Result from {tool_name}]:\n{tool_result}\n\n"
                               f"You now have all necessary data. Deliver your final spoken answer now (do NOT call any tools)."
                })
            else:
                self.messages.append({
                    "role": "user",
                    "content": f"[Tool Result from {tool_name}]:\n{tool_result}\n\n"
                               f"If you need to read a specific code file, call read_file. Otherwise, deliver your spoken answer directly."
                })

        buf = ""
        full_reply = ""
        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 2048,
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", f"{self.api_base}/chat/completions", json=payload) as response:
                    if response.status_code != 200:
                        err_text = await response.aread()
                        log(f"[brain-local] server error {response.status_code}: {err_text.decode('utf-8', errors='ignore')}")
                        yield "I had trouble connecting to the local inference server."
                        self._dirty = False
                        return

                    async for line in response.aiter_lines():
                        if self._interrupted:
                            log("[brain-local] stream interrupted")
                            break
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk["choices"][0]["delta"].get("content", "")
                            if delta:
                                if "<|im_end|>" in delta or "<|endoftext|>" in delta:
                                    delta = _SPECIAL_TOKENS.sub("", delta)
                                    buf += delta
                                    full_reply += delta
                                    break
                                buf += delta
                                full_reply += delta
                                while True:
                                    m_sent = _SENTENCE_END.search(buf)
                                    if not m_sent:
                                        break
                                    sentence, buf = buf[:m_sent.end()].strip(), buf[m_sent.end():]
                                    cleaned = _clean_text(sentence)
                                    if cleaned:
                                        yield cleaned
                        except Exception:
                            continue
        except Exception as e:
            log(f"[brain-local] error querying {self.api_base}: {e}")
            yield "Sorry, I lost connection to the local model."

        tail = _clean_text(buf)
        if tail and not self._interrupted:
            yield tail

        if full_reply:
            self.messages.append({"role": "assistant", "content": _clean_text(full_reply)})
            self.session["turns"] += 1
            self.session["out_tokens"] += len(full_reply.split()) * 2
            self.session["in_tokens"] += len(utterance.split()) * 2

        self._dirty = False
        self._interrupted = False


WarmBrain = LocalWarmBrain
