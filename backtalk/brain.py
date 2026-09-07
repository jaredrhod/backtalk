# backtalk: talk to your agent out loud.
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
"""The warm brain router — dispatches to the configured brain provider.

Supports:
- "claude" (default): Claude Agent SDK session (brain_claude.py)
- "local" / "openai": Local OpenAI-compatible server (brain_local.py)
"""
import os
from backtalk.config import CFG

SESSION_FILE = os.path.join(CFG.get("signals_dir", "/tmp/signals"), ".backtalk_session")


def get_brain_class():
    """Resolve the brain implementation class based on configuration."""
    brain_type = str(CFG.get("brain", "claude")).strip().lower()
    if brain_type in ("local", "openai", "llama", "vllm", "ollama"):
        from backtalk.brain_local import LocalWarmBrain
        return LocalWarmBrain
    from backtalk.brain_claude import ClaudeWarmBrain
    return ClaudeWarmBrain


class WarmBrain:
    """WarmBrain dispatcher: instantiates the configured brain backend."""

    def __new__(cls, *args, **kwargs):
        target_cls = get_brain_class()
        return target_cls(*args, **kwargs)


__all__ = ["WarmBrain", "SESSION_FILE", "get_brain_class"]
