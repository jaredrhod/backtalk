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
"""The board — backtalk's hand on a barehands air-board.

A barehands checkout (github.com/jaredrhod/barehands) serves a hand-tracked
glass board on localhost and takes tiny JSON commands on POST /cmd: put a
card up, present something center stage, explode a model, sweep the board
clean. Its server allowlists the action and jails every media path against
its own folder, so it is safe to hand to an assistant.

This module is only the wire. The spoken wording that turns "put the plan
on the board" into a command lives in main.py beside the other spoken
commands; here we just POST it. Point board_url at the server (the default
matches barehands' own port); leave it "" and every call is a quiet no-op,
so a box with no board never notices.

Every send is wrapped: the board must never crash the voice line.
"""
import json
from urllib.request import Request, urlopen

from backtalk.config import CFG

_URL = str(CFG.get("board_url") or "").rstrip("/")
_TIMEOUT = 2.0


def enabled() -> bool:
    """True when a board is configured. Until one is, the voice line never
    even tries to parse a board phrase, so ordinary talk is untouched."""
    return bool(_URL)


def send(cmd: dict, timeout: float = _TIMEOUT) -> bool:
    """Post one board command. True only when the board accepts it (a 2xx).

    A board that is off, wedged, or on another machine is a soft False, not
    an exception: reaching for the glass must never take the voice line
    down with it. Never raises."""
    if not _URL:
        return False
    try:
        data = json.dumps(cmd).encode()
        req = Request(_URL + "/cmd", data=data,
                      headers={"Content-Type": "application/json"},
                      method="POST")
        with urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (OSError, ValueError, TypeError):
        return False
