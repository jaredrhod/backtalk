"""The board wire, isolated: spoken commands turned into barehands /cmd calls.

board.py is one HTTP POST and nothing else, so it is driven end to end with
a fake urlopen pointed at a temp URL. What is checked is the contract the
voice line leans on: the right endpoint and method, the command serialized
as JSON, a 2xx read as success, and the promise that a board which is off,
broken, or unreachable is a quiet False rather than an exception.
"""
import json
from pathlib import Path
import sys
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtalk.config import CFG                             # noqa: E402

CFG["board_url"] = "http://127.0.0.1:8794/"

from backtalk import board                                  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print("    %s %s%s" % ("ok  " if cond else "FAIL", name,
                           "" if cond else "   <- " + str(detail)))
    if not cond:
        failures.append(name)


class _Resp:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


print("=" * 66)
print("THE BOARD WIRE")
print("=" * 66)

check("a configured url reports the board enabled", board.enabled() is True)

seen = {}


def capture(req, timeout=None):
    seen["url"] = req.full_url
    seen["method"] = req.get_method()
    seen["body"] = req.data
    seen["timeout"] = timeout
    seen["ctype"] = req.get_header("Content-type")
    return _Resp(204)


with mock.patch.object(board, "urlopen", side_effect=capture):
    ok = board.send({"a": "clear"})

check("a 204 reads as success", ok is True)
check("the call lands on /cmd", seen["url"] == "http://127.0.0.1:8794/cmd",
      seen["url"])
check("the call is a POST", seen["method"] == "POST", seen["method"])
check("the command travels as JSON",
      json.loads(seen["body"]) == {"a": "clear"}, seen["body"])
check("the content type is JSON", seen["ctype"] == "application/json",
      seen["ctype"])
check("a default timeout rides along", seen["timeout"] == board._TIMEOUT,
      seen["timeout"])

with mock.patch.object(board, "urlopen", return_value=_Resp(400)):
    check("a rejected command is a soft False",
          board.send({"a": "explode"}) is False)

with mock.patch.object(board, "urlopen", side_effect=OSError("refused")):
    check("an unreachable board is a soft False",
          board.send({"a": "reset"}) is False)

with mock.patch.object(board, "urlopen", side_effect=ValueError("bad url")):
    check("a malformed url is a soft False",
          board.send({"a": "assemble"}) is False)

with mock.patch.object(board, "_URL", ""), \
     mock.patch.object(board, "urlopen") as dead:
    check("no url configured reads as disabled", board.enabled() is False)
    check("a disabled board still sends nothing",
          board.send({"a": "clear"}) is False)
check("a disabled board never opens a socket", dead.call_count == 0,
      dead.call_count)

print("\n" + "=" * 66)
print("BOARD OK" if not failures else "BOARD FAILURES: %s" % failures)
sys.exit(1 if failures else 0)
