"""Drive backtalk's real ask_stream so the live-data injection is exercised.

This one is a manual driver, not an automated test: it asks whatever
questions you hand it and prints the answers.

    uv run python tests/test_live_path.py "what is two plus two"

It needs at least one question. Run bare it used to fall through the loop
below, assert nothing, and exit 0 -- so anything collecting tests/test_*.py
counted it as a pass that had in fact checked nothing at all. Better to say
so than to be quietly green.
"""
import asyncio
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtalk.brain import WarmBrain  # noqa: E402


async def ask(q):
    brain = WarmBrain()
    await brain.start()
    t0 = time.time()
    out = []
    async for sentence in brain.ask_stream(q):
        out.append(sentence)
        if time.time() - t0 > 180:
            break
    return time.time() - t0, " ".join(out)


if not sys.argv[1:]:
    print(__doc__.strip())
    sys.exit(2)

for question in sys.argv[1:]:
    took, reply = asyncio.run(ask(question))
    print(f"\n=== {took:.1f}s  {question}")
    print(reply[:700])
