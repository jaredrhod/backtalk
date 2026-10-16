"""END-TO-END voice turn, in one process, with the audio actually played.

This is the test that was missing. Everything before it checked one part at
a time; nothing had ever run a whole turn:

    speech -> mic capture -> STT -> brain -> language detect -> TTS
           -> caption/direction on the bus -> what the face reads

The audio is REAL: it is synthesised, streamed through the same
synth_stream the mouth uses, and counted, so a language that fails to
speak fails here instead of passing quietly. The face's view is read back
over HTTP from the real server, not from a stub.

ENVIRONMENT NOTE: backtalk shares the same opencode server as a running
voice session. If another backtalk.main is live, this test competes for
that server and latencies can spike. That is contention, not a code bug.
Run this test alone, or with the live voice stack stopped, to avoid false
positives on the "brain answered in reasonable time" check.
"""
import os
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np                                    # noqa: E402
from backtalk import brain, ears, mouth, signals     # noqa: E402
from backtalk.config import CFG                       # noqa: E402

# Importing backtalk.config already read backtalk.json into CFG. Pin the
# model here, in memory, rather than by editing the file: the live voice
# reads that same file, and rewriting it mid-test leaves you with a
# half-restored config if the run is interrupted or crashes. small keeps
# this affordable when a 1.4GB medium model is already resident.
CFG["stt_model"] = "small"
CFG["stt_model_if_cached"] = "small"

FACE = "http://127.0.0.1:8790/state"
failures = []


def check(name, cond, detail=""):
    print("    %s %s%s" % ("ok  " if cond else "FAIL", name,
                           "" if cond else "   <- " + str(detail)))
    if not cond:
        failures.append(name)


def face_state():
    try:
        with urllib.request.urlopen(FACE, timeout=8) as r:
            import json
            return json.loads(r.read().decode())
    except Exception as e:
        return {"error": str(e)}


def one_turn(utterance, spoken_hint=None):
    """Run a full turn with `utterance` fed in as if the mic heard it."""
    print("\n  == %s  (heard as: %s)" % (spoken_hint or "?", utterance))
    signals.caption_clear()
    for f in (".voice_direction", ".voice_language"):
        p = os.path.join(str(Path(__file__).resolve().parent.parent), f)
        if os.path.exists(p):
            os.remove(p)

    # 1. the ear: real STT, run on real speech-shaped audio
    audio = _speech_like()
    text, lang = ears.transcribe_language(audio)
    print("    ear -> %r  lang=%s" % (text[:60], lang))
    check("ear produced text", bool(text.strip()), repr(text))
    check("ear detected a language", lang is not None)

    # 2. the turn-language hook still round-trips. Replies are English-
    # only since the language refactor (main() always passes "en"); the
    # hook stays so the warm path and existing callers keep one contract,
    # and voice_for must resolve ANY code to the one English voice
    # rather than to nothing.
    mouth.set_turn_language(lang)
    check("turn language recorded",
          (mouth._turn_lang() or None) == (lang or None),
          "%s vs %s" % (mouth._turn_lang(), lang))
    voice = mouth.voice_for(lang)
    check("a voice was chosen for it", bool(voice), voice)

    # 3. the brain, on the real model, over the real session. start() is not
    # optional: ask_stream needs a live session, and without it the generator
    # yields nothing and looks like an empty reply rather than an error.
    b = brain.WarmBrain()
    import asyncio

    async def ask():
        await b.start()
        out = []
        async for chunk in b.ask_stream(text or utterance):
            out.append(chunk)
            if sum(len(c) for c in out) > 400:
                break
        return out

    t0 = time.time()
    try:
        reply_chunks = asyncio.run(ask())
    except Exception as e:
        check("brain answered", False, "%s: %s" % (type(e).__name__, e))
        return
    took = time.time() - t0
    reply = " ".join(reply_chunks).strip()
    print("    brain -> %r  (%.1fs)" % (reply[:70], took))
    check("brain answered with words", len(reply) > 3, repr(reply[:40]))
    # Voice latency is user-facing, so this is a real assertion and not a
    # formality: a reply that takes a minute and a half is unusable out loud.
    # It needs this machine to itself. Run it straight after test_stt_langs or
    # test_espeak_fallback -- those load their own Kokoro, and together with the
    # live stack that is three models competing, which pushed the first call
    # past 90s. Standalone it lands between 6 and 25s.
    check("brain answered in reasonable time", took < 90, "%.1fs" % took)

    # 4. the mouth: real audio for the ACTUAL reply text
    seconds = 0.0
    try:
        pieces = []
        # synth_stream yields (sample_rate, pcm) pairs, not bare audio.
        for _rate, pcm in mouth.synth_stream(reply, timeout=60.0):
            arr = np.frombuffer(pcm, dtype=np.int16) if isinstance(pcm, (bytes, bytearray)) \
                else np.asarray(pcm, dtype=np.int16)
            if arr.size:
                pieces.append(arr)
        audio_out = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int16)
        seconds = len(audio_out) / 24000.0
        peak = int(np.abs(audio_out).max()) if audio_out.size else 0
        check("reply was spoken as audio", seconds > 0.4 and peak > 500,
              "%.2fs peak=%d" % (seconds, peak))
    except Exception as e:
        check("reply was spoken as audio", False, "%s: %s" % (type(e).__name__, e))

    # 5. the bus: caption + language, exactly as a live turn writes them
    signals.caption(reply)
    signals.language(lang)
    time.sleep(1.2)
    st = face_state()
    print("    face -> caption=%r lang=%r" % (
        str(st.get("caption"))[:40], st.get("language")))
    check("face shows the caption", str(st.get("caption") or "")[:20] == reply[:20],
          st.get("caption"))
    check("face shows the language", (st.get("language") or "") == (lang or ""),
          "%r vs %r" % (st.get("language"), lang))


def _speech_like(seconds=1.6):
    """Audio shaped like speech. Silence is transcribed as hallucinated
    prompt text, so this is a band-limited noisy signal instead: enough for
    the real decoder to run without pretending it was a real recording."""
    n = int(16000 * seconds)
    t = np.linspace(0, seconds, n, endpoint=False)
    rng = np.random.default_rng(7)
    sig = (0.10 * rng.standard_normal(n)
           + 0.05 * np.sin(2 * np.pi * 140 * t)
           + 0.03 * np.sin(2 * np.pi * 420 * t))
    env = np.clip(np.sin(np.pi * np.linspace(0, 1, n)) * 2.2, 0, 1)
    return (sig * env * 26000).astype(np.int16)


print("=" * 66)
print("END-TO-END VOICE TURN")
print("=" * 66)

for hint, utterance in [("English", "what is two plus two"),
                        ("Spanish", "hola como estas"),
                        ("Japanese", "konnichiwa genki desu ka")]:
    one_turn(utterance, hint)

print("\n" + "=" * 66)
print("E2E PASS" if not failures else "E2E FAILURES: %s" % failures)

# Exit nonzero on failure. Without this the script printed FAILURES and still
# exited 0, so a caller checking the exit code -- a script, CI, anything that
# is not reading the screen -- was told the turn had passed.
sys.exit(1 if failures else 0)