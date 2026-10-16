"""Does every reply reach the ONE English voice -- and espeak-ng carry
it when Kokoro cannot?

ENGLISH ONLY. Seyon speaks one language, so there is nothing left to
route: ElevenLabs when configured, Kokoro otherwise, espeak-ng behind
Kokoro. This file pins the two halves of that promise:

  1. Foreign-script replies -- Tamil first, because it used to be the
     routed language -- come out of the English Kokoro pipeline at
     24kHz. No MMS, no espeak, no language branch: routing a reply by
     what it is written in is how a foreign accent got onto the speaker
     before, and this is the regression that must not come back.
  2. When Kokoro itself breaks, espeak-ng at 22050 still speaks --
     English or foreign text alike. Silence is the one outcome worse
     than a robotic voice.

Foreign text is checked with the old turn hints still set ("ta" etc.),
because the stale hint is exactly what the deleted detection machinery
used to act on; it must be inert now."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np                                   # noqa: E402
from backtalk import mouth                          # noqa: E402

KOKORO = 24000
ESPEAK = 22050

# Languages we do not speak. Each is unmistakable in its own script, and
# each must come out as English Kokoro audio rather than espeak.
FOREIGN = [
    ("ta", "வணக்கம், தயவுசெய்து நோட்டபாட் திறக்க முடியுமா?"),
    ("hi", "नमस्ते, क्या आप नोटपैड खोल सकते हैं?"),
    ("ko", "안녕하세요, 메모장을 열어 주세요."),
    ("ar", "مرحبا، افتح المفكرة من فضلك."),
    ("ru", "Привет, открой блокнот пожалуйста."),
    ("th", "สวัสดี กรุณาเปิดโน้ตแพด"),
    ("ja", "こんにちは、メモ帳を開いてください。"),
    ("de", "Guten Morgen, ich habe eine Frage fuer dich."),
    ("es", "Hola, puedes abrir el Bloc de notas por favor?"),
]


def render(turn_lang, text, timeout=90.0):
    """Synthesise one sentence and report (rate, seconds, peak)."""
    mouth.set_turn_language(turn_lang)
    try:
        chunks = [(r, np.frombuffer(p, dtype=np.int16)
                   if isinstance(p, (bytes, bytearray))
                   else np.asarray(p, dtype=np.int16))
                  for r, p in mouth.synth_stream(text, timeout=timeout)]
    except Exception as e:                            # noqa: BLE001
        return 0, 0.0, 0, "%s: %s" % (type(e).__name__, e)
    chunks = [(r, c) for r, c in chunks if c.size]
    pcm = (np.concatenate([c for _r, c in chunks]) if chunks
           else np.zeros(0, np.int16))
    rate = chunks[0][0] if chunks else 0
    secs = len(pcm) / rate if rate else 0.0
    peak = int(np.abs(pcm).max()) if pcm.size else 0
    return rate, secs, peak, ""


bad = []

print("espeak-ng found at: %s" % mouth._ESPEAK)
print("voice mapping: %s" % {k: mouth.espeak_voice_for(k)
                             for k in ("ta", "hi", "ko", "en")})
print()

print("--- plain English turn (must be kokoro, 24kHz) ---")
rate, secs, peak, err = render("en", "Opening Notepad now.")
ok = rate == KOKORO and secs > 0.3 and peak > 500
print("  %s en   rate=%-6d %.2fs peak=%-6d%s" % (
    "ok  " if ok else "FAIL", rate, secs, peak, ("  " + err) if err else ""))
if not ok:
    bad.append("en")

print("\n--- turn language cleared (must fall back to kokoro default) ---")
rate, secs, peak, err = render(None, "Opening Notepad now.")
ok = rate == KOKORO and secs > 0.3 and peak > 500
print("  %s none rate=%-6d %.2fs peak=%-6d%s" % (
    "ok  " if ok else "FAIL", rate, secs, peak, ("  " + err) if err else ""))
if not ok:
    bad.append("no-turn-lang")

print("\n--- foreign-script replies (must stay on the ENGLISH kokoro voice) ---")
for code, text in FOREIGN:
    # The Tamil row keeps its old 'ta' turn hint on purpose: that is the
    # exact signal the deleted detect_language machinery used to act on.
    rate, secs, peak, err = render("ta" if code == "ta" else code, text)
    ok = rate == KOKORO and secs > 0.3 and peak > 500
    print("  %s %-3s rate=%-6d %.2fs peak=%-6d%s" % (
        "ok  " if ok else "FAIL", code, rate, secs, peak,
        ("  " + err) if err else ""))
    if not ok:
        bad.append(code)

print("\n--- kokoro down -> espeak-ng must still speak (never mute) ---")
_real_kokoro = mouth._stream_kokoro


def _broken_kokoro(text):
    raise RuntimeError("simulated: kokoro unavailable")


mouth._stream_kokoro = _broken_kokoro
rate, secs, peak, err = render("en", "Opening Notepad now.")
ok = rate == ESPEAK and secs > 0.3 and peak > 500
print("  %s en   rate=%-6d %.2fs peak=%-6d%s" % (
    "ok  " if ok else "FAIL", rate, secs, peak, ("  " + err) if err else ""))
if not ok:
    bad.append("espeak-fallback")

print("\n--- kokoro down + foreign text -> espeak-ng speaks that too ---")
rate, secs, peak, err = render(
    "ta", "வணக்கம், நண்பா. உங்களுக்கு உதவ வேண்டுமா?")
ok = rate == ESPEAK and secs > 0.3 and peak > 500
print("  %s ta   rate=%-6d %.2fs peak=%-6d%s" % (
    "ok  " if ok else "FAIL", rate, secs, peak, ("  " + err) if err else ""))
if not ok:
    bad.append("espeak-foreign")
mouth._stream_kokoro = _real_kokoro

mouth.set_turn_language(None)
print("\nENGLISH VOICE OK" if not bad else "FAILURES: %s" % bad)

# Exit nonzero on failure, so a caller that checks the exit code rather than
# the screen is not told a broken fallback passed.
sys.exit(1 if bad else 0)
