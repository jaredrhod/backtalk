"""STT accuracy for the two languages the ear hears: English and Tamil.

The end-to-end wiring test fed Whisper a noise-shaped signal, so it proved
the plumbing and nothing about language handling. This one synthesises
genuine speech with the very voices Seyon speaks with, resamples it to the
16kHz mono the ear expects, and asks Whisper what it heard.

That round trip is the one that matters: it is exactly what happens when
Vijay holds the key and speaks, except the speaker is a model instead of a
person.

Only English and Tamil are here. The other languages this file used to
cover -- Spanish, French, Hindi, Italian, Japanese, Portuguese, Mandarin --
stopped being worth a two-minute test when the assistant went down to a
two-language policy, and English-only replies did not change what the EAR
is asked to do, so they stayed out.

The two checks per language fail independently:
  "lang"   - did whisper identify the language? (its own guess)
  "heard"   - did the SENTENCE SURVIVE? (needle word present) - this is
              what the brain actually receives, and the one that is scored.
The language code is asserted only for English. For Tamil it is reported
but not scored: this test's Tamil is espeak-ng's synthetic voice, and
Whisper's language ID on that particular audio is a coin flip (it has been
seen reporting Romanian and Arabic for unmistakably Tamil sentences) while
still writing much of the sentence in Tamil script. Real human Tamil through
the real microphone does detect as Tamil; a model cannot stand in for it
here, so the sentence surviving is what this pins.
"""
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np                                   # noqa: E402
from backtalk import ears, mouth                     # noqa: E402
from backtalk.config import CFG                      # noqa: E402

# Pin the model in memory before anything loads it. The committed config
# upgrades to the cached `medium` model, which is 1.4GB; combined with the
# Kokoro pipeline this test synthesises over, that overruns available RAM
# and dies in mkl_malloc. `small` is multilingual too -- this test is about
# the language coming out, not transcription quality.
CFG["stt_model"] = "small"
CFG["stt_model_if_cached"] = "small"

SR = 16000          # what the ear wants

# (code, sentence, needle word, assert_language)
# The needle is the word in THAT language: matching "notepad" in a Tamil
# sentence would fail a perfect transcription, since the word is Notepad's
# own name written in Tamil script.
CASES = [
    ("en", "Hello there, could you open Notepad for me please?",
     "notepad", True),
    # "vanakkam, please open notepad" -- rendered straight through
    # espeak-ng's Tamil voice. The reply path is English-only now, so
    # synth_stream would hand this sentence to the English Kokoro voice,
    # which cannot say Tamil; this row is about the ear hearing Tamil,
    # not about the mouth speaking it.
    ("ta", "வணக்கம், நோட்பைட் ஐ திறக்கவும்.",
     ("வணக்கம்", "vanakkam"), False),
]

OUT = tempfile.mkdtemp()


def speech_pcm(text, code="en"):
    """Real speech for `text`, as int16 at 16kHz mono.

    English renders through the live reply path (synth_stream -> kokoro).
    The Tamil row renders straight through espeak-ng's Tamil voice,
    because the reply path speaks English only and would read Tamil
    script with an English voice -- a reading Whisper could not match
    against the needle no matter how well it transcribed.
    """
    chunks, rate = [], 0
    if code == "ta":
        stream = ((mouth._espeak_rate, p)
                  for p in mouth._stream_espeak(text, "ta"))
    else:
        stream = mouth.synth_stream(text, timeout=90.0)
    for r, pcm in stream:
        a = (np.frombuffer(pcm, dtype=np.int16) if isinstance(pcm, (bytes, bytearray))
             else np.asarray(pcm, dtype=np.int16))
        if a.size:
            chunks.append(a)
            rate = rate or r
    if not chunks:
        return None, 0
    # The rate is whichever pipeline actually answered: Kokoro is 24000,
    # espeak-ng 22050. Assuming one for the other pitched the Tamil case
    # wrong by 8%, which is enough to cost a transcription.
    wave = np.concatenate(chunks).astype(np.float32)
    n_out = int(len(wave) * SR / rate)
    idx = np.linspace(0, len(wave) - 1, n_out)
    pcm = np.interp(idx, np.arange(len(wave)), wave).astype(np.int16)
    return pcm, rate


def word_overlap(heard, needle):
    """Does the transcript contain the key word, in ANY script form?"""
    def fold(s):
        import unicodedata
        s = "".join(c for c in unicodedata.normalize("NFKD", s)
                    if not unicodedata.combining(c))
        return "".join(c for c in s.lower() if c.isalnum())
    needles = (needle,) if isinstance(needle, str) else needle
    return any(fold(n) in fold(heard) for n in needles)


print("=" * 74)
print("STT ACCURACY for heard languages (English, Tamil input), real speech")
print("  model: %s" % ears._resolve_stt_model())
print("=" * 74)

rows, bad, skipped = [], [], []
for code, sentence, needle, assert_lang in CASES:
    # Each language can spin up its own Kokoro pipeline and mouth._pipes
    # keeps them alive, so pipelines accumulate. The live voice only ever
    # holds one, so this is a property of the test -- drop each pipeline
    # and prune ctranslate2's allocator between cases so the run fits in
    # RAM.
    try:
        for _p in list(getattr(mouth, "_pipes", {}).values()):
            del _p
        mouth._pipes.clear()
        mouth._pipe = None
        import gc
        gc.collect()
        from ctranslate2 import set_cpu_allocator_options  # noqa
        set_cpu_allocator_options("arena=0")  # release arena cache
        set_cpu_allocator_options("")          # back to defaults
    except Exception:
        pass
    try:
        pcm, rate = speech_pcm(sentence, code)
    except Exception as e:
        print("  %-3s SKIP  could not synthesise: %s" % (code, e))
        skipped.append(code)
        continue
    if pcm is None or pcm.size < SR // 4:
        print("  %-3s FAIL  no audio produced" % code)
        bad.append(code)
        continue

    path = os.path.join(OUT, "%s.wav" % code)
    import wave as wavemod
    with wavemod.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())

    heard, detected = ears.transcribe_language(pcm)
    heard = heard.strip()
    lang_ok = (detected or "").lower()[:2] == code
    word_ok = word_overlap(heard, needle)
    # A misidentified language is only cosmetic IF the words still arrived
    # correctly -- it picks the accent, not the meaning. So the two are
    # reported and scored separately rather than lumped into one verdict,
    # and the language guess is only fatal where we trust it (English).
    scored = word_ok and (lang_ok or not assert_lang)
    status = "ok  " if scored else "FAIL"
    print("  %s %-3s rate=%-6d det=%-4s word=%-6s %r" % (
        status, code, rate, detected, "found" if word_ok else "MISSED",
        heard[:50]))
    if not scored:
        bad.append(code)
    rows.append((code, detected, lang_ok, word_ok))

print()
total = len(CASES)
print("languages run      : %d/%d" % (len(rows), total))
print("sentence survived  : %d/%d  <- what matters" % (
    sum(1 for r in rows if r[3]), total))
print("language identified: %d/%d  <- picks the accent" % (
    sum(1 for r in rows if r[2]), total))
mislabelled = [r[0] for r in rows if not r[2] and r[3]]
if mislabelled:
    print("  (words correct but accent may differ: %s)" % mislabelled)
if skipped:
    # A skipped language used to vanish from the denominator entirely, so
    # losing the run to an out-of-memory still printed "STT OK" on a single
    # pass. Skips are unproven, not passes -- say so loudly.
    print()
    print("NOT TESTED (could not synthesise, usually RAM pressure): %s"
          % ", ".join(skipped))
    print("Close the live stack and re-run: a 1.4GB Whisper model plus this")
    print("test's own Kokoro does not fit alongside Photoshop at times.")
print()
if bad:
    print("STT PROBLEM LANGUAGES: %s" % bad)
elif skipped:
    print("STT INCOMPLETE -- %d language(s) never ran, so this is not a pass"
          % len(skipped))
else:
    print("STT OK")

# A mislabelled accent is a known model quirk, not a failure; a broken or
# unproven language is. Exit nonzero so a caller reading the exit code cannot
# be told a partial run was a pass.
sys.exit(1 if (bad or skipped) else 0)
