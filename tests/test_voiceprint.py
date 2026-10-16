"""The voiceprint: classic-DSP owner verification, on synthetic audio.

No mic and no model. Two takes of the SAME synthetic voice must sit closer
together than the same take of a different one, the profile must round-trip
through disk, and every failure path -- missing file, garbage, an old
version, an unwritable path -- must degrade to "not gated" rather than
crash the voice line.
"""
from pathlib import Path
import json
import os
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                       # noqa: E402
from backtalk import voiceprint as vp                    # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print("    %s %s%s" % ("ok  " if cond else "FAIL", name,
                           ("  " + str(detail)) if detail else ""))
    if not cond:
        FAILURES.append(name)
    return cond


def _voiced(seed, tilt, n=16000):
    """A deterministic synthetic 'voice': seeded noise with a one-pole
    spectral tilt. Same tilt -> same-sounding speaker; different tilt ->
    a different one. Two seeds at the same tilt are two takes of one
    person."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = np.empty(n)
    acc = 0.0
    for i in range(n):
        acc = tilt * acc + (1.0 - tilt) * x[i]
        y[i] = acc
    y = y / (np.max(np.abs(y)) + 1e-9)
    return (y * 20000).astype(np.int16)


def test_filterbank():
    print("\n--- the mel filterbank ---")
    fb = vp._mel_filterbank()
    check("one row per mel band", fb.shape == (vp.N_MELS, vp.N_FFT // 2 + 1),
          fb.shape)
    check("every band has weight", np.all(fb.sum(axis=1) > 0))
    dense = vp._mel_filterbank(n_mels=64, n_fft=64)
    check("crowded bins do not break the bank", dense.shape == (64, 33))


def test_mfcc_shape():
    print("\n--- mfcc ---")
    x = _voiced(1, 0.9)
    c = vp.mfcc(x)
    check("a full clip yields frames x cepstra",
          c.ndim == 2 and c.shape[1] == vp.N_CEPS and c.shape[0] > 50,
          c.shape)
    check("empty audio yields no frames", vp.mfcc(np.zeros(0, np.int16)).shape
          == (0, vp.N_CEPS))
    check("a clip shorter than a window yields no frames",
          vp.mfcc(np.zeros(100, np.int16)).shape == (0, vp.N_CEPS))


def test_feature():
    print("\n--- feature ---")
    f = vp.feature(_voiced(1, 0.9))
    check("a real clip yields a unit vector", f is not None
          and abs(float(np.linalg.norm(f)) - 1.0) < 1e-9, None if f is None
          else float(np.linalg.norm(f)))
    check("a clip too short to print yields None",
          vp.feature(np.zeros(100, np.int16)) is None)
    check("a loud clip with too few frames yields None",
          vp.feature((np.ones(400, np.int16) * 5000)) is None)


def test_silence_robustness():
    print("\n--- the print ignores silence ---")
    clean = _voiced(1, 0.9)
    padded = np.concatenate([np.zeros(8000, np.int16), clean])
    same = vp.similarity(vp.feature(clean), vp.feature(padded))
    check("a leading pause barely moves the print", same > 0.99, round(same, 3))
    mostly_quiet = np.zeros(16000, np.int16)
    mostly_quiet[8000:8160] = 12000
    check("an almost-silent clip keeps its few frames and still prints",
          vp.feature(mostly_quiet) is not None)
    check("sheer silence yields no print",
          vp.feature(np.zeros(16000, np.int16)) is None)


def test_similarity_orders_voices():
    print("\n--- similarity: same voice beats a different one ---")
    a = vp.feature(_voiced(1, 0.9))
    b = vp.feature(_voiced(2, 0.9))
    c = vp.feature(_voiced(3, 0.2))
    same = vp.similarity(a, b)
    diff = vp.similarity(a, c)
    check("two takes of one voice are close", same > 0.7, round(same, 3))
    check("a different voice is further off", diff < same, round(diff, 3))
    check("the owner is closer than the stranger", same > diff,
          (round(same, 3), round(diff, 3)))


def test_combine():
    print("\n--- combine ---")
    a = vp.feature(_voiced(1, 0.9))
    b = vp.feature(_voiced(2, 0.9))
    one = vp.combine([a, None])
    check("too few good clips yields None", vp.combine([a]) is None)
    check("a None clip does not count as a take", vp.combine([a, None]) is None)
    two = vp.combine([a, b])
    check("two takes average to a unit vector", two is not None
          and abs(float(np.linalg.norm(two)) - 1.0) < 1e-9)
    check("more than the minimum still works", vp.combine([a, b, a]) is not None)
    n = 24
    base = np.zeros(n)
    base[0] = 1.0
    check("opposite takes cancel to None", vp.combine([base, -base]) is None)


def test_profile_roundtrip():
    print("\n--- the profile on disk ---")
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "sub", "voiceprint.json")
        f = vp.feature(_voiced(1, 0.9))
        check("saving makes the parent directory", vp.save_profile(p, f) is True)
        back = vp.load_vector(p)
        check("the print round-trips", back is not None
              and np.allclose(back, f))
        check("a missing file loads as None",
              vp.load_vector(os.path.join(td, "nope.json")) is None)

        old = os.path.join(td, "old.json")
        with open(old, "w") as fh:
            json.dump({"version": 99, "vector": [1.0, 2.0]}, fh)
        check("an old-version print is refused", vp.load_vector(old) is None)

        bad = os.path.join(td, "bad.json")
        with open(bad, "w") as fh:
            fh.write("not json at all")
        check("garbage is refused", vp.load_vector(bad) is None)

        empty = os.path.join(td, "empty.json")
        with open(empty, "w") as fh:
            json.dump({"version": vp.PROFILE_VERSION}, fh)
        check("a print with no vector is refused", vp.load_vector(empty) is None)

        wrong = os.path.join(td, "wrong.json")
        with open(wrong, "w") as fh:
            json.dump({"version": vp.PROFILE_VERSION, "vector": [1.0, 2.0, 3.0]},
                      fh)
        check("a print of the wrong length is refused",
              vp.load_vector(wrong) is None)

        arr = os.path.join(td, "array.json")
        with open(arr, "w") as fh:
            fh.write("[1, 2, 3]")
        check("a non-object print is refused (not a crash)",
              vp.load_vector(arr) is None)
        check("the wrong-length print is smaller than a real one",
              vp.feature(_voiced(1, 0.9)).size == vp.FEATURE_DIM)

        check("saving onto a directory fails soft",
              vp.save_profile(td, f) is False)

    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as td:
        os.chdir(td)
        try:
            check("saving a bare filename (no directory) works",
                  vp.save_profile("vp.json", f) is True)
        finally:
            os.chdir(cwd)


def test_default_profile():
    print("\n--- default_profile ---")
    p = vp.default_profile()
    check("the default lives under a backtalk dir",
          p.endswith(os.path.join(".backtalk", "voiceprint.json")), p)


def test_speaker():
    print("\n--- Speaker: score and check ---")
    owner = vp.feature(_voiced(1, 0.9))
    s = vp.Speaker(owner, threshold=0.0)
    check("a short clip scores None", s.score(np.zeros(100, np.int16)) is None)
    check("a short clip is not the owner", s.check(np.zeros(100, np.int16)) is False)
    s.score(_voiced(1, 0.9))
    check("a real clip records its score", s.last_score != 0.0)
    check("with a zero floor, anyone passes", s.check(_voiced(9, 0.2)) is True)
    strict = vp.Speaker(owner, threshold=2.0)
    check("with an unreachable floor, no one passes",
          strict.check(_voiced(1, 0.9)) is False)


def test_load_speaker():
    print("\n--- load_speaker ---")
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "voiceprint.json")
        check("no profile -> nothing gates", vp.load_speaker(
            {"speaker_profile": p}) is None)
        vp.save_profile(p, vp.feature(_voiced(1, 0.9)))
        sp = vp.load_speaker({"speaker_profile": p})
        check("a saved profile loads", sp is not None and
              sp.threshold == vp.DEFAULT_THRESHOLD)
        spb = vp.load_speaker({"speaker_profile": p,
                               "speaker_threshold": "nonsense"})
        check("a bad threshold falls to the default",
              spb is not None and spb.threshold == vp.DEFAULT_THRESHOLD)
        spc = vp.load_speaker({"speaker_profile": p, "speaker_threshold": 0.5})
        check("a good threshold is honored", spc is not None
              and spc.threshold == 0.5)
        vp.save_profile(p, vp.feature(_voiced(1, 0.9)), threshold=0.83)
        spt = vp.load_speaker({"speaker_profile": p})
        check("the enrollment's own floor is used by default",
              spt is not None and spt.threshold == 0.83)
        spd = vp.load_speaker({"speaker_profile": p,
                               "speaker_threshold": "nonsense"})
        check("a bad config floor falls back to the enrolled one",
              spd is not None and spd.threshold == 0.83)


def test_self_threshold():
    print("\n--- the enrollment chooses its own floor ---")
    a = vp.feature(_voiced(1, 0.9))
    b = vp.feature(_voiced(2, 0.9))
    check("too few takes -> no floor", vp.self_threshold([a, None]) is None)
    check("fit refuses too few takes", vp.fit([a, None]) is None)
    got = vp.fit([a, b])
    check("fit returns a print and a floor", got is not None
          and got[0].shape == (vp.FEATURE_DIM,)
          and vp.SELF_FLOOR <= got[1] <= vp.SELF_CEILING)
    check("a perfectly consistent speaker is capped at the ceiling",
          vp.self_threshold([a, a, a]) == vp.SELF_CEILING)
    z0 = np.zeros(vp.FEATURE_DIM)
    z0[0] = 1.0
    z1 = np.zeros(vp.FEATURE_DIM)
    z1[1] = 1.0
    check("a mismatched pair cannot open the gate wide",
          vp.self_threshold([z0, z1]) == vp.SELF_FLOOR)


def test_enroll():
    print("\n--- enroll ---")
    takes = iter([_voiced(1, 0.9), _voiced(2, 0.9)])
    calls = []

    def record():
        calls.append(1)
        return next(takes)

    vec = vp.enroll(record, prompts=["a", "b"])
    check("two good takes build a print", vec is not None and
          abs(float(np.linalg.norm(vec[0])) - 1.0) < 1e-9)
    check("the print carries a self-chosen floor",
          vec is not None and vp.SELF_FLOOR <= vec[1] <= vp.SELF_CEILING)
    check("record was called once per prompt", len(calls) == 2, calls)

    def one_short():
        return np.zeros(100, np.int16)

    check("takes too short to print build nothing",
          vp.enroll(one_short, prompts=["a", "b"]) is None)

    def one_none_one_good():
        seq = iter([None, _voiced(1, 0.9)])
        return next(seq)

    check("a None take leaves too few good takes",
          vp.enroll(one_none_one_good, prompts=["a", "b"]) is None)


print("=" * 66)
print("VOICEPRINT (classic DSP)")
print("=" * 66)
test_filterbank()
test_mfcc_shape()
test_feature()
test_silence_robustness()
test_similarity_orders_voices()
test_profile_roundtrip()
test_default_profile()
test_speaker()
test_load_speaker()
test_self_threshold()
test_enroll()

print("\n" + "=" * 66)
print("VOICEPRINT OK" if not FAILURES else "VOICEPRINT FAILURES: %s" % FAILURES)
sys.exit(1 if FAILURES else 0)
