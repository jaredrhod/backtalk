"""Test config.load: the merge over defaults, the file wounds it forgives
(missing, BOM, invalid JSON), the derived fields (expanded paths, quit
phrases, greeting placeholders), and refresh_clock's per-turn timestamp."""
import json
import importlib
import os
from pathlib import Path
import sys
import tempfile
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtalk import config  # noqa: E402


def check(name, cond, detail=""):
    print("    %s %s%s" % ("ok  " if cond else "FAIL", name,
                           ("  " + str(detail)) if detail else ""))
    if not cond:
        FAILURES.append(name)
    return cond


FAILURES = []


def load_with(payload, *, bom=False, raw=None):
    """Run config.load() against a temp file carrying `payload`."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "backtalk.json"
        if raw is not None:
            path.write_text(raw, encoding="utf-8")
        else:
            text = json.dumps(payload)
            path.write_text(text, encoding="utf-8-sig" if bom else "utf-8")
        with mock.patch.object(config, "CONFIG_PATH", path):
            return config.load()


def test_refresh_clock():
    print("\n--- the clock is substituted per turn ---")
    with mock.patch.object(config.time, "strftime", lambda fmt: "WHEN"):
        out = config.refresh_clock("it is " + config.CLOCK[0] + " now")
    check("the marker became the time", out == "it is WHEN now", out)

    check("text with no marker is returned untouched",
          config.refresh_clock("nothing to see") == "nothing to see")
    check("None uses the discipline", config.CLOCK[0] not in
          config.refresh_clock())


def test_missing_file_is_defaults():
    print("\n--- a missing backtalk.json is the defaults ---")
    with tempfile.TemporaryDirectory() as d:
        with mock.patch.object(config, "CONFIG_PATH",
                               Path(d) / "nope.json"):
            cfg = config.load()
    check("model falls back to the default",
          cfg["model"] == config.DEFAULTS["model"])


def test_invalid_json_is_reported():
    print("\n--- invalid JSON warns and falls back ---")
    printed = []
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "backtalk.json"
        path.write_text("{not json", encoding="utf-8")
        with mock.patch.object(config, "CONFIG_PATH", path), \
             mock.patch("builtins.print",
                        lambda *a, **k: printed.append(" ".join(map(str, a)))):
            cfg = config.load()
    check("kept the defaults",
          cfg["model"] == config.DEFAULTS["model"])
    check("said so out loud",
          any("not valid JSON" in line for line in printed), printed)


def test_bom_is_tolerated():
    print("\n--- a Windows BOM does not break the file ---")
    cfg = load_with({"name": "Seyon"}, bom=True)
    check("the BOM'd file was read", cfg["name"] == "Seyon", cfg["name"])


def test_dicts_merge_and_scalars_override():
    print("\n--- dicts merge, scalars replace ---")
    cfg = load_with({"voices": {"en": "custom_voice"},
                     "elevenlabs": {"model": "my_model"},
                     "speed": 1.25})
    check("the inner key was overridden",
          cfg["voices"]["en"] == "custom_voice", cfg["voices"])
    check("the sibling defaults survived",
          cfg["elevenlabs"]["key_slot"] == "DEFAULTS".lower()
          or cfg["elevenlabs"].get("key_slot")
          == config.DEFAULTS["elevenlabs"]["key_slot"], cfg["elevenlabs"])
    check("a scalar replaced the default", cfg["speed"] == 1.25)


def test_visible_skills_warning():
    print("\n--- visible_skills is called out, not silently ignored ---")
    printed = []
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "backtalk.json"
        path.write_text(json.dumps({"visible_skills": True}), encoding="utf-8")
        with mock.patch.object(config, "CONFIG_PATH", path), \
             mock.patch("builtins.print",
                        lambda *a, **k: printed.append(" ".join(map(str, a)))):
            config.load()
    check("the dead option is named",
          any("visible_skills" in line for line in printed), printed)


def test_derived_fields():
    print("\n--- paths, quit phrases, and greeting placeholders ---")
    home = os.path.expanduser("~")
    cfg = load_with({
        "name": "Seyon",
        "agent_dir": "~",
        "extra_dirs": ["~/notes"],
        "signals_dir": "~",
        "thinking_sound": "assets/thinking.wav",
        "greeting": "Hi {name}, hold {ptt_key}.",
        "quit_phrases": ["bye now"],
    })
    check("agent_dir expanded", cfg["agent_dir"] == home, cfg["agent_dir"])
    check("extra_dirs expanded",
          cfg["extra_dirs"] == [os.path.expanduser("~/notes")],
          cfg["extra_dirs"])
    check("signals_dir expanded", cfg["signals_dir"] == home)
    check("relative thinking sound resolves into the repo",
          cfg["thinking_sound"] == str(config.REPO / "assets/thinking.wav"),
          cfg["thinking_sound"])
    check("a custom quit phrase is kept",
          cfg["quit_phrases"] == ("bye now",), cfg["quit_phrases"])
    check("name and key filled into the greeting",
          cfg["greeting"] == "Hi Seyon, hold the home key.", cfg["greeting"])

    abs_sound = str(Path.cwd() / "think.wav")
    cfg2 = load_with({"thinking_sound": abs_sound})
    check("an absolute thinking sound is left alone",
          cfg2["thinking_sound"] == abs_sound, cfg2["thinking_sound"])

    cfg3 = load_with({"name": ""})
    check("an empty name falls back",
          "goodbye assistant" in cfg3["quit_phrases"], cfg3["quit_phrases"])


def test_open_mic_greeting():
    print("\n--- the open-mic greeting can stand in ---")
    cfg = load_with({"name": "Seyon", "mic_mode": "open",
                     "greeting_open_mic": "Hands free, {name}.",
                     "greeting": "Hold the key."})
    check("the open-mic line replaced the greeting",
          cfg["greeting"] == "Hands free, Seyon.", cfg["greeting"])

    cfg2 = load_with({"name": "Seyon", "mic_mode": "open",
                      "greeting_open_mic": ""})
    check("with no open-mic line the normal greeting stays",
          "{ptt_key}" not in cfg2["greeting"]
          and cfg2["greeting"] == str(config.DEFAULTS["greeting"]).replace(
              "{name}", "Seyon").replace(
              "{ptt_key}", "the home key"), cfg2["greeting"])


def test_discipline_append_is_applied():
    print("\n--- discipline_append reaches DISCIPLINE at import ---")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "backtalk.json"
        path.write_text(json.dumps({"discipline_append": "PLAIN WORDS ONLY"}),
                        encoding="utf-8")
        old = os.environ.get("BACKTALK_CONFIG")
        os.environ["BACKTALK_CONFIG"] = str(path)
        try:
            importlib.reload(config)
            applied = "PLAIN WORDS ONLY" in config.DISCIPLINE
        finally:
            if old is None:
                os.environ.pop("BACKTALK_CONFIG", None)
            else:
                os.environ["BACKTALK_CONFIG"] = old
            importlib.reload(config)
    check("the note was folded in", applied)


def test_bad_shapes_are_safe():
    print("\n--- a JSON array/scalar or a string extra_dirs falls back ---")
    cfg = load_with([1, 2, 3])
    check("a non-object config uses the defaults",
          cfg["model"] == config.DEFAULTS["model"], cfg["model"])
    cfg2 = load_with({"extra_dirs": "~/vault"})
    check("a string extra_dirs is not exploded into characters",
          cfg2["extra_dirs"] == [], cfg2["extra_dirs"])
    cfg3 = load_with({"extra_dirs": 5})
    check("a scalar extra_dirs is ignored",
          cfg3["extra_dirs"] == [], cfg3["extra_dirs"])


def test_numeric_settings_are_coerced():
    print("\n--- numeric knobs are coerced; a bad one falls back ---")
    cfg = load_with({"speed": "1.25", "opencode_port": "4600",
                     "turn_timeout": True, "stt_min_rms": "loud",
                     "prebuffer_s": "0.4"})
    check("a number written as a string is honoured",
          cfg["speed"] == 1.25, cfg["speed"])
    check("the prebuffer knob is coerced too",
          cfg["prebuffer_s"] == 0.4, cfg["prebuffer_s"])
    check("an int default yields an int",
          cfg["opencode_port"] == 4600
          and isinstance(cfg["opencode_port"], int), cfg["opencode_port"])
    check("a bool for a number falls back to the default",
          cfg["turn_timeout"] == config.DEFAULTS["turn_timeout"],
          cfg["turn_timeout"])
    check("an unreadable number falls back to the default",
          cfg["stt_min_rms"] == config.DEFAULTS["stt_min_rms"],
          cfg["stt_min_rms"])


def test_path_settings_tolerate_non_strings():
    print("\n--- a non-string path setting cannot crash the load ---")
    cfg = load_with({"agent_dir": 5, "thinking_sound": ["x"],
                     "barehands_state_dir": 7, "extra_dirs": [1, "~/notes"]})
    check("a non-string agent_dir falls back to home",
          cfg["agent_dir"] == os.path.expanduser("~"), cfg["agent_dir"])
    check("a non-string thinking_sound is emptied",
          cfg["thinking_sound"] == "", cfg["thinking_sound"])
    check("a non-string barehands dir is emptied",
          cfg["barehands_state_dir"] == "", cfg["barehands_state_dir"])
    check("a non-string extra_dir becomes empty, not a crash",
          cfg["extra_dirs"] == ["", os.path.expanduser("~/notes")],
          cfg["extra_dirs"])


def test_elevenlabs_must_stay_a_table():
    print("\n--- elevenlabs is forced back to a table ---")
    cfg = load_with({"elevenlabs": None})
    check("a null elevenlabs becomes the default table",
          isinstance(cfg["elevenlabs"], dict)
          and cfg["elevenlabs"]["enabled"] is False, cfg["elevenlabs"])


print("=" * 66)
print("CONFIG")
print("=" * 66)

test_refresh_clock()
test_missing_file_is_defaults()
test_invalid_json_is_reported()
test_bom_is_tolerated()
test_dicts_merge_and_scalars_override()
test_visible_skills_warning()
test_derived_fields()
test_open_mic_greeting()
test_discipline_append_is_applied()
test_bad_shapes_are_safe()
test_numeric_settings_are_coerced()
test_path_settings_tolerate_non_strings()
test_elevenlabs_must_stay_a_table()

print("\n" + "=" * 66)
print("CONFIG OK" if not FAILURES else "CONFIG FAILURES: %s" % FAILURES)

sys.exit(1 if FAILURES else 0)
