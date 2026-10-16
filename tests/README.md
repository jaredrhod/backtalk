# backtalk tests

There is no pytest here on purpose: every test is a standalone script
that prints `ok`/`FAIL` lines and exits nonzero on failure. `run_all.py`
finds them, runs each in its own process, and summarises.

    uv run python tests/run_all.py          the standard suites
    uv run python tests/run_all.py --fast   skip the model-loading suites (CI)
    uv run python tests/run_all.py --all    include the live/e2e suites
    uv run python tests/run_all.py --fast --coverage   measure backtalk/ and
                                                       enforce the floor

Some suites are pure control-flow with a faked mic and transcriber and
run in seconds. Others run the real Whisper decoder, the real brain over
the opencode server, and the real TTS engines, and take minutes -- those
are excluded from `--fast`. `--all` adds the suites that need the live
stack running (brain on 4599, face on 8790).

| script | what it proves |
| --- | --- |
| `test_wake.py` | Wake phrases are recognised however whisper spells or punctuates them, stripped without damaging the command, and the shipped phrases are Latin-only. |
| `test_pipeline.py` | One whole turn offline: a scripted mic through the real endpointer and wake gate, `strip_wake`, and the real sentence batching into a fake mouth and brain. |
| `test_brain.py` | The brain against a fake opencode server: sentence streaming, reasoning-vs-text filtering, empty/timed-out/errored turns, permission routing, slash-commands, usage tallying, the session drain and resume, the screenshot file part a look sends (and that it skips the live lookup), plus the server lifecycle: health probing, spawn and its launch command, the streaming event reader and its reconnect, and the HTTP/SSE wire code. |
| `test_signals.py` | Every file the signal bus writes: state, caption, language, directions, rate limits, reply-done, waveform, the active-window note -- and that no write ever raises. |
| `test_ducking.py` | Spotify duck/restore/debounce logic with the AppleScript bridge stubbed, the off-macOS no-ops, and the permission-result vocabulary. |
| `test_endpointing.py` | The open-mic endpointer opens and closes on schedule, and the hands-free gates (loudness floor, minimum phrase length, and the model's own no-speech score) drop noise before whisper sees it. |
| `test_ptt.py` | Hold-to-talk timing: the release callback fires before transcription, taps are ignored, the time cap still proceeds. |
| `test_ptt_headless.py` | A machine with no keyboard backend (CI) still imports `backtalk.main`, and asking for a key listener fails cleanly instead of taking the line down. |
| `test_ptt_units.py` | ptt.py fully offline with a fake keyboard: key-name resolution (letters, friendly names, options), press/release/repeat, the settle grace window, `wait_press`/`is_held`, and the import path both with a working backend and with none. |
| `test_screen.py` | screen.py offline with a fake display: the one-shot PNG capture the spoken look rides on, the Windows/macOS/Linux focused-window readers (and every way they come up empty), the `_run` helper swallowing a missing tool, the change-only watch loop, and the import path with and without `mss`. |
| `test_encoding.py` | Every text file is clean UTF-8: no BOM, no double-encoding. |
| `test_journal.py` | What the session journal writes, what it refuses to write, and what it does when the brain is slow or broken. |
| `test_vlog.py` | The session log: the terminal line, the timestamped append, the ascii fallback when the console refuses utf-8, the swallowed unwritable log file, and `_init_console`'s Windows and non-Windows paths. |
| `test_config.py` | config.load: the merge over defaults, the file wounds it forgives (missing, BOM, invalid JSON), the derived paths and placeholders, `visible_skills` being named as a no-op, the open-mic greeting swap, `discipline_append`, and `refresh_clock`. |
| `test_bus_park.py` | The bus is never left saying `speaking` by a voice that is gone. |
| `test_turn_errors.py` | A failed turn always parks the bus and tells the listener. |
| `test_live.py` | The live-data path offline: trigger detection, the `_get` network call with its UA and timeout, each source's parsing, the news then wikipedia then duckduckgo fallback order (including a feed that raises), and the cache. |
| `test_main.py` | main.py's helpers and decisions, plus the whole `amain()` live loop driven over fakes (typed turns, console verbs, the spoken look with its screenshot and its no-display failure, the background activity tracker, live mic switches, brain-connect failure, the permission answer, the key, the open-mic fallbacks, wake-word capture, resumed sessions, startup effort, a held-key device failure, a capture dropped after a mode switch, a key pressed mid-reply, and a Ctrl-C out of a turn), the POSIX tty line editor including its odd branches, and the `main()` process boundary: console verbs, usage phrasing, paste assembly, the human permission wording, config writes, reply batching, and the spoken permission gate (allow/deny/details/interrupt/timeout). |
| `test_mouth.py` | mouth.py offline: sentence splitting, the elevenlabs then kokoro then espeak fallback chain, the elevenlabs fetch/decode/feed plumbing, the one long-lived output stream, barge-in cutting, the queue drain, the credential lookup, and orphan temp-dir sweeping. |
| `test_ears_units.py` | ears.py offline: MLX/GPU detection, model resolution and HF cache paths, mic selection and reopen, the audio-failure explanation, `warm()` on both backends (device-probe fallback and the CPU re-raise), the `wait_for_wake` loop, and `transcribe_language` over faked whisper backends. |
| `test_stt_langs.py` | A real English or Tamil sentence survives the trip through Whisper (heavy). |
| `test_espeak_fallback.py` | Every reply reaches the one English voice, and espeak-ng carries it when Kokoro cannot (heavy). |
| `test_e2e.py` | One whole turn: speech -> STT -> brain -> TTS -> caption and language on the bus (live). |
| `test_live_path.py` | A manual driver for the live-data path; pass it a question (live). |

## What these found

Worth keeping, because each one looked fine from the outside and was not:

- An English `stt_prompt` made Whisper transcribe a Japanese sentence as
  "Hello, Memo, please open your intestines." A multilingual prompt fixed
  it -- and a prompt carrying a wake phrase plus a command later made
  Whisper invent that exact turn out of room noise and Seyon executed it.
  The prompt is empty now.
- On Windows, espeak-ng produces pure **silence** for non-Latin scripts
  when the text is passed as argv or on stdin. It only works via `-f` on a
  UTF-8 file, which is why `mouth._stream_espeak` writes a temp file.
- The old language detector returned English for Tamil, so a Tamil voice
  never fired. The detector, the per-language voices, and MMS are removed:
  the ear hears anything, the mouth answers in English, and these suites
  pin that so a language branch cannot come back.
- `brain.WarmBrain.ask_stream` yields nothing at all unless `start()` has
  run first. An E2E test that skips it reports an empty reply and looks
  like a brain failure rather than a harness mistake.
- `ears._hf_cache_dir` built its path as
  `root / "models--" + name.replace("/", "--")`. Python parses `/` before
  `+`, so the `str` on the left raised `TypeError` on **every** platform
  whenever a preferred model was not already downloaded -- the fallback to
  a cached model died before it could run. Parenthesised now, and the
  offline unit test walks both the flat and the snapshot cache layouts.
