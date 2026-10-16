# AGENTS.md — working on backtalk

backtalk is a mouth and ears for an opencode agent. It owns NO
personality: the character lives in whatever AGENTS.md `agent_dir`
points at. backtalk adds only the spoken-delivery discipline
(`config.DISCIPLINE`), which is about the MEDIUM, never the character.

This file is for anyone (human or agent) changing THIS code. Read the
invariants below before you touch anything; they are not style, they are
the reasons the thing works.

## Layout

| path | what it is |
| --- | --- |
| `backtalk/main.py` | the turn loop: key/open-mic, capture, brain, mouth, bus |
| `backtalk/ears.py` | mic capture, VAD endpointing, whisper, the loudness gate |
| `backtalk/brain.py` | the warm opencode session, streaming, live-data fetch |
| `backtalk/mouth.py` | TTS: ElevenLabs -> Kokoro -> espeak-ng fallback chain |
| `backtalk/config.py` | defaults + `DEFAULTS`, the discipline, `CFG` |
| `backtalk/signals.py` | the `.voice_*` state files a face can watch |
| `backtalk/journal.py` | optional session journal written at hangup |
| `backtalk/live.py` | fetch live facts in code for time-sensitive questions |
| `tests/` | standalone scripts, run by `tests/run_all.py` |

`backtalk.json` is the LIVE config and is untracked. `backtalk.json.example`
is the tracked template. Edit the live one only through the app or by
hand with a UTF-8 (no BOM) editor.

## Invariants — break these and the product is broken

1. **Degrade, never mute.** Every path has a fallback. The mic falls
   back to the system default and then rebuilds the audio system
   (`ears._open_mic`); the voice falls back EL -> Kokoro -> espeak-en and
   only then logs "no engine could speak this reply" (`mouth.synth_stream`).
   Never add a path that ends in silence or a raw crash.

2. **English-only replies.** `handle()` forces the turn language to
   English; the `voices` table has one entry; the LANGUAGE block in
   `config.DISCIPLINE` says exactly one language. The ear may HEAR any
   language; the mouth answers only in English. Do not reintroduce
   language routing, MMS, or a per-language voice table.

3. **Whisper gets no text to hallucinate from.** `stt_prompt` is empty on
   purpose; `transcribe()` sets `condition_on_previous_text=False` and
   `no_speech_threshold=0.8`. A wake phrase or command smuggled into the
   prompt once made whisper invent that turn out of room noise and Seyon
   executed it. Keep the prompt empty. `stt_no_speech_prob` is a second
   net: an open-mic transcript the model itself scores as mostly
   non-speech is blanked (`ears.speech_is_confident`).

4. **Wake is English/Latin only.** The virama-tolerant matcher and the
   Tamil wake forms are gone; wake phrases are matched with `\b` and
   `re.escape`. `test_wake.py` asserts the shipped phrases are ASCII.

5. **Ask before acting, by default.** `permission_mode` defaults to
   `"ask"`. `"bypassPermissions"` is an explicit opt-in. Never flip the
   default to auto-approve.

6. **Bound the executors.** Blocking calls go through `main._BLOCKERS`
   or `brain._IO`. Never use `run_in_executor(None, ...)` — the default
   executor is shared and unbounded.

7. **Clean UTF-8, always.** No BOM, no double-encoded text.
   `test_encoding.py` scans every text file and fails the build on both.

8. **The hands-free gates.** Open-mic captures are filtered twice before
   a turn is possible: `ears.speech_is_audible` drops speech below
   `stt_min_rms`, and `ears._min_speech_frames` drops runs shorter than
   `stt_min_speech_ms` (a cough, not a phrase). Both are hands-free only;
   hold-to-talk records exactly what the person chose to record and is
   never gated.

## Build, run, gates

```bash
uv sync                              # install (deps + the dev group)
uv run python -m backtalk.main       # run the voice line
uv run ruff check backtalk tests     # lint (error-level rules)
uv run pyright                       # type gate (strict, minus unknowable
                                     #   third-party types -- see pyproject)
uv run python tests/run_all.py       # every suite
uv run python tests/run_all.py --fast   # skip the model-loading suites (CI)
uv run python tests/run_all.py --all    # include the live/e2e suites
uv run python tests/run_all.py --fast --coverage   # add the coverage floor
```

All of them must be green before a change is done. `--fast --coverage` is
what CI runs; the floor lives in `tests/run_all.py` (`COVERAGE_FLOOR`) and
only ever goes up.

## Test conventions

There is no pytest here on purpose. A test is a standalone script that:

- inserts the repo root on `sys.path` before importing `backtalk.*`;
- prints `ok`/`FAIL` lines and ends with `sys.exit(1 if FAILURES else 0)`;
- fakes the hardware and the models (see `test_ptt.py`,
  `test_endpointing.py`, `test_brain.py`) so it is deterministic and needs
  no mic, no whisper, and no network. `test_brain.py` swaps in a fake
  opencode server and also boots a real localhost HTTP+SSE server to cover
  the wire code; `test_signals.py` points the bus at a temp dir;
  `test_ducking.py` stubs the AppleScript bridge.

When you add a suite, `run_all.py` picks it up automatically. If it
downloads a model or starts an engine, add its filename to `HEAVY`; if it
needs the live stack, add it to `LIVE`.

## Reading the log

`logs/backtalk.log` is the receipts. Landmarks:
`[you]` a turn was heard; `[Seyon]` a reply started; `heard ... no wake
phrase match` a near-miss correctly rejected; `dropped a N-rms blip as
room noise` the loudness gate fired; `CRASH` an unhandled error;
`hung up` a clean exit.
