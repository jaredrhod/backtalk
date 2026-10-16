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
"""Configuration — backtalk.json in the repo root, merged over defaults.

backtalk deliberately owns NO personality. Your agent's identity lives in
the AGENTS.md of whatever folder `agent_dir` points at — backtalk just
gives that agent a mouth and ears. The only voice-related instruction it
adds is the spoken-delivery discipline below, which is about the MEDIUM
(writing for the ear), never the character.
"""
import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# One install, more than one assistant. Point BACKTALK_CONFIG at a different
# JSON file and you get a second agent (its own name, voice, folder and
# greeting) without a second copy of the code. A launcher exports it; nothing
# else changes.
CONFIG_PATH = Path(os.environ.get("BACKTALK_CONFIG") or (REPO / "backtalk.json"))

DEFAULTS = {
    # The folder whose AGENTS.md (or CLAUDE.md) defines WHO your agent is.
    # The voice session runs there, so it's the same assistant as your
    # terminal sessions — same name, same personality, same memory.
    "agent_dir": "~",
    # Display name, used in logs and to build the quit phrases
    # ("goodbye <name>" hangs up). Match your agent's actual name.
    "name": "Assistant",
    # The brain, in opencode's "provider/model-id" form. The FULL id on
    # purpose: opencode model ids are namespaced by the provider AND often
    # carry their own prefix, and a bare name is ambiguous. List yours with
    # `opencode models <provider>`. The fast tier is most of the speed
    # difference people ask about; a deep-work model makes every reply
    # noticeably slower and burns usage doing it.
    "model": "nvidia/nvidia/nemotron-3-super-120b-a12b",
    # The deep-work model for the voice console's "switch to the deep
    # model" command ("back to the fast model" returns to "model" above).
    # Full id ON PURPOSE, same reasoning as "model". The switch lasts one
    # session and is always spoken; this default never moves by itself.
    "deep_model": "nvidia/nvidia/nemotron-3-super-120b-a12b",
    # Which opencode agent VARIANT the voice session starts on, if your
    # agent defines any (a named preset selected with a variant field).
    # Empty means the agent's own default. The voice console's variant
    # verb flips it live, and that also saves.
    "variant": "",
    # Tool permissions for the voice session. "ask" is the default ON
    # PURPOSE (safety is opt-out, never opt-in): when the agent wants a
    # gated tool (write a file, run a real command), it ASKS OUT LOUD
    # and waits. Answer by voice or by typing. An EXACT yes approves
    # ("yes", "yeah", "go ahead", "approved"...); anything else denies,
    # and your words are passed back to the agent as the reason, so
    # "no, put it in drafts instead" actually steers it. No answer
    # within 75 seconds means no, out loud. Most read-only work passes
    # without asking; anything that changes things asks.
    # "bypassPermissions" is AUTO-APPROVE: the agent acts without
    # asking, exactly like a terminal session with approvals off.
    # (Not to be confused with hands-free LISTENING, which is about
    # the microphone: see mic_mode below.) Never hand-edit this file
    # to switch: tell your agent to change it (takes effect next
    # launch), or say "stop asking for permission" (then "confirm")
    # or "start asking again" inside a voice session for an immediate
    # flip that also saves. The legacy value "default" now
    # behaves as "ask" (a headless voice session could never render
    # the terminal prompt it promised).
    "permission_mode": "ask",
    # Extra folders the agent may access beyond agent_dir (e.g. your
    # notes vault). Absolute paths or ~ paths.
    "extra_dirs": [],
    # Where a finished voice session writes itself down: one markdown file
    # per session, named by timestamp. Empty means OFF, and empty is the
    # default because this writes outside backtalk's own folder and that
    # should be a decision rather than a surprise. Point it at your vault's
    # daily-notes folder and every hangup leaves a record behind.
    #
    # backtalk writes the transcript and, if journal_summary is on, a plain
    # summary from the brain. It does not decide what is worth remembering:
    # that judgement belongs to the agent whose AGENTS.md this points at.
    "journal_dir": "",
    # Whether to spend one model call at hangup on a summary. Worth it on a
    # real session, wasted on a two-line one, and always bounded -- if the
    # brain is slow the entry is still written, just with less in it.
    "journal_summary": True,
    # Screen awareness, in two halves. OFF by default; both are opt-in.
    #
    # screen_enabled: sample which app/window is focused every
    # screen_track_interval seconds and record the LABEL in the session
    # journal (and on the signal bus, so a face can show it). Local only --
    # a line of text like "code.exe -- main.py", never a picture.
    #
    # The spoken "look at my screen" command works regardless of this flag:
    # it only ever fires when you say it, and sends ONE screenshot to your
    # own brain so it can answer about what is on the display. Set the
    # interval to 0 to turn the periodic sampling off while keeping the look.
    "screen_enabled": False,
    "screen_track_interval": 30.0,
    # Check-ins: the assistant starting the conversation itself. OFF by
    # default, and off is the honest default -- a voice line that talks when
    # nobody asked should be something you turned on, never a surprise.
    #
    # When on, after a full check_in_interval of quiet (no turn from you, no
    # reply in flight) it may say one short line of its own. It never talks
    # over a live turn, and any input from you resets the clock: it is a
    # companion's "still here?" between silences, not a timer that interrupts.
    "check_in_enabled": False,
    "check_in_interval": 900.0,
    # A little randomness on the cadence, as a FRACTION of the interval
    # (0.15 = up to 15% longer). A metronome check-in feels like a machine;
    # a human looks up at slightly different gaps. 0 disables the jitter.
    "check_in_jitter": 0.15,
    # Local hours when the line stays quiet no matter what: [start, end],
    # 24-hour, wrapping past midnight. The default is overnight (23:00
    # until 07:00) so a late night does not get a cheery "still there?".
    # An empty list, or start == end, means no quiet hours at all.
    "check_in_quiet_hours": [23, 7],
    # Focused apps that HOLD check-ins: if the active window's app or title
    # contains any of these (case-insensitive), it stays quiet -- for a
    # meeting, a game, a full-screen video. Empty means never hold. Only
    # consulted when the screen module can read the focused window.
    "check_in_hold_apps": [],
    # How long "stay quiet for a while" (with no time named) silences
    # check-ins, in seconds. You can always name a time ("quiet for twenty
    # minutes") or say "you can check in again" to end it early.
    "check_in_default_quiet": 3600.0,
    # After the line finishes speaking, keep listening without the wake word
    # for this many seconds, so answering a check-in (or any reply) is a
    # normal back-and-forth instead of "Hey Seyon ..." every time. 0 turns
    # it off and the wake word is required for every utterance, as before.
    "followup_window": 0.0,
    # A folder where a spoken "remember this ..." drops one kept line, one
    # dated markdown file per day. Empty (the default) turns the spoken
    # command off. Point it at a notes folder -- your vault's inbox is the
    # obvious one -- and "remember that the wifi password is ..." is
    # captured in a breath, without a round trip to the model.
    "remember_dir": "",
    # Length of a spoken focus stretch ("focus for twenty minutes", "start a
    # pomodoro", "do not disturb for an hour") when no length is named, in
    # minutes. During it the line stays quiet, then says one line when it is
    # time to take a break. Reuses the check-in pause and the reminders, so
    # "you can check in again" or a bare "stop" ends it early.
    "focus_minutes": 25.0,
    # Greet the FIRST session of each day with one short spoken briefing
    # drawn from your notes -- what is on your plate, what you last worked
    # on. OFF by default: a session that launches straight into a speech is
    # a surprise, so the person turns it on.
    "daily_briefing": False,
    # Gentle self-care nudges folded into a check-in: a battery well down
    # and off the charger, a nearly-full disk, a long unbroken session. OFF
    # by default, and only ever spoken through a check-in, never on its own.
    "care_enabled": False,
    # The opencode HTTP server backtalk talks to. It starts its own on
    # this port if nothing is listening, and REUSES a running server if
    # something already is — so starting opencode yourself first is fine
    # and just saves a spawn. Only ever bound to loopback.
    "opencode_port": 4599,
    # The opencode executable, when it isn't on PATH. "" means "find it
    # on PATH" (the normal case). Set an absolute path only if backtalk
    # can't see it — e.g. "C:\\Users\\you\\AppData\\Roaming\\npm\\opencode.cmd".
    "opencode_bin": "",
    # Hold-to-talk key. Named keys ("home", "f13", "right_alt", ...)
    # or a single character.
    "ptt_key": "home",
    # The microphone mode. "ptt" (push to talk, the default and the
    # recommendation): the mic is closed except while the key is held,
    # so room audio and your own speakers can never trigger the agent.
    # "open" (hands-free listening): always listening with voice
    # detection; a video, music with vocals, or another person in the
    # room CAN trigger it, and with open speakers it can hear itself
    # (headphones recommended). The key still works in hands-free
    # listening: it interrupts, and holding it always gets you heard.
    # Switch live by voice: "go hands free" / "push to talk mode"
    # (the switch saves itself here). The --open-mic launch flag
    # forces "open" for one session.
    "mic_mode": "ptt",
    # Playback speed for the built-in voice: 1.0 is Kokoro's native
    # pace, 1.15 is noticeably brisker, 0.9 is slower. Kokoro's own
    # pipeline implements it, so quality holds across sane values
    # (roughly 0.7 to 1.5). ElevenLabs pace lives in the master chain's
    # atempo instead. (Grew out of a community proposal, issue #1.)
    "speed": 1.0,
    # How much synthesized audio to bank BEFORE playback starts, in
    # seconds. It is a head start against a synthesizer that cannot always
    # keep up with realtime: too small and the voice can gape mid-sentence,
    # too large and the reply starts late. 0.5 is the responsive default;
    # raise it toward 0.75-1.0 only if you hear gaps on a weak machine.
    "prebuffer_s": 0.5,
    # Resume the previous conversation on launch. OFF by default: a
    # fresh session every launch is the predictable behavior. Set true
    # and backtalk saves the session id after every completed turn
    # (signals_dir/.backtalk_session) and reattaches to it at the next
    # launch, so killing the window stops costing you the conversation.
    # A resume that fails falls back to a fresh session and says so in
    # the log. (Grew out of the same community proposal, issue #1.)
    "resume_last_session": False,
    # Publish your usage on the signal bus so a face can draw it. OFF by
    # default and deliberately so: this is your own account spend, and the
    # faces this feeds are frequently on a stream or a shared screen.
    # Nothing is collected at all while this is false. (Community fix,
    # ai-visualizer issue #1.)
    "show_usage": False,
    # Reasoning effort for the voice session: "" inherits the model's
    # default; "low" / "medium" / "high" / "max" applies at launch.
    # Saying "set effort to X" in a voice session saves itself here.
    "effort": "",
    # The opencode agent that answers a voice turn. "build" is the default
    # agent; "plan" is the read-only one. List yours with `opencode agent`.
    "agent": "build",
    # The voice (Kokoro, local, free). bm_lewis is the proven default —
    # British male, the butler register. Others: bm_george, bm_daniel,
    # bm_fable, am_michael, af_heart... The first letter picks the
    # language pipeline (a=American, b=British, e/f/h/i/j/p/z = other
    # languages), so keep voice and accent matched.
    "voice": "bm_lewis",
    # Per-language voices. "voice" above stays the English default.
    # The mouth reads the language your reply is written in and loads the
    # matching voice. Keys are Whisper's own language codes; the value is
    # a Kokoro voice name whose first letter IS the pipeline to load.
    #
    # Seyon answers in English only, so this table has one entry:
    # English -- and every other language is deliberately absent,
    # because no other language may ever reach the speaker.
    "voices": {
        "en": "bm_lewis",    # British, the butler register
    },
    # Speech recognition (faster-whisper, local, free).
    # NOTE: the plain multilingual models, not the ".en" ones. The ".en"
    # variants are English-only, which is what used to make Seyon
    # mishear every other language as English and answer in English.
    # tiny / base / small / medium — small is the accuracy/speed sweet
    # spot on a normal machine, and it auto-detects the language.
    "stt_model": "small",
    # Used instead of stt_model once its weights are already on this
    # machine. Set to a model you have downloaded; until then the smaller
    # one is used so the first utterance never waits on a fetch. It
    # defaults to the same model as stt_model on purpose: a bigger cached
    # model transcribes more accurately but SLOWER, and losing that much
    # per-turn latency for a voice line is a bad trade. Set it to "medium"
    # only if you would rather have accuracy than speed.
    "stt_model_if_cached": "small",
    # Fetch live facts in code for time-sensitive questions, so the model
    # never has to answer about the present out of its own memory.
    "live_data": True,
    # How long one turn may run before it is abandoned. A provider that
    # stops streaming used to hold the turn for 600s, which reads as a
    # dead assistant; this gives up, resets the turn, and stays usable.
    "turn_timeout": 150,
    # Empty on purpose. Any initial_prompt seeds whisper: the previous
    # "Vijay Seyon open Notepad YouTube." hallucinated a wake phrase +
    # command out of room noise (Seyon opened YouTube with nobody
    # speaking), and even the harmless "Vijay, Notepad, YouTube,
    # Whisper." made it invent "Hi, I'm Vijay... welcome to my
    # channel... YouTube" loops every few seconds in a quiet room. The
    # English-only measurement showed the prompt changed nothing for
    # real accuracy, so the bias is pure downside -- names survive
    # without it ("Vijay" came through either way).
    "stt_prompt": "",
    # "auto" uses CUDA when present, otherwise CPU. int8 keeps CPU fast.
    "stt_device": "auto",
    "stt_compute": "int8",
    # Pin whisper to one language instead of letting it detect. Empty (the
    # default) means detect, which is what you want unless a language keeps
    # being mistaken for a neighbour ("pt" read as "es"). Set "en" only when
    # you want English forced no matter what is said. Accepts the two-letter
    # codes whisper uses.
    "stt_language": "",
    # Hands-free loudness floor, in int16 RMS, for a captured utterance to
    # reach whisper at all. VAD passes a fan, a music bed, or a TV tail as
    # "speech shaped", and whisper then invents sentences over it
    # ("Thanks for watching!"); this is the loudness test VAD skips. Only
    # the OPEN-MIC path is gated -- hold-to-talk records exactly what the
    # person chose to record. 0 turns the gate off.
    "stt_min_rms": 200,
    # Minimum speech in a hands-free utterance before whisper sees it, in
    # milliseconds. A cough, a door, or a one-word splash of noise that VAD
    # calls speech is shorter than a real command; requiring a plausible
    # phrase length drops those without touching hold-to-talk. 0 keeps the
    # old 240ms floor.
    "stt_min_speech_ms": 400,
    # The model's own no_speech probability at or above which a transcript
    # is treated as non-speech and blanked. Whisper sometimes prints
    # confident text ("Thanks for watching!") over a music bed it
    # nonetheless scores as mostly silence; this is the check that believes
    # that score. OPEN-MIC only -- hold-to-talk records what you chose.
    # 0 disables it.
    "stt_no_speech_prob": 0.5,
    # Partial transcription: while you are still talking, transcribe the
    # audio so far on an interval and publish it to the signal bus
    # (.voice_heard), so a face can caption the words as they are heard
    # instead of after the whole turn. Off by default -- each partial is a
    # whisper call, real CPU on a machine that may not want the extra
    # work. partial_stt_ms is how often a partial fires, in milliseconds.
    "partial_stt": False,
    "partial_stt_ms": 700,
    # Barge-in: if true, listen while speaking (open mic) so a new
    # utterance interrupts the reply as soon as it is detected.
    "barge_in": False,
    # Wake word: in open-mic mode, wait for one of these phrases before
    # treating what was said as a command. Say the phrase alone and the
    # next utterance is the command; say it with the command in one breath
    # ("hey seyon, what's the weather") and that counts too. Phrase
    # variants catch the ways whisper spells "Seyon", and each carries a
    # greeting on purpose: the bare name is too easy to trip over in
    # ordinary speech ("...so Sean said...").
    "wake_word": False,
    "wake_word_phrases": [
        "hey seyon", "hey sayon", "hey sean", "hey saiyan",
        "hey siyon",
    ],
    # The microphone to record from, matched by NAME. "" means whatever
    # the OS calls the default input, which is right on most machines.
    #
    # Set a real device name to PIN the mic, so a headset connecting for
    # OUTPUT cannot steal your input -- which also keeps a Bluetooth
    # headset in high-quality A2DP instead of dropping it to the
    # narrowband call profile mid-sentence, degrading what you hear at
    # the same moment it takes your voice.
    #
    # A name and never an index: indices shift every time a device
    # connects or disconnects, the exact event this setting exists to
    # survive. Exact name wins, then the first case-insensitive
    # substring. A name matching nothing falls back to the default and
    # logs the inputs it did find; the mic degrades, it never goes mute.
    #
    # NOT "stt_device" below, which is the Whisper COMPUTE device.
    "mic_device": "",
    # Optional premium voice: ElevenLabs on YOUR key. The key NEVER
    # goes in a file: it's read from the macOS Keychain (item
    # `backtalk-elevenlabs`) or Linux secret-tool, with the
    # ELEVENLABS_API_KEY env var as last-resort fallback — see
    # mouth._get_elevenlabs_key for the seeding one-liners. Kokoro
    # remains the automatic fallback, so the voice degrades instead of
    # going mute if the cloud fails. Needs ffmpeg on the PATH.
    "elevenlabs": {
        "enabled": False,
        "voice_id": "",
        # Purely for you. Voice IDs are unreadable six months later, so put
        # the human name here; nothing reads it.
        "voice_note": "",
        "model": "eleven_turbo_v2_5",
        # Which OS credential-store entry holds the key. Change it if you
        # already keep an ElevenLabs key under a name of your own rather
        # than seeding a second copy of the same secret.
        "key_slot": "backtalk-elevenlabs",
        # Local mastering: ElevenLabs' site previews are mastered demo
        # clips and the raw API never matches them. This chain closes
        # the gap: presence lift, light chest, broadcast compression,
        # limiter. atempo is the one pace dial (1.0 = native).
        "master": ("atempo=1.12,highpass=f=70,"
                   "equalizer=f=3200:t=q:w=1.2:g=3.5,"
                   "equalizer=f=140:t=q:w=1:g=1.5,"
                   "acompressor=threshold=-18dB:ratio=2.5:attack=8:"
                   "release=120:makeup=4dB,alimiter=limit=0.95"),
    },
    # Where the signal-bus files are written (.voice_state,
    # .voice_waveform, .voice_loading_pid) — anything can watch them;
    # visualizers pair with this contract. Default: the repo root.
    "signals_dir": "",
    # THE BAREHANDS SEAM: point this at a barehands checkout's state/
    # folder and its on-screen ring becomes your agent's face — it
    # breathes while idle, spins while thinking, pulses with the voice.
    # (github.com/jaredrhod/barehands)
    "barehands_state_dir": "",
    # THE BAREHANDS HANDS: the board's own HTTP server, so spoken phrases
    # like "put the plan on the board" and "clear the board" reach the
    # glass (present, add_card, explode, clear, ...). Default matches
    # barehands' port; set "" to turn a box with no board fully quiet.
    "board_url": "http://127.0.0.1:8794",
    # OWNER-ONLY WAKE: a small voiceprint learned on this machine, so the
    # room's voices cannot wake the agent. speaker_id turns the gate on;
    # it does nothing until a print is enrolled by running the line with
    # --enroll. speaker_profile is where the print is kept ("" = a file
    # under ~/.backtalk). speaker_threshold is the cosine floor a clip
    # must clear to count as you; leave it null to use the floor the
    # enrollment chose from how consistent your own takes were, or set a
    # number to override it (higher = stricter). Classic DSP, not a
    # security boundary.
    "speaker_id": False,
    "speaker_profile": "",
    "speaker_threshold": None,
    "speaker_enroll_seconds": 3.0,
    # Sound played while the agent thinks, so a long pause never reads as
    # a dead line. The bundled one ships in assets/; a relative path
    # resolves against this repo. Set "" to think in silence.
    "thinking_sound": "assets/thinking.wav",
    # Spoken lines. {name} is replaced with "name" above.
    "greeting": "Voice line online. Hold {ptt_key} and talk to me.",
    # Spoken instead of "greeting" when mic_mode is "open", where telling
    # someone to hold a key is wrong. Leave "" to use "greeting" for both.
    "greeting_open_mic": "",
    "signoff": "Voice line closing. I'll be here when you need me.",
    # Appended to the spoken-delivery discipline below. The discipline covers
    # the MEDIUM (write for the ear, no markdown, keep it short); your agent's
    # AGENTS.md covers the character. Use this for a note that belongs to
    # neither, e.g. a rule that only applies when it is speaking.
    "discipline_append": "",
}

# The spoken-delivery discipline — the MEDIUM half of what used to be a
# persona. The CHARACTER half deliberately is not here: it's whatever
# lives in the agent_dir's AGENTS.md. One identity, one place.
DISCIPLINE = (
    "VOICE SESSION (your reply is spoken aloud through a TTS engine, "
    "not displayed): you are SPEAKING, in your own voice and "
    "personality — your AGENTS.md is who you are. The TTS engine "
    "PERFORMS your punctuation, so write like a performance, never "
    "like a memo: contractions always, punchy conversational "
    "sentences, and if a line could open a quarterly report, rewrite "
    "it like you're telling a friend. Keep replies to a few short "
    "sentences; go longer only when the question genuinely needs it. "
    "No markdown, no lists, no code blocks, no emoji, no URLs. Say "
    "numbers the way a human says them out loud — never raw figures "
    "or symbols. NEVER SPEAK A FILE PATH: say the file, not its "
    "address. 'the config' or 'ears dot py', never a string of "
    "slashes and folder names read one by one — it is unbearable "
    "aloud and carries no meaning by ear. Same for URLs and long "
    "ids: name the thing, not the address. "
    "Skip any startup sequence; answer directly. "
    "VOICE CONSOLE FACTS, answer from these whenever the person asks "
    "you to change a voice-line setting: this session is controlled "
    "by exact spoken phrases, never by you. Permissions: 'stop "
    "asking for permission' (then 'confirm'), or 'start asking "
    "again'. Microphone: 'go hands free', or 'push to talk mode'. "
    "Also: 'clear the session', 'compact the session', 'switch to "
    "the deep model', 'back to the fast model', 'set effort to low' "
    "(or medium, high, max), and 'usage report'. You cannot flip "
    "these live yourself, so when asked, give the person the exact "
    "phrase to SAY. Editing backtalk.json only changes the default "
    "for the NEXT launch."
)

DISCIPLINE = DISCIPLINE + " " + (
    "LANGUAGE. You speak exactly one language: English. Whatever "
    "language he writes or speaks to you in -- Tamil, Hindi, anything "
    "-- answer in English, every time. A reply in any other language "
    "is a failure, even when the content is right, and whatever "
    "language he used does not change this. Never announce that you "
    "are switching, never name the language, and never ask which "
    "language he prefers. Write it for the ear: short spoken "
    "sentences, no lists, no markdown, and no spelling out letters."
)

CLOCK = ("__CLOCK__", "%A, %d %B %Y at %H:%M")


def refresh_clock(text: str | None = None) -> str:
    """The discipline with the current date and time substituted in.

    The timestamp used to be formatted at import, which meant a process left
    running overnight kept telling the model it was still yesterday. It is
    now resolved every turn (brain.WarmBrain._system), so a machine that
    runs for days stays honest about what day it is. Kept as a replace
    rather than a rebuild so any hand-added discipline text is preserved.
    """
    src = DISCIPLINE if text is None else text
    if CLOCK[0] in src:
        return src.replace(CLOCK[0], time.strftime(CLOCK[1]))
    return src


DISCIPLINE = DISCIPLINE + " " + (
    "LIVE INFORMATION, this is not optional. Right now it is "
    + CLOCK[0]
    + " local time, and that is the only source of truth for what day it "
    "is. Your training does not cover the present, so never answer a "
    "question about the present out of your own memory. The assistant "
    "fetches live facts for you before you answer anything time "
    "sensitive, and those facts arrive attached to the question, marked "
    "as fetched just now. When they arrive, answer only from them and "
    "speak the source's name so the person knows how fresh it is. When "
    "no such facts arrive, nothing was looked up: answer the question "
    "normally, and never mention a lookup, a fetch, live data, or the "
    "news, because none of that happened. If the question is a vague "
    "follow up that refers to something you were just talking about, or "
    "is too vague to answer, simply ask what he means in plain words. "
    "Never say the words live lookup or live data out loud. Never "
    "invent a figure, a date, a quote, or a source."
)


def _expand(p) -> str:
    # A non-string (a number, a list, a null) is a config slip, not a path:
    # return "" instead of letting expanduser raise at import, which would
    # take the whole voice line down over one bad character in the file.
    return os.path.expanduser(p) if isinstance(p, str) else ""


def _num(value, default):
    """Read a numeric setting, keeping the default's type.

    A number written as a string ("1.25") is honoured -- hand-edited JSON
    often is one. A bool, or anything that will not read as a number, falls
    back to the default, so a typo in backtalk.json blinds a single knob
    instead of crashing the voice line the moment the value is read (the
    mic loop reads several of these live)."""
    if isinstance(value, bool):
        return default
    try:
        return int(value) if isinstance(default, int) else float(value)
    except (TypeError, ValueError):
        return default


def load() -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))          # deep copy
    try:
        # utf-8-sig, not utf-8: Windows editors love to save a BOM, and a
        # BOM is not JSON, so the file would otherwise be reported as
        # invalid on the one character that isn't wrong.
        user = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
        if not isinstance(user, dict):
            print("[config] backtalk.json must be a JSON object "
                  "(a {...} of settings) — using defaults", flush=True)
            user = {}
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    except FileNotFoundError:
        pass
    except ValueError as e:
        print(f"[config] backtalk.json is not valid JSON ({e}) — "
              f"using defaults", flush=True)
    # Normalise the risky shapes ONCE, here, so no reader has to defend
    # itself. Every numeric knob becomes the type its default is (or falls
    # back), and the ElevenLabs settings are guaranteed to stay a table --
    # a user writing `"elevenlabs": null` would otherwise turn every
    # CFG["elevenlabs"].get(...) into an AttributeError at speech time.
    for _key, _default in DEFAULTS.items():
        if isinstance(_default, (int, float)) and not isinstance(_default, bool):
            cfg[_key] = _num(cfg.get(_key, _default), _default)
    if not isinstance(cfg.get("elevenlabs"), dict):
        cfg["elevenlabs"] = json.loads(json.dumps(DEFAULTS["elevenlabs"]))
    # A privacy option that silently does nothing is worse than having none,
    # so visible_skills was removed rather than left looking like protection.
    # It promised to keep skill DESCRIPTIONS out of the session's system
    # prompt, and opencode exposes no per-session way to do that (only
    # permission.skill, which gates CALLING a skill, not disclosing it).
    # If backtalk.json still sets it, say so plainly instead of ignoring it.
    if cfg.get("visible_skills") is not None:
        print("[config] visible_skills is set in backtalk.json but is NOT "
              "implemented: skill descriptions are still visible to the "
              "session. Remove the key, it does nothing.", flush=True)
    cfg["agent_dir"] = _expand(cfg.get("agent_dir")) or os.path.expanduser("~")
    # A string here would be iterated character by character, quietly
    # granting ~ and / (and every letter) as accessible dirs; only a real
    # list is honoured.
    _dirs = cfg.get("extra_dirs")
    cfg["extra_dirs"] = [_expand(d) for d in _dirs] if isinstance(_dirs, list) else []
    cfg["signals_dir"] = _expand(cfg.get("signals_dir", "")) or str(REPO)
    cfg["barehands_state_dir"] = _expand(cfg.get("barehands_state_dir", ""))
    # Empty stays empty: that is how journaling is switched off.
    cfg["journal_dir"] = _expand(cfg.get("journal_dir", ""))
    # Empty stays empty: that is how the spoken "remember this" is switched
    # off.
    cfg["remember_dir"] = _expand(cfg.get("remember_dir", ""))
    thinking = _expand(cfg.get("thinking_sound", ""))
    if thinking and not os.path.isabs(thinking):
        thinking = str(REPO / thinking)
    cfg["thinking_sound"] = thinking
    name = str(cfg.get("name") or "Assistant")
    low = name.lower()
    cfg["quit_phrases"] = tuple(cfg.get("quit_phrases") or (
        f"goodbye {low}", f"good bye {low}", "end voice mode",
        f"hang up {low}", "hang up"))
    key_label = "the " + str(cfg.get("ptt_key", "home")).replace("_", " ") \
                + " key"
    # In hands-free there is no key to hold, so a separate line can be set.
    if str(cfg.get("mic_mode", "ptt")) == "open" and cfg.get("greeting_open_mic"):
        cfg["greeting"] = cfg["greeting_open_mic"]
    cfg["greeting"] = str(cfg["greeting"]).replace(
        "{name}", name).replace("{ptt_key}", key_label)
    cfg["signoff"] = str(cfg["signoff"]).replace("{name}", name)
    return cfg


CFG = load()

# The character half stays in YOUR agent's AGENTS.md. This is the medium.
if CFG.get("discipline_append"):
    DISCIPLINE = DISCIPLINE + " " + str(CFG["discipline_append"]).strip()
