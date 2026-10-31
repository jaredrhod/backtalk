import tempfile
import unittest
from pathlib import Path

from backtalk import config


class ConfigTests(unittest.TestCase):
    def test_load_merges_nested_settings_and_expands_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "backtalk.json"
            path.write_text(
                '{"name":"Test Voice","agent_dir":"~/agent",'
                '"extra_dirs":["~/notes"],"signals_dir":"~/signals",'
                '"elevenlabs":{"enabled":true,"voice_id":"voice-123"}}'
            )
            original = config.CONFIG_PATH
            config.CONFIG_PATH = path
            try:
                loaded = config.load()
            finally:
                config.CONFIG_PATH = original

        self.assertEqual(loaded["name"], "Test Voice")
        self.assertTrue(loaded["agent_dir"].endswith("/agent"))
        self.assertTrue(loaded["extra_dirs"][0].endswith("/notes"))
        self.assertTrue(loaded["signals_dir"].endswith("/signals"))
        self.assertTrue(loaded["elevenlabs"]["enabled"])
        self.assertEqual(loaded["elevenlabs"]["voice_id"], "voice-123")
        self.assertEqual(loaded["elevenlabs"]["model"], config.DEFAULTS["elevenlabs"]["model"])

    def test_invalid_config_falls_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "backtalk.json"
            path.write_text("{not valid json")
            original = config.CONFIG_PATH
            config.CONFIG_PATH = path
            try:
                loaded = config.load()
            finally:
                config.CONFIG_PATH = original

        self.assertEqual(loaded["name"], "Assistant")
        self.assertEqual(loaded["mic_mode"], "ptt")
