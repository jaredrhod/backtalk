import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from backtalk import signals


class SignalBusTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        directory = Path(self._temporary.name)
        self._original = {
            "_STATE_FILE": signals._STATE_FILE,
            "_WAVEFORM_FILE": signals._WAVEFORM_FILE,
            "_REPLY_DONE_FILE": signals._REPLY_DONE_FILE,
            "_DIRECTION_FILE": signals._DIRECTION_FILE,
            "_BH_STATE": signals._BH_STATE,
            "_BH_WAVE": signals._BH_WAVE,
            "_last_waveform_write": signals._last_waveform_write,
        }
        signals._STATE_FILE = str(directory / ".voice_state")
        signals._WAVEFORM_FILE = str(directory / ".voice_waveform")
        signals._REPLY_DONE_FILE = str(directory / ".voice_reply_done")
        signals._DIRECTION_FILE = str(directory / ".voice_direction")
        signals._BH_STATE = ""
        signals._BH_WAVE = ""
        signals._last_waveform_write = 0.0
        self.directory = directory

    def tearDown(self):
        for name, value in self._original.items():
            setattr(signals, name, value)
        self._temporary.cleanup()

    def test_set_state_and_reply_done_publish_readable_bus_files(self):
        signals.set_state("thinking")
        signals.reply_done()

        self.assertEqual((self.directory / ".voice_state").read_text(), "thinking")
        payload = json.loads((self.directory / ".voice_reply_done").read_text())
        self.assertIn("ts", payload)

    def test_feed_waveform_publishes_speaking_and_64_samples(self):
        signals.feed_waveform(np.arange(128, dtype=np.int16))

        self.assertEqual((self.directory / ".voice_state").read_text(), "speaking")
        payload = json.loads((self.directory / ".voice_waveform").read_text())
        self.assertIn("ts", payload)
        self.assertEqual(len(payload["samples"]), 64)
        self.assertEqual(payload["samples"][0], 0.0)
        self.assertEqual(payload["samples"][-1], 127.0)

    def test_direction_ignores_empty_and_publishes_nonempty_items(self):
        signals.direction([])
        self.assertFalse((self.directory / ".voice_direction").exists())

        signals.direction(["look left", "raise hand"])
        payload = json.loads((self.directory / ".voice_direction").read_text())
        self.assertEqual(payload["directions"], ["look left", "raise hand"])
