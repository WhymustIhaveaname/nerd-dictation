#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Unit tests for the denoise_audio function.

Run with:
    python -m pytest tests/test_noise_reduction.py -v
or:
    python tests/test_noise_reduction.py
"""

import importlib.machinery
import os
import sys
import unittest
import wave

import numpy as np

# ---------------------------------------------------------------------------
# Import denoise_audio from the main script.
#
# nerd-dictation is a single executable script (no .py extension), so we use
# SourceFileLoader - the same technique used in tests/from_words_to_digits.py.
# ---------------------------------------------------------------------------
_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "..", "nerd-dictation")
_loader = importlib.machinery.SourceFileLoader("nerd_dictation", _SCRIPT_PATH)
_mod = _loader.load_module()
denoise_audio = _mod.denoise_audio

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
_WAV_PATH = os.path.join(os.path.dirname(__file__), "test_wavs", "zh_test_raw.wav")
_SAMPLE_RATE = 16000  # matches zh_test_raw.wav


def _load_wav_as_int16_bytes() -> bytes:
    """Read the test wav and return raw int16 PCM bytes."""
    with wave.open(_WAV_PATH, "rb") as f:
        return f.readframes(f.getnframes())


def _load_wav_as_float32_bytes() -> bytes:
    """Return the test wav samples as float32 bytes (sherpa-path format)."""
    raw = _load_wav_as_int16_bytes()
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples.tobytes()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestDenoiseAudioLevel0(unittest.TestCase):
    """level=0 must be a no-op regardless of dtype."""

    def test_int16_passthrough(self):
        data = _load_wav_as_int16_bytes()
        result = denoise_audio(data, _SAMPLE_RATE, level=0)
        self.assertIs(result, data, "level=0 must return the original object unchanged")

    def test_float32_passthrough(self):
        data = _load_wav_as_float32_bytes()
        result = denoise_audio(data, _SAMPLE_RATE, level=0, dtype="float32")
        self.assertIs(result, data, "level=0 must return the original object unchanged")


class TestDenoiseAudioInt16(unittest.TestCase):
    """Tests for the VOSK path (int16 bytes)."""

    def setUp(self):
        try:
            import noisereduce  # noqa: F401
        except ImportError:
            self.skipTest("noisereduce not installed")
        self.data = _load_wav_as_int16_bytes()

    def test_level1_returns_bytes(self):
        result = denoise_audio(self.data, _SAMPLE_RATE, level=1)
        self.assertIsInstance(result, bytes)

    def test_level2_returns_bytes(self):
        result = denoise_audio(self.data, _SAMPLE_RATE, level=2)
        self.assertIsInstance(result, bytes)

    def test_length_preserved(self):
        """Denoised output must have the same byte length as the input."""
        result = denoise_audio(self.data, _SAMPLE_RATE, level=1)
        self.assertEqual(len(result), len(self.data))

    def test_level1_modifies_audio(self):
        """Denoised output should differ from the input (not a pass-through)."""
        result = denoise_audio(self.data, _SAMPLE_RATE, level=1)
        self.assertNotEqual(result, self.data)

    def test_level2_more_aggressive_than_level1(self):
        """Level 2 should produce a larger deviation from the original than level 1."""
        result1 = np.frombuffer(denoise_audio(self.data, _SAMPLE_RATE, level=1), dtype=np.int16).astype(np.float32)
        result2 = np.frombuffer(denoise_audio(self.data, _SAMPLE_RATE, level=2), dtype=np.int16).astype(np.float32)
        original = np.frombuffer(self.data, dtype=np.int16).astype(np.float32)
        diff1 = float(np.mean(np.abs(result1 - original)))
        diff2 = float(np.mean(np.abs(result2 - original)))
        self.assertGreater(diff2, diff1, "Level 2 should alter the signal more than level 1")


class TestDenoiseAudioFloat32(unittest.TestCase):
    """Tests for the sherpa path (float32 bytes)."""

    def setUp(self):
        try:
            import noisereduce  # noqa: F401
        except ImportError:
            self.skipTest("noisereduce not installed")
        self.data = _load_wav_as_float32_bytes()

    def test_level1_returns_bytes(self):
        result = denoise_audio(self.data, _SAMPLE_RATE, level=1, dtype="float32")
        self.assertIsInstance(result, bytes)

    def test_length_preserved(self):
        """Float32 byte length must be unchanged after denoising."""
        result = denoise_audio(self.data, _SAMPLE_RATE, level=1, dtype="float32")
        self.assertEqual(len(result), len(self.data))

    def test_samples_are_valid_float32(self):
        """Output bytes must be interpretable as finite float32 samples."""
        result = denoise_audio(self.data, _SAMPLE_RATE, level=1, dtype="float32")
        samples = np.frombuffer(result, dtype=np.float32)
        self.assertTrue(np.all(np.isfinite(samples)), "Denoised float32 output contains NaN or Inf")

    def test_level1_modifies_audio(self):
        result = denoise_audio(self.data, _SAMPLE_RATE, level=1, dtype="float32")
        self.assertNotEqual(result, self.data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
