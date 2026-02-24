#!/usr/bin/env python3
"""Test how noise reduction affects recognition accuracy (CER) on sherpa-large."""

import importlib.machinery
import json
import os
import tempfile
import wave

import time

import jiwer
import numpy as np

from tests.test_sherpa_recognition import create_recognizer, recognize_wav

_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "..", "nerd-dictation")
_loader = importlib.machinery.SourceFileLoader("nerd_dictation", _SCRIPT_PATH)
_mod = _loader.load_module()
denoise_audio = _mod.denoise_audio

TESTS_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(TESTS_DIR, "..", "..", "vosk-models")
MODEL_DIR = os.path.join(MODELS_DIR, "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20")
WAV_DIR = os.path.join(TESTS_DIR, "test_wavs")
MANIFEST = json.load(open(os.path.join(WAV_DIR, "manifest.json")))


def cer(result, truth):
    return jiwer.cer(
        truth.replace(" ", "").lower().strip(),
        result.replace(" ", "").lower().strip(),
    )


def recognize_denoised(recognizer, wav_path, level):
    with wave.open(wav_path, "rb") as f:
        raw = f.readframes(f.getnframes())
        sr = f.getframerate()
        sw = f.getsampwidth()
        nc = f.getnchannels()

    denoised = denoise_audio(raw, sr, level=level)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(nc)
        wf.setsampwidth(sw)
        wf.setframerate(sr)
        wf.writeframes(denoised)

    text = recognize_wav(recognizer, tmp_path)
    os.unlink(tmp_path)
    return text


def main():
    if not os.path.isdir(MODEL_DIR):
        print(f"[SKIP] model not found: {MODEL_DIR}")
        return

    recognizer = create_recognizer(MODEL_DIR)
    levels = (0, 1, 2)

    print(f"{'wav':<28s}", end="")
    for lv in levels:
        print(f"  {'lv' + str(lv):>12s}", end="")
    print()
    print("-" * 70)

    totals_cer = {lv: [] for lv in levels}
    totals_time = {lv: [] for lv in levels}
    for entry in MANIFEST:
        wav_file = entry["wav"]
        wav_path = os.path.join(WAV_DIR, wav_file)
        if not os.path.exists(wav_path):
            print(f"  [SKIP] {wav_file}")
            continue

        row = f"{wav_file:<28s}"
        for lv in levels:
            t0 = time.time()
            text = recognize_denoised(recognizer, wav_path, lv)
            elapsed = time.time() - t0
            err = cer(text, entry["text"])
            totals_cer[lv].append(err)
            totals_time[lv].append(elapsed)
            row += f"  {err:>4.0%} {elapsed:>5.1f}s"
        print(row)

    print("-" * 70)
    row = f"{'AVG CER':<28s}"
    for lv in levels:
        avg = sum(totals_cer[lv]) / len(totals_cer[lv])
        tot = sum(totals_time[lv])
        row += f"  {avg:>4.0%} {tot:>5.1f}s"
    print(row)


if __name__ == "__main__":
    main()
