#!/usr/bin/env python3
"""Benchmark speech recognition accuracy across models using recorded test wavs."""

import json
import os
import time
import wave

import jiwer
import numpy as np

from tests.test_sherpa_recognition import create_recognizer as create_sherpa_recognizer
from tests.test_sherpa_recognition import recognize_wav as sherpa_recognize_wav

TESTS_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(TESTS_DIR, "..", "..", "vosk-models")
WAV_DIR = os.path.join(TESTS_DIR, "test_wavs")
MANIFEST = json.loads(open(os.path.join(WAV_DIR, "manifest.json")).read())


def read_wav_samples(wav_path):
    with wave.open(wav_path, "rb") as f:
        raw = f.readframes(f.getnframes())
        sample_rate = f.getframerate()
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sample_rate


def cer(result, truth):
    """Character Error Rate via jiwer. Lower is better."""
    return jiwer.cer(
        truth.replace(" ", "").lower().strip(),
        result.replace(" ", "").lower().strip(),
    )


def load_vosk(model_path):
    # lazy import: vosk is optional, sherpa-only users won't have it
    import vosk
    vosk.SetLogLevel(-1)
    return vosk.Model(model_path)


def recognize_vosk(model, wav_path):
    samples, sample_rate = read_wav_samples(wav_path)
    raw_bytes = (samples * 32768).astype(np.int16).tobytes()
    rec = model.__class__.__module__  # need vosk module for KaldiRecognizer
    import vosk
    rec = vosk.KaldiRecognizer(model, sample_rate)
    chunk_size = int(0.1 * sample_rate) * 2
    for i in range(0, len(raw_bytes), chunk_size):
        rec.AcceptWaveform(raw_bytes[i:i + chunk_size])
    return json.loads(rec.FinalResult())["text"]


def load_sherpa(model_path):
    return create_sherpa_recognizer(model_path)


def recognize_sherpa(model, wav_path):
    return sherpa_recognize_wav(model, wav_path)


MODELS = [
    ("vosk-small", os.path.join(MODELS_DIR, "vosk-model-small-cn-0.22"), load_vosk, recognize_vosk),
    ("vosk-large", os.path.join(MODELS_DIR, "vosk-model-cn-0.22"), load_vosk, recognize_vosk),
    ("sherpa-small", os.path.join(MODELS_DIR, "sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16"), load_sherpa, recognize_sherpa),
    ("sherpa-large", os.path.join(MODELS_DIR, "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"), load_sherpa, recognize_sherpa),
]


def main():
    results = {}
    for model_name, model_path, load_fn, recognize_fn in MODELS:
        if not os.path.isdir(model_path):
            print(f"[SKIP] {model_name}: {model_path} not found")
            continue
        print(f"\n{'='*60}")
        print(f"  Model: {model_name} (loading...)")
        model = load_fn(model_path)
        print(f"{'='*60}")
        results[model_name] = {}
        for entry in MANIFEST:
            wav_file = entry["wav"]
            wav_path = os.path.join(WAV_DIR, wav_file)
            if not os.path.exists(wav_path):
                print(f"  [SKIP] {wav_file}")
                continue
            t0 = time.time()
            text = recognize_fn(model, wav_path)
            elapsed = time.time() - t0
            err = cer(text, entry["text"])
            results[model_name][wav_file] = (text, err, elapsed)
            print(f"  {wav_file:30s}  CER={err:.0%}  {elapsed:.1f}s")
            print(f"    -> {text}")

    print(f"\n\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")
    header = f"{'wav':<30s}"
    model_names = [n for n, _, _, _ in MODELS if n in results]
    for mn in model_names:
        header += f"  {mn:>14s}"
    print(header)
    print("-" * len(header))
    for entry in MANIFEST:
        wav_file = entry["wav"]
        row = f"{wav_file:<30s}"
        for mn in model_names:
            if wav_file in results[mn]:
                row += f"  {results[mn][wav_file][1]:>13.0%}"
            else:
                row += f"  {'N/A':>13s}"
        print(row)

    print("-" * len(header))
    row = f"{'AVG CER (lower=better)':<30s}"
    for mn in model_names:
        errs = [v[1] for v in results[mn].values()]
        avg = sum(errs) / len(errs) if errs else 0
        row += f"  {avg:>13.0%}"
    print(row)


if __name__ == "__main__":
    main()
