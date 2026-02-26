#!/usr/bin/env python3
"""Generate a sherpa-onnx hotwords file from Fcitx5 user dictionary.

Reads the Fcitx5 pinyin user dictionary, tokenizes words with the model's
BPE, and writes a hotwords file with uniform boost score.

Usage:
    python scripts/generate_hotwords.py DICT_PATH
    python scripts/generate_hotwords.py DICT_PATH --score 2.0
"""

import argparse
import os
import subprocess
import sys
import tempfile

import sherpa_onnx


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(
    SCRIPT_DIR, "..", "..", "vosk-models",
    "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
)
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "..", "hotwords.txt")


def dump_fcitx5_dict(dict_path):
    """Dump Fcitx5 binary pinyin dict to text, return set of words."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        tmp_path = f.name
    try:
        subprocess.run(
            ["libime_pinyindict", "-d", dict_path, tmp_path],
            check=True, capture_output=True,
        )
        words = set()
        with open(tmp_path, encoding="utf-8") as f:
            for line in f:
                words.add(line.split()[0])
        return words
    finally:
        os.unlink(tmp_path)


def tokenize_words(words, model_dir):
    """Tokenize words using sherpa-onnx text2token with the model's BPE."""
    tokens_file = os.path.join(model_dir, "tokens.txt")
    bpe_model = os.path.join(model_dir, "bpe.model")

    word_list = list(words)
    results = sherpa_onnx.text2token(
        word_list, tokens=tokens_file, bpe_model=bpe_model,
    )

    if len(results) != len(word_list):
        sys.stderr.write(
            "Warning: text2token returned {:d} results for {:d} words, "
            "some words may have been skipped.\n".format(len(results), len(word_list))
        )

    tokenized = []
    for word, tokens in zip(word_list, results):
        if tokens:
            tokenized.append((word, tokens))
    return tokenized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dict",
        help="Path to Fcitx5 pinyin user dict (binary format)",
    )
    parser.add_argument(
        "--model-dir", default=DEFAULT_MODEL_DIR,
        help="Path to sherpa-onnx model directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output", "-o", default=DEFAULT_OUTPUT,
        help="Output hotwords file path (default: %(default)s)",
    )
    parser.add_argument(
        "--score", type=float, default=0.5,
        help="Uniform boost score for all hotwords (default: %(default)s)",
    )
    args = parser.parse_args()

    sys.stderr.write("Dumping Fcitx5 dict: {:s}\n".format(args.dict))
    words = dump_fcitx5_dict(args.dict)
    sys.stderr.write("  {:d} unique words found\n".format(len(words)))

    sys.stderr.write("Tokenizing with BPE model...\n")
    tokenized = tokenize_words(words, args.model_dir)
    sys.stderr.write("  {:d} words tokenized successfully\n".format(len(tokenized)))

    with open(args.output, "w", encoding="utf-8") as f:
        for word, tokens in sorted(tokenized):
            f.write("{:s} :{:.1f}\n".format(" ".join(tokens), args.score))

    sys.stderr.write("Wrote {:d} hotwords to {:s}\n".format(len(tokenized), args.output))


if __name__ == "__main__":
    main()
