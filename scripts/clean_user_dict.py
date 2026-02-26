#!/usr/bin/env python3
"""Dry-run cleaner for Fcitx5 pinyin user dictionary.

Flags words for removal based on:
  1. Usage frequency from user.history (never used or low frequency)
  2. Pinyin mismatch (recorded pinyin ≠ standard pinyin → likely wrong candidate)

Outputs a report for manual review. Use --apply to write a cleaned dict.

Usage:
    python scripts/clean_user_dict.py                    # dry run
    python scripts/clean_user_dict.py --min-freq 2       # only keep words used ≥2 times
    python scripts/clean_user_dict.py --apply             # actually write cleaned dict
"""

import argparse
import os
import subprocess
import sys
import tempfile
from collections import Counter

from pypinyin import pinyin, pinyin_dict, Style

FCITX5_DICT = os.path.expanduser(
    "~/.local/share/fcitx5/pinyin/user.dict"
)
FCITX5_HISTORY = os.path.expanduser(
    "~/.local/share/fcitx5/pinyin/user.history"
)


def dump_dict(dict_path):
    """Dump binary dict to text, return list of (word, pinyin, freq_str) tuples."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        tmp = f.name
    try:
        subprocess.run(
            ["libime_pinyindict", "-d", dict_path, tmp],
            check=True, capture_output=True,
        )
        entries = []
        with open(tmp, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    entries.append((parts[0], parts[1], parts[2] if len(parts) > 2 else "0"))
        return entries
    finally:
        os.unlink(tmp)


def dump_history(history_path):
    """Dump binary history to text, return word frequency counter."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        tmp = f.name
    try:
        subprocess.run(
            ["libime_history", history_path, tmp],
            check=True, capture_output=True,
        )
        counter = Counter()
        with open(tmp, encoding="utf-8") as f:
            for line in f:
                for token in line.strip().split():
                    counter[token] += 1
        return counter
    finally:
        os.unlink(tmp)


_TONE_MAP = str.maketrans(
    "āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ",
    "aaaaeeeeiiiioooouuuuvvvv",
)


def _strip_tone(s):
    return s.translate(_TONE_MAP)


def _build_polyphonic_set():
    """Build set of polyphonic characters from pypinyin's internal dict.

    A character is polyphonic if it has multiple distinct base syllables
    (ignoring tone differences like zhōng/zhòng).
    """
    poly = set()
    for cp, py_str in pinyin_dict.pinyin_dict.items():
        bases = {_strip_tone(r) for r in py_str.split(",")}
        if len(bases) > 1:
            poly.add(chr(cp))
    return poly


_POLYPHONIC = None


def _get_polyphonic():
    global _POLYPHONIC
    if _POLYPHONIC is None:
        _POLYPHONIC = _build_polyphonic_set()
    return _POLYPHONIC


# Fuzzy pinyin pairs: initials and finals that input methods commonly conflate
_FUZZY_INITIALS = [("zh", "z"), ("ch", "c"), ("sh", "s")]
_FUZZY_FINALS = [("eng", "en"), ("ing", "in"), ("ang", "an")]


def _fuzzy_eq(a, b):
    """Check if two pinyin syllables are fuzzy-equal."""
    for x, y in _FUZZY_INITIALS:
        if a.startswith(x) and b.startswith(y) and a[len(x):] == b[len(y):]:
            return True
        if a.startswith(y) and b.startswith(x) and a[len(y):] == b[len(x):]:
            return True
    for x, y in _FUZZY_FINALS:
        if a.endswith(x) and b.endswith(y) and a[:-len(x)] == b[:-len(y)]:
            return True
        if a.endswith(y) and b.endswith(x) and a[:-len(y)] == b[:-len(x)]:
            return True
    return False


def check_pinyin(word, recorded_pinyin):
    """Check if word's pinyin matches, skipping polyphonic characters.

    Returns (match: bool, standard_pinyin: str).
    """
    syllables = [p[0] for p in pinyin(word, style=Style.NORMAL)]
    standard = "'".join(syllables)

    if standard == recorded_pinyin:
        return True, standard

    rec_parts = recorded_pinyin.split("'")
    if len(syllables) != len(rec_parts):
        return False, standard

    poly = _get_polyphonic()
    for char, std_py, rec_py in zip(word, syllables, rec_parts):
        if char in poly:
            continue
        if std_py == rec_py:
            continue
        if _fuzzy_eq(std_py, rec_py):
            continue
        return False, standard

    return True, standard


def compile_dict(entries, output_path):
    """Compile text entries back to binary dict."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for word, py, freq in entries:
            f.write("{} {} {}\n".format(word, py, freq))
        tmp = f.name
    try:
        subprocess.run(
            ["libime_pinyindict", tmp, output_path],
            check=True, capture_output=True,
        )
    finally:
        os.unlink(tmp)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dict", default=FCITX5_DICT,
        help="Path to Fcitx5 user dict (default: %(default)s)",
    )
    parser.add_argument(
        "--history", default=FCITX5_HISTORY,
        help="Path to Fcitx5 user history (default: %(default)s)",
    )
    parser.add_argument(
        "--min-freq", type=int, default=1,
        help="Minimum usage frequency to keep a word (default: %(default)s)",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually write the cleaned dict (default: dry run only)",
    )
    args = parser.parse_args()

    print("Loading dict: {}".format(args.dict))
    entries = dump_dict(args.dict)
    print("  {} entries".format(len(entries)))

    print("Loading history: {}".format(args.history))
    freq = dump_history(args.history)
    print("  {} unique tokens".format(len(freq)))

    pinyin_bad = []
    never_used = []
    low_freq = []
    keep = []

    for word, py, f in entries:
        usage = freq[word]
        py_ok, std_py = check_pinyin(word, py)

        if not py_ok:
            pinyin_bad.append((word, py, std_py, usage))
        elif usage == 0:
            never_used.append((word, py))
        elif usage < args.min_freq:
            low_freq.append((word, py, usage))
        else:
            keep.append((word, py, f))

    total_remove = len(pinyin_bad) + len(never_used) + len(low_freq)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("  Total entries:      {}".format(len(entries)))
    print("  Keep:               {}".format(len(keep)))
    print("  Remove:             {}".format(total_remove))
    print("    Pinyin mismatch:  {}".format(len(pinyin_bad)))
    print("    Never used:       {}".format(len(never_used)))
    if args.min_freq > 1:
        print("    Low freq (<{}):    {}".format(args.min_freq, len(low_freq)))
    print()

    if pinyin_bad:
        print("-" * 60)
        print("PINYIN MISMATCH (non-polyphonic chars with wrong pinyin)")
        print("-" * 60)
        pinyin_bad.sort(key=lambda x: x[3], reverse=True)
        for word, rec_py, std_py, usage in pinyin_bad:
            print("  {:12s}  recorded={:20s}  standard={:20s}  freq={}".format(
                word, rec_py, std_py, usage,
            ))
        print()

    if never_used:
        print("-" * 60)
        print("NEVER USED (not found in history)")
        print("-" * 60)
        never_used.sort(key=lambda x: x[0])
        for word, py in never_used[:80]:
            print("  {}  ({})".format(word, py))
        if len(never_used) > 80:
            print("  ... and {} more".format(len(never_used) - 80))
        print()

    if low_freq:
        print("-" * 60)
        print("LOW FREQUENCY (used <{} times)".format(args.min_freq))
        print("-" * 60)
        low_freq.sort(key=lambda x: x[2])
        for word, py, usage in low_freq[:80]:
            print("  {:12s}  freq={}  ({})".format(word, usage, py))
        if len(low_freq) > 80:
            print("  ... and {} more".format(len(low_freq) - 80))
        print()

    if args.apply:
        backup = args.dict + ".bak"
        print("Backing up original to: {}".format(backup))
        subprocess.run(["cp", args.dict, backup], check=True)
        print("Compiling cleaned dict ({} entries)...".format(len(keep)))
        compile_dict(keep, args.dict)
        print("Done. Restart Fcitx5 to apply.")
    else:
        print("This is a DRY RUN. Use --apply to write changes.")
        print("Tip: adjust --min-freq to be more aggressive.")


if __name__ == "__main__":
    main()
