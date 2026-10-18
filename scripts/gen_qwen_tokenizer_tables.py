#!/usr/bin/env python3
"""Usage:
  python3 scripts/gen_qwen_tokenizer_tables.py

Generates src/KokkosDNN_QwenTokenizerTables.inc from the local Python's
unicodedata tables (composability is probed with NFC itself, so no UCD
data files and no version-skew risk):

  - kWhiteSpace      : exact codepoints stripped by Python str.strip()
  - kLetterRanges    : sorted [lo,hi] ranges of Unicode L* (pre-tokenizer)
  - kNumberRanges    : sorted [lo,hi] ranges of Unicode N* (pre-tokenizer)
  - kCcc             : codepoints with nonzero canonical combining class
  - kDecompPool/kDecomp : fully expanded canonical decompositions
                         (Hangul syllables excluded: algorithmic in C++)
  - kCompose         : (starter,combiner)->composite, minus
                       CompositionExclusions (Hangul excluded: algorithmic)
"""
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "src" / "KokkosDNN_QwenTokenizerTables.inc"

HANG_LO, HANG_HI = 0xAC00, 0xD7A3
# Exact White_Space set stripped by CPython str.strip().
STRIP_CPS = sorted(
    list(range(0x09, 0x0E)) + [0x20, 0x1C, 0x1D, 0x1E, 0x1F, 0x85, 0xA0,
                               0x1680] + list(range(0x2000, 0x200B)) +
    [0x2028, 0x2029, 0x202F, 0x205F, 0x3000])
# \s in the pre-tokenizer regex = Unicode White_Space.
SPACE_CPS = sorted(
    list(range(0x09, 0x0E)) + [0x20, 0x85, 0xA0, 0x1680] +
    list(range(0x2000, 0x200B)) + [0x2028, 0x2029, 0x202F, 0x205F, 0x3000])


def is_composable(a, b, comp):
    """Exact probe: (a,b) composes to comp iff NFC(a+b) == comp.

    This subsumes both CompositionExclusions.txt and the starter check
    (a non-composable pair normalizes back to two characters), so no UCD
    data file is needed and there is no version-skew risk.
    """
    return unicodedata.normalize("NFC", chr(a) + chr(b)) == chr(comp)


def canon_decomp(cp):
    """Fully-expanded canonical decomposition, or None.

    None for Hangul syllables (algorithmic in C++) and for characters
    with no canonical decomposition.
    """
    if HANG_LO <= cp <= HANG_HI:
        return None
    d = unicodedata.decomposition(chr(cp))
    if not d or d.startswith("<"):
        return None
    out = []
    for part in d.split():
        out.extend(canon_decomp(int(part, 16)) or [int(part, 16)])
    return out


def ranges_of(pred):
    ranges = []
    start = None
    for cp in range(0x110000):
        if 0xD800 <= cp < 0xE000:
            continue
        if pred(cp):
            if start is None:
                start = cp
        elif start is not None:
            ranges.append((start, cp - 1))
            start = None
    if start is not None:
        ranges.append((start, 0x10FFFF))
    # merge adjacency (surrogate gap splits a range in two)
    merged = []
    for lo, hi in ranges:
        if merged and lo == merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], hi)
        else:
            merged.append((lo, hi))
    return merged


def fmt_u32_list(name, vals, per_line=8):
    lines = [f"inline constexpr uint32_t {name}[] = {{"]
    for i in range(0, len(vals), per_line):
        lines.append("    " + ", ".join(f"{c:#x}" for c in vals[i:i + 8]) +
                     ",")
    lines.append("};")
    return "\n".join(lines)


def fmt_ranges(name, ranges):
    lines = [f"inline constexpr Range {name}[] = {{"]
    for lo, hi in ranges:
        lines.append(f"    {{{lo:#x}, {hi:#x}}},")
    lines.append("};")
    return "\n".join(lines)


def main():
    print(f"unicode {unicodedata.unidata_version} "
          "(composability probed, no data files)", file=sys.stderr)

    cats = {}
    for cp in range(0x110000):
        if 0xD800 <= cp < 0xE000:
            continue
        cats[cp] = unicodedata.category(chr(cp))
    letters = ranges_of(lambda cp: cats[cp].startswith("L"))
    numbers = ranges_of(lambda cp: cats[cp].startswith("N"))
    print(f"letter ranges: {len(letters)}, number ranges: {len(numbers)}",
          file=sys.stderr)

    ccc = []
    for cp in range(0x110000):
        if 0xD800 <= cp < 0xE000:
            continue
        c = unicodedata.combining(chr(cp))
        if c:
            ccc.append((cp, c))
    print(f"nonzero ccc: {len(ccc)}", file=sys.stderr)

    decomp = {}
    for cp in range(0x110000):
        if 0xD800 <= cp < 0xE000:
            continue
        d = canon_decomp(cp)
        if d:
            decomp[cp] = d
    print(f"canonical decompositions: {len(decomp)}", file=sys.stderr)

    compose = {}
    for comp, d in decomp.items():
        if len(d) == 2:
            a, b = d
            # Exact probe (subsumes starter check + CompositionExclusions).
            if (a, b) not in compose and is_composable(a, b, comp):
                compose[(a, b)] = comp
    # Hangul Jamo pairs compose algorithmically in C++.
    compose = {(a, b): c for (a, b), c in compose.items()
               if not (0x1100 <= a <= 0x11FF)}
    print(f"composition pairs: {len(compose)}", file=sys.stderr)

    pool = []
    decomp_idx = {}
    for cp in sorted(decomp):
        decomp_idx[cp] = (len(pool), len(decomp[cp]))
        pool.extend(decomp[cp])

    L = []
    L.append("// Generated by scripts/gen_qwen_tokenizer_tables.py "
             f"(unicode {unicodedata.unidata_version}). DO NOT EDIT.")
    L.append("#pragma once")
    L.append("#include <cstdint>")
    L.append("namespace KokkosDNN {")
    L.append("namespace QwenTokenizer {")
    L.append("namespace Impl {")
    L.append(fmt_u32_list("kStripCps", STRIP_CPS))
    L.append(fmt_u32_list("kSpaceCps", SPACE_CPS))
    L.append("struct Range { uint32_t lo, hi; };")
    L.append(fmt_ranges("kLetterRanges", letters))
    L.append(fmt_ranges("kNumberRanges", numbers))
    L.append("struct CccEntry { uint32_t cp; uint8_t ccc; };")
    L.append("inline constexpr CccEntry kCcc[] = {")
    for cp, c in ccc:
        L.append(f"    {{{cp:#x}, {c}}},")
    L.append("};")
    L.append("inline constexpr uint32_t kDecompPool[] = {")
    for i in range(0, len(pool), 8):
        L.append("    " + ", ".join(f"{c:#x}" for c in pool[i:i + 8]) + ",")
    L.append("};")
    L.append("struct DecompEntry { uint32_t cp, off, len; };")
    L.append("inline constexpr DecompEntry kDecomp[] = {")
    for cp in sorted(decomp):
        off, ln = decomp_idx[cp]
        L.append(f"    {{{cp:#x}, {off}, {ln}}},")
    L.append("};")
    L.append("struct ComposeEntry { uint64_t key; uint32_t composite; };")
    L.append("// key = ((uint64_t)starter << 21) | combiner")
    L.append("inline constexpr ComposeEntry kCompose[] = {")
    for (a, b) in sorted(compose):
        L.append(f"    {{{(a << 21) | b:#x}, {compose[(a, b)]:#x}}},")
    L.append("};")
    L.append("}  // namespace Impl")
    L.append("}  // namespace QwenTokenizer")
    L.append("}  // namespace KokkosDNN")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
