#!/usr/bin/env python3
"""
Split new_data phrase folders into:
  - new_data/single_hit/
  - new_data/double_hit/
based on a line like:
  Scores -> Fencer 1: HIT, Fencer 2: HIT

We parse the *last* matching Scores line in the .txt (some files include it twice),
then move the entire phrase folder accordingly.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple


SCORES_RE = re.compile(
    r"Scores\s*->\s*Fencer\s*1:\s*(HIT|MISS)\s*,\s*Fencer\s*2:\s*(HIT|MISS)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Classification:
    kind: str  # "single_hit" | "double_hit" | "unclassified"
    f1: Optional[str] = None
    f2: Optional[str] = None
    txt: Optional[str] = None


def _iter_phrase_dirs(root: Path) -> Iterable[Path]:
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if p.name in {"single_hit", "double_hit", "unclassified"}:
            continue
        if p.name.startswith("part_"):
            # Shouldn't exist after flattening, but skip just in case.
            continue
        yield p


def _pick_txt_file(d: Path) -> Optional[Path]:
    txts = sorted(d.glob("*.txt"))
    if not txts:
        return None
    same = d / f"{d.name}.txt"
    if same.exists():
        return same
    return txts[0]


def _parse_scores(txt_path: Path) -> Optional[Tuple[str, str]]:
    last = None
    with txt_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = SCORES_RE.search(line.rstrip("\n"))
            if not m:
                continue
            last = (m.group(1).upper(), m.group(2).upper())
    return last


def _classify_dir(d: Path) -> Classification:
    txt = _pick_txt_file(d)
    if txt is None:
        return Classification(kind="unclassified")
    scores = _parse_scores(txt)
    if scores is None:
        return Classification(kind="unclassified", txt=str(txt))
    f1, f2 = scores
    if f1 == "HIT" and f2 == "HIT":
        return Classification(kind="double_hit", f1=f1, f2=f2, txt=str(txt))
    if (f1 == "HIT" and f2 == "MISS") or (f1 == "MISS" and f2 == "HIT"):
        return Classification(kind="single_hit", f1=f1, f2=f2, txt=str(txt))
    return Classification(kind="unclassified", f1=f1, f2=f2, txt=str(txt))


def _unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    base = dest.name
    parent = dest.parent
    i = 2
    while True:
        cand = parent / f"{base}__{i}"
        if not cand.exists():
            return cand
        i += 1


def _move_dir(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(src, dst)
    except OSError:
        shutil.move(str(src), str(dst))


def main() -> int:
    root = Path("new_data")
    if not root.exists():
        raise SystemExit("new_data/ not found (run from repo root)")

    out_single = root / "single_hit"
    out_double = root / "double_hit"
    out_uncl = root / "unclassified"

    log_path = root / "_hit_split_log.jsonl"
    counts = {"single_hit": 0, "double_hit": 0, "unclassified": 0}

    phrase_dirs = list(_iter_phrase_dirs(root))
    if not phrase_dirs:
        print("No candidate phrase folders found under new_data/. Nothing to do.")
        return 0

    with log_path.open("w", encoding="utf-8") as log:
        for d in phrase_dirs:
            cls = _classify_dir(d)
            if cls.kind == "single_hit":
                dest_root = out_single
            elif cls.kind == "double_hit":
                dest_root = out_double
            else:
                dest_root = out_uncl

            dst = _unique_dest(dest_root / d.name)
            _move_dir(d, dst)
            counts[cls.kind] += 1

            log.write(
                json.dumps(
                    {
                        "src": str(d),
                        "dst": str(dst),
                        "kind": cls.kind,
                        "fencer_1": cls.f1,
                        "fencer_2": cls.f2,
                        "txt": cls.txt,
                    }
                )
                + "\n"
            )

    # Remove unclassified dir if we didn't need it.
    if counts["unclassified"] == 0:
        try:
            out_uncl.rmdir()
        except OSError:
            pass

    print(json.dumps(counts, indent=2, sort_keys=True))
    print(f"log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

