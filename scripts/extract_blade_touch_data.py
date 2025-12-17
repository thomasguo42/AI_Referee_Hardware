#!/usr/bin/env python3
"""Utility for copying phrases where blade-to-blade contact precedes a hit within 1s."""
from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

TIME_LINE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)s\s*\|\s*(.*)$")


@dataclass
class PhraseWindow:
    folder: Path
    txt_path: Path
    last_blade_time: float
    first_hit_time: float

    @property
    def delta(self) -> float:
        return self.first_hit_time - self.last_blade_time


def iter_phrase_dirs(base_dir: Path) -> Iterable[Path]:
    for child in sorted(base_dir.iterdir()):
        if child.is_dir():
            yield child


def find_txt_file(folder: Path) -> Optional[Path]:
    txt_files = sorted(folder.glob("*.txt"))
    if not txt_files:
        return None
    if len(txt_files) == 1:
        return txt_files[0]
    for txt in txt_files:
        if txt.stem in folder.name:
            return txt
    return txt_files[0]


def evaluate_phrase(txt_path: Path) -> Optional[PhraseWindow]:
    last_blade: Optional[float] = None
    first_hit: Optional[float] = None

    with txt_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = TIME_LINE.match(line)
            if not match:
                continue
            timestamp = float(match.group(1))
            event_desc = match.group(2).strip()
            lower_desc = event_desc.lower()

            if "blade-to-blade" in lower_desc:
                last_blade = timestamp

            if first_hit is None and event_desc.upper().startswith("HIT:"):
                first_hit = timestamp
                break

    if last_blade is None or first_hit is None:
        return None

    return PhraseWindow(
        folder=txt_path.parent,
        txt_path=txt_path,
        last_blade_time=last_blade,
        first_hit_time=first_hit,
    )


def copy_phrase(folder: Path, destination_root: Path, dry_run: bool) -> Path:
    destination = destination_root / folder.name
    if dry_run:
        return destination
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(folder, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "training_dir",
        type=Path,
        default=Path("data/training_data"),
        nargs="?",
        help="Directory that contains phrase folders",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        default=Path("data/blade_touch_data"),
        nargs="?",
        help="Directory where qualifying phrases will be copied",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Maximum allowed seconds between last blade touch and first hit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List qualifying folders without copying",
    )
    args = parser.parse_args()

    training_dir = args.training_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    qualifying: list[PhraseWindow] = []
    for phrase_dir in iter_phrase_dirs(training_dir):
        txt_file = find_txt_file(phrase_dir)
        if not txt_file:
            continue
        window = evaluate_phrase(txt_file)
        if not window:
            continue
        if window.delta <= args.threshold:
            qualifying.append(window)
            copy_phrase(phrase_dir, output_dir, args.dry_run)

    if qualifying:
        print("Copied the following phrases (delta <= %.2fs):" % args.threshold)
        for window in qualifying:
            print(
                f"- {window.folder.name}: last blade {window.last_blade_time:.3f}s, "
                f"first hit {window.first_hit_time:.3f}s, delta {window.delta:.3f}s"
            )
    else:
        print("No phrases met the blade-touch threshold.")


if __name__ == "__main__":
    main()
