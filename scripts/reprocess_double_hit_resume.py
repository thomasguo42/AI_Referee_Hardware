#!/usr/bin/env python3
"""
Resume batch reprocessing in small chunks to avoid long-running-process issues.

This runs scripts/reprocess_double_hit_after.py repeatedly with --limit N and
advances the --start-after marker each time.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _sorted_dirs(base: Path) -> list[Path]:
    return sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name)


def _remaining_after(base: Path, start_after: str) -> list[Path]:
    dirs = [p for p in _sorted_dirs(base) if p.name > start_after]
    return [p for p in dirs if not (p / "analysis_result.json").exists()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Resume double_hit reprocessing in chunks.")
    ap.add_argument("--base-dir", type=Path, default=Path("new_data/double_hit"))
    ap.add_argument("--start-after", type=str, required=True)
    ap.add_argument("--model-path", type=Path, default=Path("models/yolo11x-pose.pt"))
    ap.add_argument("--chunk-size", type=int, default=5)
    args = ap.parse_args()

    base = args.base_dir
    if not base.exists():
        raise SystemExit(f"Base dir not found: {base}")

    start_after = args.start_after
    it = 0
    while True:
        remaining = _remaining_after(base, start_after)
        if not remaining:
            print("All remaining folders processed.")
            return 0

        # Advance by folder order, not by "processed" count, to prevent re-check loops.
        window_all = [p for p in _sorted_dirs(base) if p.name > start_after]
        batch = window_all[: args.chunk_size]
        if not batch:
            print("No further folders found after start-after marker.")
            return 0

        it += 1
        print(f"\n=== Chunk {it}: start_after={start_after}, batch_last={batch[-1].name}, remaining={len(remaining)} ===")

        cmd = [
            "python3",
            "scripts/reprocess_double_hit_after.py",
            "--base-dir",
            str(base),
            "--start-after",
            start_after,
            "--model-path",
            str(args.model_path),
            "--skip-existing",
            "--limit",
            str(args.chunk_size),
        ]
        subprocess.run(cmd, check=False)

        start_after = batch[-1].name


if __name__ == "__main__":
    raise SystemExit(main())

