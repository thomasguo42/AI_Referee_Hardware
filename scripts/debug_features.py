#!/usr/bin/env python3
"""Quickly inspect extracted features for a sample fencing phrase."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

import blade_touch_referee as btr

DEFAULT_DATA_DIR = Path('data/blade_touch_data')


def pick_sample(data_dir: Path, name: Optional[str]):
    folders = []
    if name:
        folders.append(data_dir / name)
    else:
        folders.extend(sorted(d for d in data_dir.iterdir() if d.is_dir()))
    for folder in folders:
        if not folder.exists():
            continue
        try:
            return btr.build_sample(folder)
        except Exception as exc:
            print(f"Skipping {folder.name}: {exc}")
            continue
    raise RuntimeError(f"No usable phrase folders found under {data_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=DEFAULT_DATA_DIR,
        help='Directory containing phrase folders',
    )
    parser.add_argument(
        '--folder',
        type=str,
        help='Specific phrase folder name to inspect',
    )
    args = parser.parse_args()

    sample = pick_sample(args.data_dir, args.folder)
    print(f"Inspecting phrase: {sample.name}")
    print(f"  Winner: {sample.winner}")
    print(f"  Contact time: {sample.contact_time:.3f}s")
    df = pd.DataFrame(sample.features, index=[0]).T
    df.columns = ['value']
    df = df.sort_index()
    print('\nFirst 20 feature entries:')
    print(df.head(20))


if __name__ == '__main__':
    main()
