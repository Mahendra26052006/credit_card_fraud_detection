"""Download the ULB credit-card fraud CSV into data/raw/."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import download_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download creditcard.csv")
    parser.add_argument("--source", type=str, default=None, help="Existing local CSV to copy")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    path = download_dataset(
        source_path=Path(args.source) if args.source else None,
        force=args.force,
    )
    print(f"Dataset ready: {path}")


if __name__ == "__main__":
    main()
