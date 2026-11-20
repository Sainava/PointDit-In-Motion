"""Collect every sweep_status.json and run.json under a sweep folder into sweep_latency.csv (CPU only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pim.sweep import summary_rows, write_summary  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True, help="holds <run name>/run.json")
    parser.add_argument("--out", type=Path, help="default: <sweep-dir>/sweep_latency.csv")
    args = parser.parse_args(argv)

    rows = summary_rows(args.sweep_dir)
    out = args.out or args.sweep_dir / "sweep_latency.csv"
    write_summary(rows, out)
    print(f"wrote {out} ({len(rows)} configs)")


if __name__ == "__main__":
    main()
