"""Pick the fewest-holes window of each sequence and write it into the sequences config."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pim.data.tum import load_sequence  # noqa: E402
from pim.data.windows import WindowChoice, count_holes, select_window  # noqa: E402

DEFAULT_COUNT = 300
SEQUENCE_KEY = re.compile(r"^  (\S+):\s*$")
WINDOW_LINE = re.compile(r"^(\s+)window:\s*\{.*\}\s*(#.*)?$")


def set_windows(config_text: str, windows: dict[str, WindowChoice]) -> str:
    """Rewrite only the `window:` line of each listed sequence; everything else is kept verbatim."""
    lines = config_text.splitlines(keepends=True)
    in_sequences = False
    current = None
    replaced: dict[str, int] = {}
    for i, line in enumerate(lines):
        if line.strip() and not line.startswith((" ", "#")):
            in_sequences = line.rstrip() == "sequences:"
            current = None
            continue
        key = SEQUENCE_KEY.match(line) if in_sequences else None
        if key:
            current = key.group(1)
            continue
        window = WINDOW_LINE.match(line)
        if window and current in windows:
            choice = windows[current]
            ending = "\n" if line.endswith("\n") else ""
            lines[i] = f"{window.group(1)}window: {{start: {choice.start}, count: {choice.count}}}{ending}"
            replaced[current] = replaced.get(current, 0) + 1

    wrong = {name: replaced.get(name, 0) for name in windows if replaced.get(name, 0) != 1}
    if wrong:
        raise ValueError(f"expected exactly one inline `window: {{...}}` line per sequence, found {wrong}")
    updated = "".join(lines)
    parsed = yaml.safe_load(updated)["sequences"]
    for name, choice in windows.items():
        if parsed[name]["window"] != {"start": choice.start, "count": choice.count}:
            raise RuntimeError(f"rewritten config does not parse back to the chosen window for {name}")
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="folder containing the sequence folders")
    parser.add_argument("--sequences-config", type=Path, default=Path("configs/sequences.yaml"))
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    args = parser.parse_args()

    config_text = args.sequences_config.read_text()
    names = list(yaml.safe_load(config_text)["sequences"])
    windows = {}
    for name in names:
        sequence = load_sequence(args.data_root / name, args.sequences_config)
        times = [f.timestamp for f in sequence.frames]
        choice = select_window(times, args.count)
        windows[name] = choice
        print(f"{name}: start {choice.start}, count {choice.count}, "
              f"holes {choice.holes} in window / {count_holes(times)} in sequence "
              f"({len(times)} frames), "
              f"t {times[choice.start]:.6f} -> {times[choice.start + choice.count - 1]:.6f}")

    args.sequences_config.write_text(set_windows(config_text, windows))
    print(f"wrote {len(windows)} windows to {args.sequences_config}")


if __name__ == "__main__":
    main()
