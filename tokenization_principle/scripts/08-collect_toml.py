#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Any

import toml


def main():
    parser = argparse.ArgumentParser(description="Collect per-config output.toml files into a single results.toml.")
    parser.add_argument("--prepared-dir", default="data/prepared", type=Path)
    parser.add_argument("--output", default="results.toml", type=Path)
    args = parser.parse_args()

    out: dict[str, dict[str, Any]] = {}
    for path in sorted(args.prepared_dir.glob("*/*/output.toml")):
        lang_pair = path.parent.parent.name
        signature = path.parent.name
        with open(path, "r", encoding="utf-8") as f:
            out.setdefault(lang_pair, {})[signature] = toml.load(f)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        toml.dump(out, f)

    print(f"Collected {sum(len(v) for v in out.values())} configs into {args.output}")


if __name__ == "__main__":
    main()
