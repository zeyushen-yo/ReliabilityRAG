#!/usr/bin/env python3
"""
fix_choices.py  –  ensure `"incorrect answer"` is present in `"choices"`
Usage:
    python fix_choices.py input.json output.json
"""

import json
import sys
from pathlib import Path

def main(in_path: str, out_path: str) -> None:
    data = json.loads(Path(in_path).read_text())

    for entry in data:
        choices          = entry["choices"]
        correct_idx      = int(entry["choices answer"])       # stored as string -> int
        incorrect_ans    = entry["incorrect answer"]

        # already present ⇒ nothing to do
        if incorrect_ans in choices:
            continue

        # find a slot that is *not* the correct‑answer slot to overwrite
        for i, choice in enumerate(choices):
            if i != correct_idx:
                choices[i] = incorrect_ans
                break
        else:
            # Should never happen unless `choices` is malformed
            raise ValueError(f"No writable slot found in entry id {entry.get('id')}")

        entry["choices"] = choices  # (list is already mutated, but explicit is better)

    Path(out_path).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Fixed file written to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python fix_choices.py input.json output.json")
    main(sys.argv[1], sys.argv[2])
