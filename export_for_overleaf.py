#!/usr/bin/env python3
# Collect figures from results/* into for_overleaf/figs and zip them.

from __future__ import annotations
import shutil
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import itertools

# Source folders with figures
SRC_DIRS = [
    Path("results"),
    Path("results/stats"),
    Path("results/sensitivity"),
]

# Figure extensions to include
EXTS = {".png", ".pdf", ".jpg", ".jpeg"}

DEST_ROOT = Path("for_overleaf")
DEST_FIGS = DEST_ROOT / "figs"
ZIP_PATH = DEST_ROOT / "figs_for_overleaf.zip"

def main() -> None:
    DEST_FIGS.mkdir(parents=True, exist_ok=True)

    copied = []
    for p in itertools.chain.from_iterable(d.glob("*") for d in SRC_DIRS if d.exists()):
        if p.suffix.lower() in EXTS and p.is_file():
            target = DEST_FIGS / p.name
            shutil.copy2(p, target)
            copied.append(target)

    # Create a zip with just the figs directory
    with ZipFile(ZIP_PATH, "w", ZIP_DEFLATED) as z:
        for f in copied:
            z.write(f, arcname=f.relative_to(DEST_ROOT))

    print(f"Copied {len(copied)} figures to {DEST_FIGS}")
    print(f"ZIP ready: {ZIP_PATH}")

if __name__ == "__main__":
    main()
