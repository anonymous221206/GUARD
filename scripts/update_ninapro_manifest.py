#!/usr/bin/env python3
"""Replace the NinaPro MISSING row after preserving the old manifest."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path("$WORKSPACE")
ARTIFACT = ROOT / "artifacts/ninapro_cnn"
MANIFEST = ROOT / "artifacts/MANIFEST.md"
STAGED = ROOT / "artifacts/MANIFEST.ninapro.new.md"
BACKUP = ROOT / "artifacts/.superseded/MANIFEST.pre_ninapro.md"
OLD_ROW = "| ninapro_cnn | N/A | N/A | 0 | N/A | MISSING: checkpoint and pipeline dump |"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    if not ARTIFACT.is_dir():
        raise FileNotFoundError(ARTIFACT)
    if STAGED.exists() or BACKUP.exists():
        raise FileExistsError("refusing to overwrite staged/backup manifest")
    source = MANIFEST.read_text()
    if source.count(OLD_ROW) != 1:
        raise ValueError("expected exactly one NinaPro MISSING row")
    files = sorted(path for path in ARTIFACT.rglob("*") if path.is_file())
    entries = []
    total = 0
    for path in files:
        size = path.stat().st_size
        total += size
        entries.append(f"{path.relative_to(ARTIFACT)}:{size}:{digest(path)}")
    row = (
        "| ninapro_cnn | raw $WORKSPACE/data/ninapro; "
        "trainer GUARD/scripts/train_ninapro_retrained.py; exporter "
        "GUARD/scripts/export_ninapro_dumps.py | "
        "$WORKSPACE/artifacts/ninapro_cnn | "
        f"{total} | {'<br>'.join(entries)} | "
        "complete dump + 100 retrained checkpoints (seed 0/1, 10 subjects, "
        "rungs 16/12/8/6/4); NOT the original paper checkpoints; maximum "
        "absolute printed-cell deviation 0.0193422621 exceeds maximum inter-seed "
        "gap 0.0046207674, indicating systematic difference beyond measured "
        "two-seed variation |"
    )
    STAGED.write_text(source.replace(OLD_ROW, row))
    MANIFEST.rename(BACKUP)
    STAGED.rename(MANIFEST)
    print(f"files={len(files)} bytes={total} manifest_bytes={MANIFEST.stat().st_size}")


if __name__ == "__main__":
    main()
