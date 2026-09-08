#!/usr/bin/env python3
"""Re-score the DrugBAN protein ladder with the current gate.

The eleven frozen dumps are expensive to regenerate: they need the DrugBAN
model and its graph stack. Scoring them does not, so this runs only the GUARD
half of ``drugban_protladder_v2.py`` and rewrites ``guard_results.json``, which
is what the severity figure reads. The dumps themselves are never touched.
"""
import importlib.util, json, sys
from pathlib import Path

B = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "protladder", B / "scripts/drugban_protladder_v2.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

missing = [p for p in mod.PCTS if not (mod.OUT / mod.file_name(p)).exists()]
if missing:
    sys.exit(f"missing frozen dumps for {missing}; run drugban_protladder_v2.py first")

out = mod.OUT / "guard_results.json"
out.write_text(json.dumps(mod.run_guard(), indent=2) + "\n")
print("wrote", out)
