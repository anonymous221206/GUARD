#!/usr/bin/env python3
"""The body of the CMU-MOSEI comparison table, printed as LaTeX.

Our six rows are read from ``results/gates/mosei_hosts.json``, written by
``experiments/mosei_hosts.py``. The nine rows above them are quoted from the
papers named beside them and are literals here; they are never recomputed, and
this script is the only place they are written down. Bolding is decided across
the whole table, so it cannot drift when our rows move.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/gates/mosei_hosts.json"
MASKS = ["t", "a", "v", "ta", "tv", "av", "tav"]

# acc/F1 per observed-modality subset, in the order of MASKS, then the paper's own Avg.
QUOTED = [
    (r"MCTN~\citep{pham2019mctn}",     [(82.6, 82.8), (62.7, 54.5), (62.6, 57.1), (83.5, 83.3), (83.2, 83.2), (63.7, 62.7), (84.2, 84.2)], (74.6, 72.5)),
    (r"MMIN~\citep{zhao2021mmin}",     [(82.3, 82.4), (58.9, 59.5), (59.3, 60.0), (83.7, 83.3), (83.8, 83.4), (63.5, 61.9), (84.3, 84.2)], (73.7, 73.5)),
    (r"GCNet~\citep{lian2023gcnet}",   [(83.0, 83.2), (60.2, 60.3), (61.9, 61.6), (84.3, 84.4), (84.3, 84.4), (64.1, 57.2), (85.2, 85.1)], (74.7, 73.7)),
    (r"IMDer~\citep{wang2023imder}",   [(84.5, 84.5), (63.8, 60.6), (63.9, 63.6), (85.1, 85.1), (85.0, 85.0), (64.9, 63.5), (85.1, 85.1)], (76.0, 75.3)),
    (r"DiCMoR~\citep{wang2023dicmor}", [(84.2, 84.3), (62.9, 60.4), (63.6, 63.6), (85.0, 84.9), (84.9, 84.9), (65.2, 64.4), (85.1, 85.1)], (75.8, 75.4)),
    (r"MPLMM~\citep{guo2024mplmm}",    [(84.6, 84.6), (61.8, 49.4), (62.2, 53.2), (85.1, 85.1), (85.2, 85.1), (63.2, 51.0), (85.3, 85.3)], (75.3, 70.5)),
    (r"UMDF~\citep{li2024umdf}",       [(85.3, 85.2), (62.9, 59.4), (62.1, 62.4), (85.3, 85.3), (85.3, 85.2), (65.4, 62.8), (85.7, 85.6)], (76.0, 75.1)),
    (r"CorrKD~\citep{li2024corrkd}",   [(85.3, 85.1), (61.9, 56.8), (63.0, 62.5), (85.3, 85.2), (85.6, 85.5), (64.5, 62.3), (85.7, 85.6)], (75.9, 74.7)),
    (r"MMANet~\citep{wei2023mmanet}",  [(84.6, 84.6), (62.9, 48.5), (63.7, 63.1), (84.8, 84.8), (85.3, 85.2), (64.5, 60.9), (85.3, 85.2)], (75.9, 73.2)),
]
HOSTS = [("CMAD", r"CMAD~\citep{zhuang2025cmad}$^{\dagger}$"),
         ("TMDC", r"TMDC~\citep{zhuang2026tmdc}$^{\dagger}$"),
         ("MoMKE", r"MoMKE~\citep{xu2024momke}$^{\dagger}$")]


def main():
    if not SRC.exists():
        sys.exit(f"thieu {SRC}: chay experiments/mosei_hosts.py truoc")
    d = json.load(open(SRC))
    rows = []
    for label, cells, avg in QUOTED:
        rows.append((label, cells, avg, False))
    for key, label in HOSTS:                       # frozen reproductions
        cells = [(d[f"{key}|{m}"][2], d[f"{key}|{m}"][3]) for m in MASKS]
        rows.append((label, cells, None, False))
    for key, _ in HOSTS:                           # the same hosts, corrected
        cells = [(d[f"{key}|{m}"][0], d[f"{key}|{m}"][1]) for m in MASKS]
        rows.append((f"{key} $+$ GUARD", cells, None, True))

    filled = []
    for label, cells, avg, ours in rows:
        if avg is None:
            avg = (sum(c[0] for c in cells) / len(cells),
                   sum(c[1] for c in cells) / len(cells))
        filled.append((label, cells + [avg], ours))

    ncol = len(MASKS) + 1
    best = [(max(r[1][j][0] for r in filled), max(r[1][j][1] for r in filled))
            for j in range(ncol)]

    def cell(v, b):
        # ties are bolded too: two methods can genuinely reach the same number
        f = f"{v:.1f}"
        return f"\\textbf{{{f}}}" if abs(v - b) < 0.05 else f

    print("% sinh boi scripts/tables/mosei_rows.py")
    for i, (label, cells, ours) in enumerate(filled):
        if i == len(QUOTED) + len(HOSTS):
            print(r"\midrule")
        body = " & ".join(f"{cell(a, best[j][0])}/{cell(f, best[j][1])}"
                          for j, (a, f) in enumerate(cells))
        print(f"{label:<40} & {body} \\\\")


if __name__ == "__main__":
    main()
