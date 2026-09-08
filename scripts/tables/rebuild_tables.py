#!/usr/bin/env python3
"""Print the numeric body of every results table, straight from results/.

verify_paper.py says which cells disagree; this writes the replacements, so a
rerun is transcribed by the same code that checks it rather than by hand.
"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/tables"))
import verify_paper as V

mf, ie, av = V.mosei_full(), V.iemocap_cells(), V.ave_cond()
nl, pl, op = V.nina_ladder(), V.ptbxl_ladder(), V.opp_sev()
db = V.drugban_rows()
gc = V.gate_cells()
f3 = lambda x: f"${x:.3f}$"
b3 = lambda x, w: f"$\\mathbf{{{x:.3f}}}$" if w else f"${x:.3f}$"


def block(rows, head1, head2, resid=None):
    out = []
    for i, (label, fr, gu, hm, ap) in enumerate(rows):
        lead = head1 if i == 0 else (head2 if i == 1 else "")
        r = resid[i] if resid else "---"
        out.append(f"{lead:<21} & {label:<19} & {b3(fr, fr > gu)} & {b3(gu, gu > fr)} & "
                   f"${hm:.3f}$ & ${100*ap:.0f}$ & {r} \\\\")
    return out


print("%%%% ---- Table 11 ----")
rows = [("audio $+$ visual", mf['av']['base'], mf['av']['guard'], mf['av']['harm'], mf['av']['apply']),
        ("audio only", mf['a']['base'], mf['a']['guard'], mf['a']['harm'], mf['a']['apply']),
        ("visual only", mf['v']['base'], mf['v']['guard'], mf['v']['harm'], mf['v']['apply']),
        ("\\emph{intact}", mf['tav']['base'], mf['tav']['guard'], mf['tav']['harm'], mf['tav']['apply'])]
print("\n".join(block(rows, "CMU-MOSEI (CMAD)", "\\emph{binary accuracy}")))
print("\\midrule")
IE = [("audio only", 'a'), ("text only", 't'), ("visual only", 'v'), ("audio $+$ text", 'at'),
      ("audio $+$ visual", 'av'), ("text $+$ visual", 'tv'), ("\\emph{intact}", 'atv')]
rows = [(n, ie[m]['frozen'], ie[m]['guard'], ie[m]['harm'], ie[m]['apply']) for n, m in IE]
print("\n".join(block(rows, "IEMOCAP (MoMKE)", "\\emph{$4$-class WA}",
                      ["$+0.082$"] + ["---"] * 6)))
print("\\midrule")
rows = [("audio only", av['audio_only']['frozen'], av['audio_only']['guard'], av['audio_only']['harm'], av['audio_only']['apply']),
        ("visual only", av['visual_only']['frozen'], av['visual_only']['guard'], av['visual_only']['harm'], av['visual_only']['apply']),
        ("\\emph{intact}", av['full']['frozen'], av['full']['guard'], av['full']['harm'], av['full']['apply'])]
print("\n".join(block(rows, "AVE (AV-att)", "\\emph{$29$-way acc.}", ["$+0.132$", "$+0.174$", "---"])))
print("\\midrule")
rows = [(f"${k}$ electrodes" if k != 16 else "\\emph{intact}",
         nl[k]['frozen'], nl[k]['guard'], nl[k]['harm'], nl[k]['apply'])
        for k in (14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 16)]
print("\n".join(block(rows, "NinaPro DB5 (sEMG CNN)", "\\emph{accuracy}")))
print("\\midrule")
rows = [(f"${k}$ leads" if k != 12 else "\\emph{intact}",
         pl[k]['frozen'], pl[k]['guard'], pl[k]['harm'], pl[k]['apply'])
        for k in (11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 12)]
print("\n".join(block(rows, "PTB-XL (resnet1d\\_wang)", "\\emph{macro AUC}")))

print("\n%%%% ---- Table 2 ----")
SUM = [("DrugBAN ($3$ datasets)", "AUROC",
        [dict(frozen=d['auroc'], guard=d['auroc_g'], harm=d['harm'], apply=d['apply']) for d in db]),
       ("OPPORTUNITY (DeepConvLSTM)", "wtd.\\ F1", list(op.values())),
       ("CMU-MOSEI (CMAD)", "binary acc.",
        [dict(frozen=mf[k]['base'], guard=mf[k]['guard'], harm=mf[k]['harm'], apply=mf[k]['apply']) for k in ('a', 'v', 'av')]),
       ("IEMOCAP (MoMKE)", "$4$-class WA", [ie[m] for m in ('a', 't', 'v', 'at', 'av', 'tv')]),
       ("AVE (AV-att)", "$29$-way acc.", [av[c] for c in ('audio_only', 'visual_only')]),
       ("NinaPro DB5 (sEMG CNN)", "accuracy", [nl[k] for k in (14, 12, 10, 8, 7, 6, 5, 4, 3, 2)]),
       ("PTB-XL (resnet1d\\_wang)", "macro AUC", [pl[k] for k in (11, 10, 9, 8, 7, 6, 5, 4, 3, 2)])]
for name, metric, cells in SUM:
    fr = np.mean([c['frozen'] for c in cells]); gu = np.mean([c['guard'] for c in cells])
    hs = [c['harm'] for c in cells]; ap = [c['apply'] for c in cells]
    print(f"{name:<26} & {metric:<13} & ${len(cells)}$ & ${fr:.3f}$ & $\\mathbf{{{gu:.3f}}}$ & "
          f"${min(hs):.3f}$--${max(hs):.3f}$ & ${100*min(ap):.0f}$--${100*max(ap):.0f}$ \\\\")

print("\n%%%% ---- Table 3 ----")
RULES = [("blanket", "Blanket"), ("CRC-confidence", "Conf.\\ $+$ CRC"),
         ("LTT-learned", "Learned $+$ LTT"), ("GUARD", "GUARD")]
for name in ("CMU-MOSEI", "IEMOCAP", "OPPORTUNITY", "AVE", "NinaPro DB5", "PTB-XL", "DrugBAN"):
    cells = []
    for rule, _ in RULES:
        rm = V.rule_means(gc[name], rule)
        star = "^{\\ast}" if rm[1] > 0.2 else ""
        cells.append((rm[0], rm[1], star))
    lo = min(c[1] for c in cells)
    print(f"{name} & " + " & ".join(
        f"${g:+.3f}$/$" + (f"\\mathbf{{{h:.3f}}}" if abs(h - lo) < 1e-12 and i > 0 else f"{h:.3f}{s}") + "$"
        for i, (g, h, s) in enumerate(cells)) + " \\\\")

print("\n%%%% ---- Table 13 ----")
for key, tag in (('a', "$A$"), ('v', "$V$"), ('av', "$A,V$"), ('tav', "$L,A,V$")):
    d = mf[key]
    print(f"{tag:<8} & ${d['blanket_harm']:.3f}$ & $\\mathbf{{{d['harm']:.3f}}}$ & ${100*d['apply']:.0f}$ \\\\")

print("\n%%%% ---- Table 14 ----")
for name in ("CMU-MOSEI", "IEMOCAP", "OPPORTUNITY", "AVE", "NinaPro DB5", "PTB-XL", "DrugBAN"):
    bl, gu = V.rule_means(gc[name], "blanket"), V.rule_means(gc[name], "GUARD")
    print(f"{name:<12} & ${len(gc[name])}$ & ${bl[0]:+.3f}$ & "
          f"${min(bl[2]):.3f}$--${max(bl[2]):.3f}$ & "
          f"$\\mathbf{{{min(gu[2]):.3f}}}$--$\\mathbf{{{max(gu[2]):.3f}}}$ \\\\")

print("\n%%%% ---- Table 10 ----")
NAMES = [("low-cost accels only", "low_cost_accels_only"), ("no IMU family", "no_imu_family"),
         ("no shoe sensors", "no_shoes"), ("three sensors only", "severe_three_sensors")]
for pretty, key in NAMES:
    d = op[key]
    print(f"{pretty:<21} & ${d['frozen']:.3f}$ & $\\mathbf{{{d['guard']:.3f}}}$ & "
          f"${d['acc_frozen']:.3f}$ & $\\mathbf{{{d['acc_guard']:.3f}}}$ & "
          f"${d['harm']:.3f}$ & ${100*d['apply']:.0f}$ \\\\")
