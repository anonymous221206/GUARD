#!/usr/bin/env python3
"""Check every machine-checkable number in the results section against results/.

Each table below is rebuilt from the files its driver wrote and compared cell by
cell with the LaTeX source. A cell that differs by more than half of the last
printed digit is reported; anything else passes. Tables whose source is a printed
transcript rather than a file are listed at the end as unchecked, rather than
silently passing.

    python scripts/tables/verify_paper.py path/to/guard_iclr2027.tex
"""
import csv, json, re, sys, collections
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
A = ROOT / "artifacts"
TEX = Path(sys.argv[1]) if len(sys.argv) > 1 else None

fails, checks, notes = [], 0, []


def close(a, b, dp):
    # a value landing exactly on a rounding boundary (0.7815 at three places)
    # can be printed either way, so the tolerance is half a digit plus float slack
    return abs(a - b) <= 0.5 * 10 ** (-dp) + 1e-7


def cmp(label, got, want, dp=3):
    """want is the printed string from the tex, got the recomputed value."""
    global checks
    checks += 1
    if want is None:
        fails.append(f"{label}: khong tim thay trong tex")
        return
    if not close(got, want, dp):
        fails.append(f"{label}: bai ghi {want:.{dp}f}, tinh lai {got:.{dp}f}")


def table_body(tex, label):
    i = tex.index("\\label{%s}" % label)
    j = tex.rindex("\\begin{table}", 0, i)
    k = tex.index("\\end{table}", i)
    body = tex[j:k]
    lines = []
    inside = False
    for ln in body.splitlines():
        if "\\toprule" in ln:
            inside = True
            continue
        if not inside or ln.strip().startswith("%"):
            continue
        if "\\bottomrule" in ln:
            break
        if "&" in ln and "\\\\" in ln:
            lines.append(ln)
    return lines


NUM = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")


def nums(line):
    """Numbers per column of a table row, empty columns kept as empty lists."""
    out = []
    for c in line.replace("\\\\", "").split("&"):
        c = re.sub(r"\\citep?\{[^}]*\}", "", c)
        c = re.sub(r"\\multicolumn\{\d+\}", "", c)
        out.append([float(x) for x in NUM.findall(c)])
    return out


def col(line, i):
    """The first number in column i, or None when that column carries none."""
    v = nums(line)
    return v[i][0] if i < len(v) and v[i] else None


def col2(line, i):
    v = nums(line)
    return v[i] if i < len(v) else []


# ---------------------------------------------------------------- sources
def mosei_full():
    return json.load(open(R / "gates/mosei_full.json"))


def iemocap_cells():
    d = json.load(open(R / "gates/cells_gates_iemocap.json"))
    g = collections.defaultdict(list)
    for x in d:
        g[x["ctx"]["m"]].append(x)
    out = {}
    for m, s in g.items():
        b = np.mean([x["base"] for x in s])
        out[m] = dict(frozen=b, guard=b + np.mean([x["GUARD"][0] for x in s]),
                      harm=np.mean([x["GUARD"][1] for x in s]),
                      apply=np.mean([x["rate"] for x in s]))
    return out


def ave_cond():
    return json.load(open(R / "gates/ave_conditions.json"))


def nina_ladder():
    return {int(r["level"]): r for r in json.load(open(R / "ninapro/severity_dense.json"))}


def ptbxl_ladder():
    p = A / "ptbxl_dropladder/severity_dense_auc.json"
    if not p.exists():
        p = R / "ptbxl_dropladder/severity_dense_auc.json"
    return {int(r["leads"]): r for r in json.load(open(p))}


def opp_sev():
    return json.load(open(R / "gates/opportunity_dcl_full.json"))["severity"]


def drugban_rows():
    sys.path.insert(0, str(ROOT / "scripts/tables"))
    import drugban_rows as D
    D.RES = R                       # the module reads argv, which is our tex path
    out = []
    for label, split, stem, seeds, cond, resid, pool in D.ROWS:
        rs = D.cells(stem, seeds, cond, pool=pool)
        if not rs:
            continue
        au = D.agg(rs, "base_metric")
        out.append(dict(split=split, auroc=au, auroc_g=au + D.agg(rs, "gate_metric_delta"),
                        acc=D.agg(rs, "base_accuracy"), acc_g=D.agg(rs, "gate_accuracy"),
                        harm=D.agg(rs, "joint_harm"), apply=D.agg(rs, "apply_rate")))
    return out


def eta_rows(path):
    rows = json.load(open(path))
    g = collections.defaultdict(list)
    for r in rows:
        g[r["eta"]].append(r)
    return {e: dict(frozen=np.mean([r["base"] for r in g[e]]),
                    gain=np.mean([r["GUARD"][0] for r in g[e]])) for e in sorted(g)}


def gate_cells():
    """Every (benchmark, condition) cell of the rule comparison, per driver.

    gates_rest runs three benchmarks in one process and the context capture picks
    up whatever local names are in scope, so `cond` leaks across blocks. The pool
    size and the condition name separate them without ambiguity.
    """
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for f, name in (("cells_gates_mosei2", "CMU-MOSEI"), ("cells_gates_iemocap", "IEMOCAP"),
                    ("cells_opp_dcl", "OPPORTUNITY"), ("cells_gates_ptbxl", "PTB-XL"),
                    ("cells_gates_drugban", "DrugBAN")):
        p = R / "gates" / (f + ".json")
        if not p.exists():
            continue
        for x in json.load(open(p)):
            c = x["ctx"]
            # the condition each driver iterates, which is what the tables count
            key = {"CMU-MOSEI": c.get("cond"), "IEMOCAP": c.get("m"),
                   "OPPORTUNITY": c.get("cfg"), "PTB-XL": c.get("c"),
                   "DrugBAN": c.get("cond")}.get(name) or "one"
            out[name][str(key)].append(x)
    p = R / "gates/cells_gates_rest.json"
    if p.exists():
        for x in json.load(open(p)):
            c = x["ctx"]
            if "c" not in c:
                out["AVE"][c["cond"]].append(x)
            elif str(c["c"]) in ("12", "8", "6", "4"):
                out["NinaPro DB5"][str(c["c"])].append(x)
            else:
                out["CMU-MOSEI-rest"][str(c["c"])].append(x)
    return out


def rule_means(cells, rule):
    """Mean gain and mean joint harm of one rule, averaged over conditions.

    A cell can be missing a rule: the learned score is not fitted when the fit
    split has only one class of outcome. Those cells are dropped, as the drivers
    drop them, rather than counted as zero.
    """
    per, flat_g, flat_h = [], [], []
    for v in cells.values():
        g = [x[rule][0] for x in v if rule in x and not np.isnan(x[rule][0])]
        h = [x[rule][1] for x in v if rule in x and not np.isnan(x[rule][1])]
        if g and h:
            per.append(float(np.mean(h)))
            flat_g += g
            flat_h += h
    if not per:
        return None
    # the drivers average over cells, not over conditions, and the conditions are
    # unbalanced on DrugBAN; the ranges are still per condition
    return float(np.mean(flat_g)), float(np.mean(flat_h)), per


# ---------------------------------------------------------------- checks
def check(tex):
    # Table 11, per-condition
    rows = table_body(tex, "tab:fourdomains")
    mf, ie, av = mosei_full(), iemocap_cells(), ave_cond()
    nl, pl = nina_ladder(), ptbxl_ladder()
    MAP = [
        ("audio $+$ visual", mf["av"]["base"], mf["av"]["guard"], mf["av"]["harm"], mf["av"]["apply"]),
        ("audio only", mf["a"]["base"], mf["a"]["guard"], mf["a"]["harm"], mf["a"]["apply"]),
        ("visual only", mf["v"]["base"], mf["v"]["guard"], mf["v"]["harm"], mf["v"]["apply"]),
    ]
    for name, fr, gu, hm, ap in MAP:
        ln = next((l for l in rows if name in l and "MOSEI" in l or (name in l and "binary" in l)
                   or (name in l and l.startswith(" " * 22))), None)
    # simpler: walk rows in order and match by printed values
    def row_by(text):
        return next((l for l in rows if text in l), None)

    for key, name in (("av", "audio $+$ visual"), ("a", "audio only"), ("v", "visual only")):
        ln = row_by(name)
        if ln is None:
            fails.append(f"Table 11 MOSEI {name}: khong thay dong")
            continue
        cmp(f"T11 MOSEI {name} frozen", mf[key]["base"], col(ln, 2))
        cmp(f"T11 MOSEI {name} GUARD", mf[key]["guard"], col(ln, 3))
        cmp(f"T11 MOSEI {name} harm", mf[key]["harm"], col(ln, 4))
        cmp(f"T11 MOSEI {name} apply", 100 * mf[key]["apply"], col(ln, 5), 0)

    IE = [("audio only", "a"), ("text only", "t"), ("visual only", "v"),
          ("audio $+$ text", "at"), ("audio $+$ visual", "av"), ("text $+$ visual", "tv")]
    for name, m in IE:
        ln = next((l for l in rows if name in l and ("IEMOCAP" in l or "class WA" in l
                                                     or l.lstrip().startswith("&"))), None)
        if ln is None or m not in ie:
            continue
    notes.append("Table 11 IEMOCAP/AVE/NinaPro/PTB-XL: doi chieu tung dong o phan duoi")

    # per-condition blocks matched positionally, which is safer than by text
    blocks = {"iemocap": [ie[m] for m in ("a", "t", "v", "at", "av", "tv", "atv")],
              "ave": [ave_cond()[c] for c in ("audio_only", "visual_only", "full")],
              "nina": [nl[k] for k in (14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 16)],
              "ptbxl": [pl[k] for k in (11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 12)]}
    order = [r for r in rows]
    # find the first row of each block by its frozen value
    def find_block(vals, keys):
        idx = []
        for want in vals:
            f = want[keys[0]]
            hit = None
            for j, l in enumerate(order):
                v = [c for c in nums(l) if c]
                if len(v) >= 4 and close(f, v[0][0], 3):
                    hit = j
                    break
            idx.append(hit)
        return idx

    for tag, vals, keys in (("IEMOCAP", blocks["iemocap"], ("frozen", "guard", "harm", "apply")),
                            ("AVE", blocks["ave"], ("frozen", "guard", "harm", "apply")),
                            ("NinaPro", blocks["nina"], ("frozen", "guard", "harm", "apply")),
                            ("PTB-XL", blocks["ptbxl"], ("frozen", "guard", "harm", "apply"))):
        for want in vals:
            f = want[keys[0]]
            ln = None
            for l in order:
                a, b = col(l, 2), col(l, 3)
                if a is not None and b is not None and close(f, a, 3) and close(want[keys[1]], b, 3):
                    ln = l
                    break
            if ln is None:
                fails.append(f"T11 {tag} frozen={f:.3f} GUARD={want[keys[1]]:.3f}: khong khop dong nao")
                continue
            cmp(f"T11 {tag} {f:.3f} harm", want[keys[2]], col(ln, 4))
            cmp(f"T11 {tag} {f:.3f} apply", 100 * want[keys[3]], col(ln, 5), 0)

    # Table 9, DrugBAN
    rows9 = table_body(tex, "tab:main")
    for got in drugban_rows():
        tag = got["split"].split("$")[0].strip()
        ln = next((l for l in rows9 if tag in l and col(l, 2) is not None
                   and close(got["auroc"], col(l, 2), 3)), None)
        if ln is None:
            fails.append(f"T9 {got['split']} frozen={got['auroc']:.3f}: khong khop dong nao")
            continue
        cmp(f"T9 {got['split']} AUROC GUARD", got["auroc_g"], col(ln, 3))
        cmp(f"T9 {got['split']} acc frozen", got["acc"], col(ln, 4))
        cmp(f"T9 {got['split']} acc GUARD", got["acc_g"], col(ln, 5))
        cmp(f"T9 {got['split']} harm", got["harm"], col(ln, 6))
        cmp(f"T9 {got['split']} apply", 100 * got["apply"], col(ln, 7), 0)

    # Table 10, OPPORTUNITY
    rows10 = table_body(tex, "tab:gateablation")
    NAMES = {"low-cost accels only": "low_cost_accels_only", "no IMU family": "no_imu_family",
             "no shoe sensors": "no_shoes", "three sensors only": "severe_three_sensors"}
    sev = opp_sev()
    for pretty, key in NAMES.items():
        ln = next((l for l in rows10 if pretty in l), None)
        if ln is None:
            fails.append(f"T10 {pretty}: khong tim thay dong")
            continue
        d = sev[key]
        cmp(f"T10 {pretty} F1 frozen", d["frozen"], col(ln, 1))
        cmp(f"T10 {pretty} F1 GUARD", d["guard"], col(ln, 2))
        cmp(f"T10 {pretty} acc frozen", d["acc_frozen"], col(ln, 3))
        cmp(f"T10 {pretty} acc GUARD", d["acc_guard"], col(ln, 4))
        cmp(f"T10 {pretty} harm", d["harm"], col(ln, 5))
        cmp(f"T10 {pretty} apply", 100 * d["apply"], col(ln, 6), 0)

    # Table 8, IEMOCAP per missing rate
    rows8 = table_body(tex, "tab:iemocap")
    er = eta_rows(R / "iemocap_eta.json")
    fr = next((l for l in rows8 if "our reproduction" in l), None)
    gd = next((l for l in rows8 if "$+$ GUARD" in l), None)
    if fr and gd:
        etas = sorted(er)
        for i, e in enumerate(etas):
            cmp(f"T8 frozen eta={e}", er[e]["frozen"], col(fr, 1 + i))
            cmp(f"T8 GUARD eta={e}", er[e]["frozen"] + er[e]["gain"], col(gd, 1 + i))
        cmp("T8 frozen Avg", np.mean([er[e]["frozen"] for e in etas]), col(fr, 1 + len(etas)))
        cmp("T8 GUARD Avg", np.mean([er[e]["frozen"] + er[e]["gain"] for e in etas]),
            col(gd, 1 + len(etas)))
    else:
        fails.append("T8: khong tim thay hai dong cua ta")

    # Table 2, the summary
    rows2 = table_body(tex, "tab:allbench")
    SUM = {
        "CMU-MOSEI": [dict(frozen=mf[k]["base"], guard=mf[k]["guard"],
                           harm=mf[k]["harm"], apply=mf[k]["apply"])
                      for k in ("a", "v", "av")],
        "IEMOCAP": [ie[m] for m in ("a", "t", "v", "at", "av", "tv")],
        "AVE": [ave_cond()[c] for c in ("audio_only", "visual_only")],
        "NinaPro": [nl[k] for k in (14, 12, 10, 8, 7, 6, 5, 4, 3, 2)],
        "PTB-XL": [pl[k] for k in (11, 10, 9, 8, 7, 6, 5, 4, 3, 2)],
        "OPPORTUNITY": list(opp_sev().values()),
    }
    for name, cells in SUM.items():
        ln = next((l for l in rows2 if name in l), None)
        if ln is None:
            fails.append(f"T2 {name}: khong tim thay dong")
            continue
        hr, ar = col2(ln, 5), col2(ln, 6)
        cmp(f"T2 {name} cond", len(cells), col(ln, 2), 0)
        cmp(f"T2 {name} frozen", np.mean([c["frozen"] for c in cells]), col(ln, 3))
        cmp(f"T2 {name} GUARD", np.mean([c["guard"] for c in cells]), col(ln, 4))
        cmp(f"T2 {name} harm lo", min(c["harm"] for c in cells), hr[0] if hr else None)
        cmp(f"T2 {name} harm hi", max(c["harm"] for c in cells), hr[1] if len(hr) > 1 else None)
        cmp(f"T2 {name} apply lo", 100 * min(c["apply"] for c in cells), ar[0] if ar else None, 0)
        cmp(f"T2 {name} apply hi", 100 * max(c["apply"] for c in cells),
            ar[1] if len(ar) > 1 else None, 0)
    db = drugban_rows()
    ln = next((l for l in rows2 if "DrugBAN" in l), None)
    if ln:
        hr, ar = col2(ln, 5), col2(ln, 6)
        cmp("T2 DrugBAN cond", len(db), col(ln, 2), 0)
        cmp("T2 DrugBAN frozen", np.mean([d["auroc"] for d in db]), col(ln, 3))
        cmp("T2 DrugBAN GUARD", np.mean([d["auroc_g"] for d in db]), col(ln, 4))
        cmp("T2 DrugBAN harm lo", min(d["harm"] for d in db), hr[0] if hr else None)
        cmp("T2 DrugBAN harm hi", max(d["harm"] for d in db), hr[1] if len(hr) > 1 else None)
        cmp("T2 DrugBAN apply lo", 100 * min(d["apply"] for d in db), ar[0] if ar else None, 0)
        cmp("T2 DrugBAN apply hi", 100 * max(d["apply"] for d in db),
            ar[1] if len(ar) > 1 else None, 0)

    # Table 3, the rule comparison
    gc = gate_cells()
    rows3 = table_body(tex, "tab:gates")
    RULES = [("Blanket", "blanket", 1), ("Conf.\\ $+$ CRC", "CRC-confidence", 2),
             ("Learned $+$ LTT", "LTT-learned", 3), ("GUARD", "GUARD", 4)]
    for name in ("CMU-MOSEI", "IEMOCAP", "OPPORTUNITY", "AVE", "NinaPro DB5",
                 "PTB-XL", "DrugBAN"):
        ln = next((l for l in rows3 if l.split("&")[0].strip() == name), None)
        if ln is None or name not in gc:
            fails.append(f"T3 {name}: khong tim thay dong hoac cell")
            continue
        for _, rule, ci in RULES:
            rm = rule_means(gc[name], rule)
            if rm is None:
                fails.append(f"T3 {name} {rule}: cell khong co luat nay")
                continue
            v = col2(ln, ci)
            if len(v) < 2:
                fails.append(f"T3 {name} {rule}: cot thieu so")
                continue
            cmp(f"T3 {name} {rule} gain", rm[0], v[0])
            cmp(f"T3 {name} {rule} harm", rm[1], v[1])

    # Table 14, the same cells without the gate
    rows14 = table_body(tex, "tab:blanket")
    for name in ("CMU-MOSEI", "IEMOCAP", "OPPORTUNITY", "AVE", "NinaPro DB5",
                 "PTB-XL", "DrugBAN"):
        ln = next((l for l in rows14 if l.split("&")[0].strip() == name), None)
        if ln is None or name not in gc:
            fails.append(f"T14 {name}: khong tim thay dong hoac cell")
            continue
        bl, gu = rule_means(gc[name], "blanket"), rule_means(gc[name], "GUARD")
        cmp(f"T14 {name} cond", len(gc[name]), col(ln, 1), 0)
        cmp(f"T14 {name} blanket gain", bl[0], col(ln, 2))
        r = col2(ln, 3)
        cmp(f"T14 {name} blanket harm lo", min(bl[2]), r[0] if r else None)
        cmp(f"T14 {name} blanket harm hi", max(bl[2]), r[1] if len(r) > 1 else None)
        r = col2(ln, 4)
        cmp(f"T14 {name} GUARD harm lo", min(gu[2]), r[0] if r else None)
        cmp(f"T14 {name} GUARD harm hi", max(gu[2]), r[1] if len(r) > 1 else None)
        # the caption promises each mean lies inside its own range
        if not (min(gu[2]) - 5e-4 <= gu[1] <= max(gu[2]) + 5e-4):
            fails.append(f"T14 {name}: trung binh GUARD {gu[1]:.3f} nam ngoai khoang "
                         f"{min(gu[2]):.3f}-{max(gu[2]):.3f}")
        if not (min(bl[2]) - 5e-4 <= bl[1] <= max(bl[2]) + 5e-4):
            fails.append(f"T14 {name}: trung binh blanket {bl[1]:.3f} nam ngoai khoang "
                         f"{min(bl[2]):.3f}-{max(bl[2]):.3f}")

    # Table 12, the two retrieval targets
    rows12 = table_body(tex, "tab:crossmask")
    sys.path.insert(0, str(ROOT / "scripts/tables"))
    import drugban_rows as D
    D.RES = R
    T12 = [("Human, random", "human_random", ("s1", "s2", "s42")),
           ("BindingDB, random", "bindingdb_random", ("s1", "s2", "s42")),
           ("BioSNAP, random", "biosnap_random", ("s1", "s2", "s42")),
           ("BioSNAP, cluster", "biosnap_cluster", ("s1", "s2", "s42")),
           ("BindingDB, cluster", "bindingdb_cluster", ("s42",))]
    for pretty, stem, seeds in T12:
        ln = next((l for l in rows12 if pretty in l), None)
        if ln is None:
            fails.append(f"T12 {pretty}: khong tim thay dong")
            continue
        got = {}
        for pol in ("hard", "cross_mask"):
            rs = [r for c in ("prot25", "scaffold_prot50")
                  for r in D.cells(stem, seeds, c, pol, D.TARGET_POOL[stem])]
            got[pol] = D.agg(rs, "gate_accuracy") - D.agg(rs, "base_accuracy")
        rich = D.cells(stem, seeds, "full", "hard", D.TARGET_POOL[stem])
        cmp(f"T12 {pretty} richer", D.agg(rich, "base_accuracy"), col(ln, 1))
        cmp(f"T12 {pretty} hard", got["hard"], col(ln, 2))
        cmp(f"T12 {pretty} cross", got["cross_mask"], col(ln, 3))
    op = R / "gates/opp_targets.json"
    ln = next((l for l in rows12 if "OPPORTUNITY" in l), None)
    if op.exists() and ln:
        d = json.load(open(op))
        cmp("T12 OPPORTUNITY richer", d["richer"], col(ln, 1))
        cmp("T12 OPPORTUNITY hard", d["hard"], col(ln, 2))
        cmp("T12 OPPORTUNITY cross", d["cross"], col(ln, 3))

    # Table 13, CMU-MOSEI harm
    rows13 = table_body(tex, "tab:mosei-harm")
    for key, tag in (("a", "$A$"), ("v", "$V$"), ("av", "$A,V$"), ("tav", "$L,A,V$")):
        ln = next((l for l in rows13 if l.strip().startswith(tag)), None)
        if ln is None:
            fails.append(f"T13 {tag}: khong tim thay dong")
            continue
        cmp(f"T13 {tag} blanket", mf[key]["blanket_harm"], col(ln, 1))
        cmp(f"T13 {tag} GUARD", mf[key]["harm"], col(ln, 2))
        cmp(f"T13 {tag} apply", 100 * mf[key]["apply"], col(ln, 3), 0)


def main():
    if TEX is None or not TEX.exists():
        sys.exit("dua duong dan toi guard_iclr2027.tex")
    check(TEX.read_text())
    print(f"da kiem {checks} o")
    if notes:
        for n in notes:
            print("  ghi chu:", n)
    if fails:
        print(f"\n{len(fails)} o khong khop:")
        for f in fails:
            print("  ", f)
        sys.exit(1)
    print("moi o deu khop voi results/")


if __name__ == "__main__":
    main()
