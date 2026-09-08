from pathlib import Path
p = Path('experiments/mixed_deploy.py'); s = p.read_text()

a = "'GUARD-LTT','GUARD-mask-LTT')"
b = "'GUARD-LTT','GUARD-mask-LTT','GUARD-soft-LTT')"
assert s.count(a) == 1
s = s.replace(a, b)

s = s.replace("n=len(y); res=collections", "n=len(y); diag=[]; res=collections")

a2 = "    for k in RULES:\n        if k in row: res[k].append(row[k])\n"
assert s.count(a2) == 1
s = s.replace(a2, a2 + "    diag.append((row.get('_eta'), row.get('_lambda'), row['_meta']['apply']))\n")

a3 = "print()\nprint('harm theo tung pattern"
assert s.count(a3) == 1
s = s.replace(a3,
    "print()\n"
    "print('eta chon tren fit :', [d[0] for d in diag])\n"
    "print('lambda LTT chon   :', [None if d[1] is None else round(d[1],3) for d in diag])\n"
    "print('apply rate GUARD  :', [round(d[2],3) for d in diag])\n"
    "print()\nprint('harm theo tung pattern")

p.write_text(s)
print('them soft + chan doan')
