from pathlib import Path
p = Path('experiments/ltt_run.py'); s = p.read_text()

old = """    g={'__name__':'__main__','__file__':drv+'.py'}
    exec(compile(open(drv+'.py').read(),drv+'.py','exec'),g)"""
new = """    src=_HERE/(drv+'.py')
    g={'__name__':'__main__','__file__':str(src)}
    exec(compile(src.read_text(),str(src),'exec'),g)"""
assert s.count(old) == 1, 'khong thay khoi exec'
s = s.replace(old, new)

anchor = "import gates_core, gates_ltt"
assert s.count(anchor) == 1
s = s.replace(anchor, "_HERE = Path(__file__).resolve().parent\nimport gates_core, gates_ltt")
if 'from pathlib import Path' not in s:
    s = s.replace('import sys, json, os', 'import sys, json, os\nfrom pathlib import Path')

out = "results/gates/ltt_cells.json"
s = s.replace("open('ltt_cells.json','w')", f"open(str(_HERE.parent / '{out}'),'w')")
p.write_text(s)
print('sua ltt_run: driver va output theo duong dan tuyet doi cua release')
print('con mo tuong doi:', [l.strip() for l in s.split(chr(10)) if "open('" in l][:3])
