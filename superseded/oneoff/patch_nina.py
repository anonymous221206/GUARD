from pathlib import Path
p = Path('experiments/mixed_nina.py'); s = p.read_text()
a = "'LTT-mask-learned','GUARD-mask')"
b = ("'LTT-mask-learned','GUARD-mask','GUARD-LTT','GUARD-mask-LTT',"
     "'GUARD-exp-LTT','GUARD-mask-exp-LTT','GUARD-phat-LTT','GUARD-mask-phat-LTT')")
assert s.count(a) == 1
p.write_text(s.replace(a, b))
print('nina: them rule hybrid')
