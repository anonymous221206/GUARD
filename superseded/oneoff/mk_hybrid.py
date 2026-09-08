from pathlib import Path
src = Path('experiments/mixed_deploy.py').read_text()

src = src.replace("mixed_deploy.json", "hybrid_mosei.json")
old = "'GUARD-LTT','GUARD-mask-LTT','GUARD-soft-LTT')"
new = ("'GUARD-LTT','GUARD-mask-LTT','GUARD-soft-LTT',\n"
       "       'GUARD-exp-LTT','GUARD-mask-exp-LTT','GUARD-phat-LTT','GUARD-mask-phat-LTT')")
assert src.count(old) == 1
src = src.replace(old, new)
src = src.replace('"""Mixed availability: one calibration set, many conditions at deployment.',
                  '"""Hybrid scores: same certificate machinery, ranked by different aggregations.')
Path('experiments/hybrid_mosei.py').write_text(src)
print('tao experiments/hybrid_mosei.py')
