#!/usr/bin/env python3
"""Check the displayed HME table against the verified paired results."""
import json
import re
import sys
from pathlib import Path


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit('usage: verify_hme_table.py MANUSCRIPT.tex [paired_comparison.json]')
    root = Path(__file__).resolve().parents[2]
    tex = Path(sys.argv[1]).read_text()
    path = (Path(sys.argv[2]) if len(sys.argv) == 3 else
            root/'results/external_review_20260917/hme/paired_comparison.json')
    result = json.loads(path.read_text())
    assert result['status'] == 'PASS' and result['verified_metric_fields'] == 70
    start = tex.index('\\label{tab:hme-paired}')
    body = tex[start:tex.index('\\end{table}', start)]
    names = {'t': '$L$', 'a': '$A$', 'v': '$V$', 'ta': '$LA$', 'tv': '$LV$',
             'av': '$AV$', 'tav': 'Intact ($LAV$)',
             'incomplete_six': 'Six incomplete masks',
             'language_absent': 'Language absent ($A,V,AV$)'}
    fields = 0
    for key, name in names.items():
        line = next(row for row in body.splitlines() if row.startswith(name+' & '))
        actual = re.findall(r'\$([\d.]+)/([\d.]+)\$', line)
        expected = [tuple(f'{x:.1f}' for x in result['rows'][key][method])
                    for method in ('HME', 'CMAD', 'CMAD_GUARD')]
        assert actual == expected, (key, actual, expected)
        fields += len(expected)*2
    print(f'PASS: {fields} displayed HME/CMAD/GUARD values match {path}')


if __name__ == '__main__':
    main()

