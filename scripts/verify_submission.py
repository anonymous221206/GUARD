#!/usr/bin/env python3
"""Run every manuscript/result synchronization check."""
import subprocess
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        raise SystemExit('usage: verify_submission.py MANUSCRIPT.tex')
    root = Path(__file__).resolve().parents[1]
    tex = Path(sys.argv[1]).resolve()
    commands = [
        [sys.executable, root/'scripts/tables/verify_paper.py', tex],
        [sys.executable, root/'scripts/tables/verify_review_additions.py', tex,
         root/'results/review_20260917'],
        [sys.executable, root/'scripts/tables/verify_hme_table.py', tex],
    ]
    for command in commands:
        print('+', ' '.join(map(str, command)), flush=True)
        subprocess.run(command, cwd=root, check=True)
    print('PASS: all submission/result synchronization checks completed')


if __name__ == '__main__':
    main()

