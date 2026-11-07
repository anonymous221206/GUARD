# Reproducing the results

## 1. Check the method, no downloads

```bash
pip install -e ".[dev]"
pytest -q
```

## 2. Check the paper against the saved results

```bash
pip install -r requirements-reproduce.txt
python scripts/verify_submission.py /path/to/guard_iclr2027.tex
```

The LaTeX source is passed explicitly because it is not part of this repository.

## 3. Rerun from saved frozen-model outputs, no GPU

```bash
bash data/download_artifacts.sh dumps
bash scripts/rerun_all_results.sh
```

Set `RUN_TIMINGS=1` to repeat the latency measurements.

## 4. Retrain

Frozen models are retrained through `hosts/`, and `experiments/review_hme.py` runs the
HME comparator. IEMOCAP, NinaPro DB5, CMU-MOSEI and AVE must be obtained under their
original licences; `scripts/download/` documents the expected layouts.
