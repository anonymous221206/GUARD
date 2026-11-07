# GUARD

Risk-controlled post-hoc correction for a **frozen** model deployed with missing
modalities. GUARD leaves the model's weights fixed, constructs a candidate correction,
and gates it with a finite-sample bound on joint harm.

```python
import numpy as np
from guard import HostOutputs, random_split, run

split  = random_split(np.arange(n), seed=0)
result = run(HostOutputs(probs, features, labels), split,
             target="hard", alpha=0.2, delta=0.05)

result.gate_metric_delta   # gain
result.joint_harm          # realized test statistic
```

## What is guaranteed

Conditioned on all pre-calibration fitting, GUARD bounds
`P(Delta_loss > delta AND applied) <= alpha` marginally over the calibration draw and
a fresh deployment point, when they are exchangeable. The bound is joint, not
conditional on intervention, and it is a population statement, so a finite test split
can exceed `alpha`.

## Install and reproduce

```bash
pip install -e ".[dev]"
pytest -q
```

`docs/REPRODUCING.md` gives the steps, `docs/PAPER_MAP.md` maps each table and figure
to its driver and saved outputs, and `docs/CHANGELOG.md` lists the changes since the
submitted version.

## Layout

```
src/guard/    the method
hosts/        adapters for the frozen models
experiments/  paper drivers
scripts/      rerun, verification, download and figure helpers
results/      saved outputs behind the paper
tests/        the guarantee checked on synthetic data
data/         download scripts; no data is committed
docs/         reproduction notes
```

Released under the MIT licence. Datasets and checkpoints keep their own licences; see
`data/README.md`.
