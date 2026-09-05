"""Learn-then-Test comparator: a selective policy family calibrated to the same budget.

GUARD screens each sample against every label its plausible set admits. The natural
guaranteed alternative tunes one scalar threshold on a heuristic score until a
risk-control procedure certifies the resulting policy. This builds that alternative on
the same frozen model, the same corrector and the same calibration third, so the two
differ only in how the decision is made.

Risk is the joint harm GUARD controls: apply and raise the loss by more than delta.
Thresholds are tested from the most conservative down, in fixed sequence, and testing
stops at the first threshold that fails, so the family-wise error is delta.
"""
import numpy as np, sys
from guard import losses as _L, targets as _T, certify as _C
from guard.pipeline import _select_beta
from gates_core import _score, _at_rate, KS, TS, SPACES, WTS, ALPHA, DELTA
from sklearn.linear_model import LogisticRegression
from scipy.stats import binom
import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get('GUARD_ARTIFACTS', _ROOT / 'artifacts'))


NLAM=100

def _h1(a,b):
    a=min(max(a,1e-12),1-1e-12); b=min(max(b,1e-12),1-1e-12)
    return a*np.log(a/b)+(1-a)*np.log((1-a)/(1-b))

def hb_pvalue(Rhat,n,alpha):
    """Hoeffding-Bentkus p-value for H0: risk > alpha (Bates et al., 2021)."""
    if Rhat>=alpha: return 1.0
    pb=float(np.e*binom.cdf(np.ceil(n*Rhat),n,alpha))
    ph=float(np.exp(-n*_h1(Rhat,alpha)))
    return float(min(1.0,min(pb,ph)))

def ltt_threshold(score_c,harm_c,alpha,delta):
    """Most permissive threshold whose risk is certified at level alpha."""
    n=len(score_c)
    lams=np.quantile(score_c,np.linspace(1.0,0.0,NLAM))
    chosen=None
    for lam in lams:
        ap=score_c>=lam
        R=float((ap&harm_c).mean())
        if hb_pvalue(R,n,alpha)>delta: break
        chosen=float(lam)
    return chosen
