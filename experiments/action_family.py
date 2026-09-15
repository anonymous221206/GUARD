"""Which learner, and which features, for the action score of Certify.

Usage: python experiments/action_family.py [benchmark ...]

Loaders below are copied unchanged from probe_vs_knn.py, which takes them from
the shared gate drivers, so the logistic column here reproduces the GUARD column
of Table 3 and acts as this script's internal control.
"""
import sys, os, json, glob, collections
import numpy as np
from pathlib import Path
from action_core import rows, NAMES

_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get('GUARD_ARTIFACTS', _ROOT / 'artifacts'))
A = str(ARTIFACTS)
os.makedirs(_ROOT / 'results/gates', exist_ok=True)

args = [a for a in sys.argv[1:] if not a.startswith('--')]
KINDS = (tuple(os.environ['ACTION_VARIANTS'].split(','))
         if os.environ.get('ACTION_VARIANTS') else NAMES)
WANT = set(args) if args else {'AVE', 'NinaPro', 'PTB-XL', 'IEMOCAP', 'DrugBAN'}
SUF = os.environ.get('ACTION_SUFFIX', '')
OUT = str(_ROOT / f'results/gates/action_family{SUF}.json')
res = {}


def report(name, acc):
    out = {}
    print(f'--- {name} ---', flush=True)
    for k in KINDS:
        r = {f: float(np.nanmean([x[f] for x in acc[k]]))
             for f in ('gain', 'harm', 'blanket_gain', 'blanket_harm', 'apply', 'base', 'auroc')}
        out[k] = r
        print(f"  {k:18}gain {r['gain']:+.4f}  harm {r['harm']:.3f}  "
              f"apply {r['apply']:.3f}  auroc {r['auroc']:.3f}", flush=True)
    out['cells'] = len(acc[KINDS[0]])
    print(f"  cells {out['cells']}", flush=True)
    res[name] = out
    json.dump(res, open(OUT, 'w'), indent=1)


def collect(acc, r):
    for k in KINDS:
        acc[k].append(r[k])


if 'AVE' in WANT:
    acc = collections.defaultdict(list)
    for cond in ('audio_only', 'visual_only'):
        d = np.load(f'{A}/ave_av_att/dumps/AV_att_{cond}.npz', allow_pickle=True)
        r_ = np.load(f'{A}/ave_av_att/dumps/AV_att_{cond}_retrieval_paper.npz', allow_pickle=True)
        P = d['probs'].reshape(-1, d['probs'].shape[-1]).astype(np.float64)
        Y = d['labels'].reshape(-1, d['labels'].shape[-1]).argmax(1)
        F = r_['deploy_features'].astype(np.float64)
        pP, pF, pY = (r_['pool_probs'].astype(np.float64),
                      r_['pool_features'].astype(np.float64), r_['pool_labels'])
        n = len(pY)
        PP = np.concatenate([pP, P]); FF = np.concatenate([pF, F]); YY = np.concatenate([pY, Y])
        for seed in range(3):
            perm = np.random.default_rng(seed).permutation(len(Y)) + n
            k = len(perm) // 3
            collect(acc, rows(PP, FF, YY, (np.arange(n), perm[:k], perm[k:2 * k], perm[2 * k:]),
                              variants=KINDS))
    report('AVE', acc)

if 'NinaPro' in WANT:
    acc = collections.defaultdict(list)
    for sd in sorted(glob.glob(f'{A}/ninapro_cnn/seed*')):
        for sub in sorted(glob.glob(f'{sd}/subject*'))[:10]:
            p = np.load(f'{sub}/preds.npz', allow_pickle=True)
            e = np.load(f'{sub}/masked_embeddings.npz', allow_pickle=True)
            for c in ('12', '8', '6', '4'):
                P = np.concatenate([p[f'train_{c}'], p[f'sess1_{c}']]).astype(np.float64)
                P = np.clip(P, 1e-12, None); P /= P.sum(1, keepdims=True)
                F = np.concatenate([e[f'train_{c}'], e[f'sess1_{c}']]).astype(np.float64)
                Y = np.concatenate([p['train_y'], p['sess1_y']]).astype(int)
                n, m = len(p['train_y']), len(p['sess1_y'])
                perm = np.random.default_rng(0).permutation(m) + n; k = m // 3
                collect(acc, rows(P, F, Y, (np.arange(n), perm[:k], perm[k:2 * k], perm[2 * k:]),
                                  variants=KINDS))
        print('  ninapro', sd, flush=True)
    report('NinaPro', acc)

if 'PTB-XL' in WANT:
    # the six conditions of gates_ptbxl, which is the driver Table 3 reports,
    # not the three of gates_rest: the feature key differs from the condition
    acc = collections.defaultdict(list)
    B = f'{A}/ptbxl_resnet1d_wang'
    p = np.load(B + '/preds.npz', allow_pickle=True)
    r_ = np.load(B + '/raw_features.npz', allow_pickle=True)
    for c, fk in (('a', 'a'), ('t', 't'), ('v', 'v'), ('at', 'a'), ('av', 'a'), ('tv', 't')):
        P = np.concatenate([p[f'train_{c}'], p[f'sess1_{c}']]).astype(np.float64)
        F = np.concatenate([r_[f'train_{fk}'], r_[f'sess1_{fk}']]).astype(np.float64)
        Y = np.concatenate([r_['train_y'], r_['sess1_y']]).astype(np.float64)
        n, m = len(r_['train_y']), len(r_['sess1_y'])
        for seed in range(3):
            perm = np.random.default_rng(seed).permutation(m) + n; k = m // 3
            collect(acc, rows(P, F, Y, (np.arange(n), perm[:k], perm[k:2 * k], perm[2 * k:]),
                              loss_name='bernoulli', variants=KINDS))
        print('  ptbxl', c, flush=True)
    report('PTB-XL', acc)

if 'IEMOCAP' in WANT:
    D = f'{A}/iemocap_momke/folds'
    def key(d):
        cnt, out = {}, []
        for v in d['vid']:
            cnt[v] = cnt.get(v, 0) + 1; out.append((v, cnt[v] - 1))
        return out
    acc = collections.defaultdict(list)
    for f_ in range(5):
        ref = np.load(f'{D}/fold{f_}_atv.npz', allow_pickle=True)
        kr = {k: i for i, k in enumerate(key(ref))}; n = len(ref['labels'])
        rich = np.zeros((n, ref['probs'].shape[1])); rich[[kr[k] for k in key(ref)]] = ref['probs']
        for m in ['a', 't', 'v', 'at', 'av', 'tv', 'atv']:
            d = np.load(f'{D}/fold{f_}_{m}.npz', allow_pickle=True)
            idx = np.array([kr[k] for k in key(d)])
            pr = np.zeros((n, d['probs'].shape[1])); pr[idx] = d['probs']
            ft = np.zeros((n, d['feats'].shape[1])); ft[idx] = d['feats']
            pf, pl = d['pool_feats'].astype(np.float64), d['pool_labels']
            P = np.concatenate([d['pool_probs'].astype(np.float64), pr])
            F = np.concatenate([pf, ft]); Y = np.concatenate([pl, ref['labels']])
            Rh = np.concatenate([ref['pool_probs'].astype(np.float64), rich])
            npool = len(pl)
            for cut in range(5):
                uv = np.unique(ref['vid']); perm = np.random.default_rng(cut).permutation(len(uv))
                parts = [np.where(np.isin(ref['vid'], g))[0] + npool
                         for g in np.array_split(uv[perm], 3)]
                collect(acc, rows(P, F, Y, (np.arange(npool), parts[0], parts[1], parts[2]),
                                  targets=('hard', 'cross'), richer=Rh, variants=KINDS))
        print('  iemocap fold', f_, flush=True)
    report('IEMOCAP', acc)

if 'DrugBAN' in WANT:
    ROOT = str(_ROOT / 'data/processed')
    CELLS = ([('drugban_biosnap_random_s42', c) for c in ('prot25', 'prot50', 'scaffold')] +
             [('drugban_bindingdb_random_s42', c) for c in ('prot25', 'prot50', 'scaffold')] +
             [('drugban_human_random_s42', c) for c in ('prot25', 'prot50')] +
             [('drugban_biosnap_cluster_s42', 'prot50')])
    acc = collections.defaultdict(list)
    for d, cond in CELLS:
        p = f'{ROOT}/{d}/{cond}.npz'
        if not os.path.exists(p):
            print('skip', p, flush=True); continue
        z = np.load(p, allow_pickle=True)
        P = np.concatenate([z['pool_probs'], z['calib_probs'], z['test_probs']]).astype(np.float64)
        F = np.concatenate([z['pool_feats'], z['calib_feats'], z['test_feats']]).astype(np.float64)
        Y = np.concatenate([z['pool_labels'], z['calib_labels'], z['test_labels']])
        npool, ncal = len(z['pool_labels']), len(z['calib_labels'])
        rp = f'{ROOT}/{d}/full.npz'
        R = None
        if os.path.exists(rp):
            zr = np.load(rp, allow_pickle=True)
            R = np.concatenate([zr['pool_probs'], zr['calib_probs'],
                                zr['test_probs']]).astype(np.float64)
        for seed in range(3):
            perm = np.random.default_rng(seed).permutation(ncal) + npool
            split = (np.arange(npool), perm[:ncal // 2], perm[ncal // 2:],
                     np.arange(len(z['test_labels'])) + npool + ncal)
            collect(acc, rows(P, F, Y, split,
                              targets=('hard', 'cross') if R is not None else ('hard',),
                              richer=R, variants=KINDS))
        print('  drugban', d, cond, flush=True)
    report('DrugBAN', acc)

print(json.dumps(res, indent=1))
