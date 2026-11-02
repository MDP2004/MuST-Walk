"""What weight does a one-step-old edge really have when the sketch is read, under the strict buffer?"""
import glob
import math
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "..", "code", "tpnet")
DATA = os.environ.get("TPNET_DATA", os.path.abspath(os.path.join(CODE, "processed_data")))
sys.path.insert(0, CODE)
os.chdir(CODE)
from models.MuSTWalk import MultiScaleRandomProjectionModule as MS

YEAR = 3.15e7


def build(lam, n, t0, strict=True, lazy=True, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return MS(lambdas=[lam], fusion="concat", base_index=0, node_num=n, edge_num=60000, dim_factor=10, num_layer=3,
              device="cpu", use_matrix=False, beginning_time=np.float64(t0), not_scale=False, enforce_dim=-1,
              strict_time=strict, lazy=lazy)


def layer(m, node, l):
    return m.scales[0].get_random_projections(np.array([node]))[l][0]


print("== 1. toy stream, query in year 2")
for lazy in [True, False]:
    for lam in [1e-5, 1e-6, 1e-7]:
        m = build(lam, 10, 0.0, lazy=lazy)
        for (a, b, y) in [(1, 2, 0.0), (3, 4, 1.0)]:
            m.update(np.array([a]), np.array([b]), np.array([y * YEAR]))
        m.advance_to(np.array([2.0 * YEAR]))
        s = m.scales[0]
        p = layer(m, 4, 0)
        read = float(layer(m, 3, 1).dot(p) / p.dot(p))
        old = float(layer(m, 2, 1).norm())
        print(f"lazy={lazy!s:5} lambda={lam:g}: one-step-old edge read at weight {read:.6f} "
              f"(report's formula e^-lambda*dt = {math.exp(-lam * YEAR):.3e}); two-step-old layer norm {old:.3e}; "
              f"sketch clock {float(s.now_time) / YEAR:.1f} y, query 2.0 y")


def find(ds):
    c = glob.glob(f"{DATA}/{ds}/ml_{ds}.csv")
    return c[0] if c else None


print("\n== 2. real data: can the sketch at the tuned rate see 'active one step ago'? (test split, positives + all pairs seen)")
for ds, lam in [("USLegis", 1e-6), ("CanParl", 1e-5), ("UNtrade", 1e-7), ("UNvote", 1e-5)]:
    f = find(ds)
    if not f:
        print(ds, "no data found")
        continue
    df = pd.read_csv(f)
    src, dst, ts = df.u.values.astype(np.int64), df.i.values.astype(np.int64), df.ts.values.astype(np.float64)
    ts = (pd.Series(ts).rank(method="dense").values - 1) * 86400 * 365  # same rule as DataLoader, convert_time=True
    n = int(max(src.max(), dst.max())) + 1
    cut = np.quantile(ts, 0.85)
    steps = np.unique(ts)
    gap = float(np.median(np.diff(steps)))
    test_start = int(np.searchsorted(ts, cut, side="right"))
    m = build(lam, n, ts[0])
    bs = 200
    for s in range(0, test_start, bs):
        e = min(s + bs, test_start)
        m.advance_to(ts[s:e])
        m.update(src[s:e], dst[s:e], ts[s:e])
    # evaluate the first test timestep only, so 'one step ago' is unambiguous
    t = ts[test_start]
    m.advance_to(np.array([t]))
    prev = steps[np.searchsorted(steps, t) - 1]
    active = set(zip(src[ts == prev].tolist(), dst[ts == prev].tolist()))
    earlier = set(zip(src[ts < prev].tolist(), dst[ts < prev].tolist())) - active
    rng = np.random.default_rng(0)
    pick = lambda s: [tuple(x) for x in np.array(sorted(s))[rng.choice(len(s), min(3000, len(s)), replace=False)].tolist()]
    a = pick(active)
    b = pick(earlier)
    if not a or not b:
        print(ds, "too few pairs", len(a), len(b))
        continue

    def feat(pairs):
        u = np.array([p[0] for p in pairs])
        v = np.array([p[1] for p in pairs])
        ru = m.scales[0].get_random_projections(u)
        rv = m.scales[0].get_random_projections(v)
        return (ru[1] * rv[0]).sum(1).numpy()

    fa, fb = feat(a), feat(b)
    x = np.concatenate([fa, fb])
    r = pd.Series(x).rank().values
    auc = (r[:len(fa)].sum() - len(fa) * (len(fa) + 1) / 2) / (len(fa) * len(fb))
    print(f"{ds:8s} step {gap / YEAR:.2f} y, lambda {lam:g}, formula weight {math.exp(-lam * gap):.2e}: "
          f"mean feature, pairs active one step ago {fa.mean():.3f} (n={len(fa)}), seen earlier only {fb.mean():.3f} (n={len(fb)}), AUC {auc:.3f}")
