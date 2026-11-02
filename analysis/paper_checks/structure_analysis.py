import os
"""Graph structure and feature-level signal per dataset, on CPU, with no neural model.

Mirrors the pipeline: chronological 70/15/15 split, strict read (a group sees only strictly older groups), historical negatives,
tuned rates and ladders from the run lists, sketch width min(floor(ln 2E) * 10, n). Walks use exact matrices, and a separate pass
pushes them through a real Gaussian projection of that width. Usage: python structure_analysis.py <dataset> <out_dir>
"""
import json
import math
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

YEAR = 86400.0 * 365
DAY = 86400.0
CFG = {
    'USLegis': dict(yearly=True, tuned=1e-6, ladder=[1e-5, 1e-6, 1e-7, 2.2e-8, 0.0]),
    'CanParl': dict(yearly=True, tuned=1e-5, ladder=[1e-5, 1e-7, 2.2e-8, 0.0]),
    'UNtrade': dict(yearly=True, tuned=1e-7, ladder=[1e-5, 1e-7, 2.2e-8, 0.0]),
    'UNvote': dict(yearly=True, tuned=1e-5, ladder=[1e-5, 1e-7, 2.2e-8, 0.0]),
    'uci': dict(yearly=False, tuned=1e-7, ladder=[1e-6, 1e-7, 1e-8, 0.0]),
    'enron': dict(yearly=False, tuned=1e-6, ladder=[1e-5, 1e-6, 1e-7, 0.0]),
}
GRID = {True: [1e-5, 1e-6, 1e-7, 2.2e-8, 5.5e-9, 0.0], False: [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 0.0]}
NEG_SEEDS = 5
P_SEEDS = 3
MAX_POS_PER_GROUP = {True: 4000, False: 200}
FEATS = ['direct', 'two_hop', 'common', 'act_u', 'act_v']

ds, out_dir = sys.argv[1], sys.argv[2]
cfg = CFG[ds]
yearly = cfg['yearly']
t0 = time.time()
df = pd.read_csv(os.path.join(os.environ.get('TPNET_DATA', 'code/tpnet/processed_data'), ds, f'ml_{ds}.csv'))
src, dst = df.u.values.astype(np.int64), df.i.values.astype(np.int64)
ts = df.ts.values.astype(np.float64)
if yearly:
    ts = (pd.Series(ts).rank(method='dense').values - 1) * YEAR
order = np.argsort(ts, kind='stable')
src, dst, ts = src[order], dst[order], ts[order]
E = len(ts)
N = int(max(src.max(), dst.max())) + 1
val_time, test_time = np.quantile(ts, [0.70, 0.85])
d_R = min(int(math.log((E + 1) * 2)) * 10, N)
res = {'dataset': ds, 'edges': E, 'd_R': d_R, 'tuned': cfg['tuned'], 'ladder': cfg['ladder'], 'val_time': val_time, 'test_time': test_time}

# ---------------------------------------------------------------- static and temporal structure
active = np.unique(np.concatenate([src, dst]))
n = len(active)
lo, hi = np.minimum(src, dst), np.maximum(src, dst)
und = lo * N + hi
ord_keys = src * N + dst
upairs = np.unique(und)
opairs = np.unique(ord_keys)
rev = set(((opairs % N) * N + opairs // N).tolist())
reciprocal = float(np.mean([k in rev for k in opairs.tolist()]))
A = sp.csr_matrix((np.ones(len(upairs)), (upairs // N, upairs % N)), shape=(N, N))
A = ((A + A.T) > 0).astype(np.float64).tolil()
A.setdiag(0)
A = A.tocsr()
A.eliminate_zeros()
deg = np.asarray(A.sum(1)).ravel()
deg_a = deg[active]
sorted_deg = np.sort(deg_a)
gini = float((2 * np.arange(1, n + 1) - n - 1).dot(sorted_deg) / (n * sorted_deg.sum()))
D = A.toarray()
A2d = D @ D
tri = float((A2d * D).sum())
wedges = float((deg_a * (deg_a - 1)).sum())
steps = np.unique(ts)
span = float(ts.max() - ts.min())
res['structure'] = dict(
    nodes=int(n), id_range=int(N), steps=int(len(steps)), span_days=span / DAY, unique_pairs=int(len(upairs)),
    density=len(upairs) / (n * (n - 1) / 2), mean_degree=float(deg_a.mean()), median_degree=float(np.median(deg_a)),
    mean_degree_frac=float(deg_a.mean() / (n - 1)), degree_gini=gini, transitivity=tri / wedges if wedges else 0.0,
    edges_per_pair=E / len(upairs), reciprocal_ordered=reciprocal, median_step_gap=float(np.median(np.diff(steps))),
)

if yearly:
    dens, pers = [], []
    prev = None
    for t in steps:
        cur = np.unique(und[ts == t])
        dens.append(len(cur) / (n * (n - 1) / 2))
        if prev is not None and len(prev):
            pers.append(len(np.intersect1d(prev, cur)) / len(prev))
        prev = cur
    res['structure'].update(snapshot_density=float(np.mean(dens)), snapshot_density_last=float(dens[-1]), step_persistence=float(np.mean(pers)),
                            step_persistence_last=float(np.mean(pers[-5:])))
else:
    bins = np.floor((ts - ts.min()) / (7 * DAY)).astype(np.int64)
    pers = []
    for b in np.unique(bins):
        a, c = np.unique(und[bins == b]), np.unique(und[bins == b + 1])
        if len(a) and len(c):
            pers.append(len(np.intersect1d(a, c)) / len(a))
    res['structure'].update(step_persistence=float(np.mean(pers)), snapshot_density=float(np.mean([len(np.unique(und[bins == b])) / (n * (n - 1) / 2) for b in np.unique(bins)])))

# node turnover: share of active nodes that stay active into the next period, and new nodes in the test period
period = np.searchsorted(steps, ts) if yearly else np.floor((ts - ts.min()) / (30 * DAY)).astype(np.int64)
stay = []
sets = [np.unique(np.concatenate([src[period == b], dst[period == b]])) for b in np.unique(period)]
for a, c in zip(sets[:-1], sets[1:]):
    stay.append(len(np.intersect1d(a, c)) / len(a))
before = np.unique(np.concatenate([src[ts <= test_time], dst[ts <= test_time]]))
tm = ts > test_time
res['nodes'] = dict(stay_next_period=float(np.mean(stay)), test_edges_with_new_node=float((~np.isin(src[tm], before) | ~np.isin(dst[tm], before)).mean()))

# recurrence gaps of a pair: consecutive distinct contact times
gaps = []
frame = pd.DataFrame({'k': und, 't': ts}).drop_duplicates().sort_values(['k', 't'])
g = frame.groupby('k')['t'].diff().dropna().values
if len(g):
    res['recurrence'] = dict(median_gap_days=float(np.median(g) / DAY), p90_gap_days=float(np.quantile(g, 0.9) / DAY), pairs_with_repeat=float(frame.groupby('k').size().gt(1).mean()))
    per = frame.assign(g=frame.groupby('k')['t'].diff()).dropna().groupby('k')['g']
    mu, sd = per.mean(), per.std()
    keep = per.size() >= 5
    if keep.any():
        res['recurrence']['burstiness'] = float(((sd[keep] - mu[keep]) / (sd[keep] + mu[keep])).mean())

# empirical memory kernel: chance that a pair is active at a step, given its age since last contact
kernel = {}
last = np.full((N, N), np.nan)
if yearly:
    idx = {t: i for i, t in enumerate(steps)}
    num, den = np.zeros(len(steps) + 1), np.zeros(len(steps) + 1)
    start = int(len(steps) * 0.3)
    for i, t in enumerate(steps):
        m = ts == t
        S = np.zeros((N, N), dtype=bool)
        S[src[m], dst[m]] = True
        S |= S.T
        if i >= start:
            seen = ~np.isnan(last)
            age = (i - last[seen]).astype(np.int64)
            den += np.bincount(age, minlength=len(den))[:len(den)]
            num += np.bincount(age, weights=S[seen].astype(np.float64), minlength=len(den))[:len(den)]
        last[S] = i
    for a in range(1, min(13, len(steps))):
        if den[a] >= 50:
            kernel[str(a)] = dict(p=float(num[a] / den[a]), n=int(den[a]))
else:
    edges_days = [0, 1, 3, 7, 14, 30, 90, 1e9]
    labels = ['<1d', '1-3d', '3-7d', '1-2w', '2-4w', '1-3m', '>3m']
    num, den = np.zeros(7), np.zeros(7)
    anchors = np.arange(ts.min() + 0.3 * span, ts.max() - 7 * DAY, 7 * DAY)
    p = 0
    for a in anchors:
        while p < E and ts[p] <= a:
            last[src[p], dst[p]] = ts[p]
            last[dst[p], src[p]] = ts[p]
            p += 1
        seen = ~np.isnan(last)
        age = (a - last[seen]) / DAY
        b = np.clip(np.searchsorted(edges_days, age, side='right') - 1, 0, 6)
        m = (ts > a) & (ts <= a + 7 * DAY)
        S = np.zeros((N, N), dtype=bool)
        S[src[m], dst[m]] = True
        S |= S.T
        den += np.bincount(b, minlength=7)
        num += np.bincount(b, weights=S[seen].astype(np.float64), minlength=7)
    for i, lab in enumerate(labels):
        if den[i] >= 50:
            kernel[lab] = dict(p=float(num[i] / den[i]), n=int(den[i]))
res['memory_kernel'] = kernel
del last
print(ds, 'structure done', round(time.time() - t0, 1), flush=True)

# ---------------------------------------------------------------- streaming exact walks and pair features
if yearly:
    groups = [np.flatnonzero(ts == t) for t in steps]
else:
    groups = [np.arange(s, min(s + 200, E)) for s in range(0, E, 200)]
unique_dst = np.unique(dst)
rates = sorted(set(GRID[yearly] + cfg['ladder'] + [cfg['tuned']]), reverse=True)
sk_rates = sorted(set(cfg['ladder'] + [cfg['tuned']]), reverse=True)
state = {r: dict(A1=np.zeros((N, N)), A2=np.zeros((N, N)), now=ts[0]) for r in rates}
Ps = [np.random.default_rng(100 + s).normal(0, 1 / math.sqrt(d_R), (N, d_R)) for s in range(P_SEEDS)]
rng = np.random.default_rng(7)
seen = np.zeros((N, N), dtype=bool)
last_und = np.full((N, N), np.nan)
last_ord = np.full((N, N), np.nan)


def update(r, idx):
    st = state[r]
    T = ts[idx].max()
    g1 = math.exp(-r * (T - st['now']))
    if g1 != 1.0:
        st['A1'] *= g1
        st['A2'] *= g1 * g1
    w = np.exp(-r * (T - ts[idx]))
    S = sp.csr_matrix((np.concatenate([w, w]), (np.concatenate([src[idx], dst[idx]]), np.concatenate([dst[idx], src[idx]]))), shape=(N, N))
    st['A2'] += S @ st['A1']
    st['A1'] += S.toarray() if N < 800 else 0
    if N >= 800:
        np.add.at(st['A1'], (src[idx], dst[idx]), w)
        np.add.at(st['A1'], (dst[idx], src[idx]), w)
    st['now'] = T


def raw(A1, A2, u, v):
    x = np.zeros((len(u), 5))
    x[:, 0] = A1[u, v]
    x[:, 1] = 0.5 * (A2[u, v] + A2[v, u])
    x[:, 2] = np.einsum('ij,ij->i', A1[u], A1[v])
    x[:, 3] = np.einsum('ij,ij->i', A1[u], A1[u])
    x[:, 4] = np.einsum('ij,ij->i', A1[v], A1[v])
    return x


def sketched(A1, A2, u, v, P):
    nodes, inv = np.unique(np.concatenate([u, v]), return_inverse=True)
    iu, iv = inv[:len(u)], inv[len(u):]
    R1, R2 = A1[nodes] @ P, A2[nodes] @ P
    x = np.zeros((len(u), 5))
    x[:, 0] = np.einsum('ij,ij->i', R1[iu], P[v])
    x[:, 1] = 0.5 * (np.einsum('ij,ij->i', R2[iu], P[v]) + np.einsum('ij,ij->i', R2[iv], P[u]))
    x[:, 2] = np.einsum('ij,ij->i', R1[iu], R1[iv])
    x[:, 3] = np.einsum('ij,ij->i', R1[iu], R1[iu])
    x[:, 4] = np.einsum('ij,ij->i', R1[iv], R1[iv])
    return x


coll = {r: [] for r in rates}
coll_sk = {(r, p): [] for r in sk_rates for p in range(P_SEEDS)}
meta = []
res_rows = []
for gi, idx in enumerate(groups):
    tmin, tmax = ts[idx].min(), ts[idx].max()
    split = 'val' if tmin > val_time and tmax <= test_time else ('test' if tmin > test_time else None)
    if split:
        pos = idx if len(idx) <= MAX_POS_PER_GROUP[yearly] else rng.choice(idx, MAX_POS_PER_GROUP[yearly], replace=False)
        cur = np.zeros((N, N), dtype=bool)
        cur[src[idx], dst[idx]] = True
        pool = np.flatnonzero((seen & ~cur).ravel())
        if len(pool) >= 10:
            us, vs, lab, ns = [src[pos]], [dst[pos]], [np.ones(len(pos))], [np.full(len(pos), -1)]
            for s in range(NEG_SEEDS):
                pick = np.random.default_rng(1000 * s + gi).choice(pool, min(len(pos), len(pool)), replace=False)
                us.append(pick // N)
                vs.append(pick % N)
                lab.append(np.zeros(len(pick)))
                ns.append(np.full(len(pick), s))
                if split == 'val':
                    rr = np.random.default_rng(5000 + 1000 * s + gi)
                    rand_dst = rr.choice(unique_dst, len(pos))
                    us.append(src[pos])
                    vs.append(rand_dst)
                    lab.append(np.zeros(len(pos)))
                    ns.append(np.full(len(pos), -2 - s))
            u, v, y, seed = np.concatenate(us), np.concatenate(vs), np.concatenate(lab), np.concatenate(ns)
            for r in rates:
                st = state[r]
                coll[r].append(raw(st['A1'], st['A2'], u, v))
            sel = (seed == -1) | (seed == 0)
            for r in sk_rates:
                st = state[r]
                for p in range(P_SEEDS):
                    coll_sk[(r, p)].append(sketched(st['A1'], st['A2'], u[sel], v[sel], Ps[p]))
            lag = np.full(len(u), tmin - state[rates[0]]['now'])
            meta.append(dict(split=np.full(len(u), 0 if split == 'val' else 1), y=y, seed=seed, u=u, v=v, lag=lag, sel=sel, group=np.full(len(u), gi),
                             age_und=tmin - last_und[u, v], age_ord=tmin - last_ord[u, v]))
    for r in rates:
        update(r, idx)
    seen[src[idx], dst[idx]] = True
    last_ord[src[idx], dst[idx]] = ts[idx]
    last_und[src[idx], dst[idx]] = ts[idx]
    last_und[dst[idx], src[idx]] = ts[idx]
print(ds, 'stream done', round(time.time() - t0, 1), 'val/test groups', len(meta), flush=True)

M = {k: np.concatenate([m[k] for m in meta]) for k in meta[0]}
X = {r: np.vstack(v) for r, v in coll.items()}
Xs = {k: np.vstack(v) for k, v in coll_sk.items()}
in_val, in_test = M['split'] == 0, M['split'] == 1


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=400))


def fx(a):
    return np.log1p(np.clip(a, 0, None))


def run(F, mask_val, mask_test, y):
    c = lr().fit(F[mask_val], y[mask_val])
    return float(average_precision_score(y[mask_test], c.predict_proba(F[mask_test])[:, 1]))


feat = {}
for s in range(NEG_SEEDS):
    keep = (M['seed'] == -1) | (M['seed'] == s)
    tv, te = keep & in_val, keep & in_test
    y = M['y']
    row = {}
    cvs = {}
    for r in rates:
        F = fx(X[r])
        row[str(r)] = run(F, tv, te, y)
        cvs[r] = float(cross_val_score(lr(), F[tv], y[tv], cv=3, scoring='average_precision').mean())
    Fl = np.hstack([fx(X[r]) for r in cfg['ladder']])
    row['ladder'] = run(Fl, tv, te, y)
    pick = max(cvs, key=cvs.get)
    row['picked_rate'] = pick
    row['picked_ap'] = row[str(pick)]
    gt = np.exp(-cfg['tuned'] * M['lag'])
    Xa = X[cfg['tuned']].copy()
    Xa[:, 0] *= gt
    Xa[:, 1] *= gt * gt
    Xa[:, 2:] *= (gt * gt)[:, None]
    row['tuned_read_at_t'] = run(fx(Xa), tv, te, y)
    rnd = ((M['seed'] == -1) | (M['seed'] == -2 - s)) & in_val
    row['tuned_random_train'] = run(fx(X[cfg['tuned']]), rnd, te, y)
    row['ladder_random_train'] = run(Fl, rnd, te, y)
    row['zero_random_train'] = run(fx(X[0.0]), rnd, te, y)
    unit = YEAR if yearly else DAY
    au, ao = M['age_und'] / unit, M['age_ord'] / unit
    rec_b = np.where(np.isnan(au), 0.0, 1 / (1 + au))
    rec_a = np.where(np.isnan(ao), 0.0, 1 / (1 + ao))
    nov_b = np.where(np.isnan(au), 2.0, 1 / (1 + au))
    nov_a = np.where(np.isnan(ao), 2.0, 1 / (1 + ao))
    for k, sc in [('rule_recency_blind', rec_b), ('rule_recency_aware', rec_a), ('rule_novel_blind', nov_b), ('rule_novel_aware', nov_a)]:
        row[k] = float(average_precision_score(y[te], sc[te]))
    # the same models with the free positives removed: every positive left could also have been drawn as a negative
    te2 = te & ~np.isnan(M['age_ord'])
    tv2 = tv & ~np.isnan(M['age_ord'])
    bal = np.flatnonzero(te2 & (y == 0))
    keep_n = np.random.default_rng(s).choice(bal, min(int((te2 & (y == 1)).sum()), len(bal)), replace=False)
    te3 = te2 & (y == 1)
    te3[keep_n] = True
    for k, F in [('tuned', fx(X[cfg['tuned']])), ('ladder', Fl), ('picked', fx(X[pick]))]:
        row['seen_' + k] = run(F, tv2, te3, y)
    row['seen_rule_aware'] = float(average_precision_score(y[te3], rec_a[te3]))
    row['seen_rule_blind'] = float(average_precision_score(y[te3], rec_b[te3]))
    feat[s] = row
res['feature_level'] = {k: dict(mean=float(np.mean([feat[s][k] for s in feat])), std=float(np.std([feat[s][k] for s in feat]))) for k in feat[0] if k != 'picked_rate'}
res['feature_level_picked_rates'] = [feat[s]['picked_rate'] for s in feat]

pt = (M['y'] == 1) & in_test
res['positives'] = dict(never_seen_either_direction=float(np.isnan(M['age_und'][pt]).mean()), never_seen_stored_direction=float(np.isnan(M['age_ord'][pt]).mean()))
ng = (M['y'] == 0) & in_test & (M['seed'] >= 0)
res['negatives'] = dict(never_seen_stored_direction=float(np.isnan(M['age_ord'][ng]).mean()), median_age_und=float(np.nanmedian(M['age_und'][ng]) / (YEAR if yearly else DAY)))
res['positives']['median_age_und'] = float(np.nanmedian(M['age_und'][pt]) / (YEAR if yearly else DAY)) if (~np.isnan(M['age_und'][pt])).any() else None

# what the tuned rate and no decay read as zero: a pair last seen long ago ties with a pair never seen
tiep = (M['y'] == 1) & in_test
tien = (M['y'] == 0) & in_test & (M['seed'] >= 0)
res['zero_reads'] = {str(r): dict(pos=float((X[r][tiep, 0] < 1e-6).mean()), neg=float((X[r][tien, 0] < 1e-6).mean())) for r in sorted(set([cfg['tuned'], 0.0]))}

# single-feature signal on the test split, positives vs seed-0 historical negatives
sel = M['sel'] & in_test
y = M['y'][sel]
single = {}
for r in sorted(set([cfg['tuned'], 0.0])):
    Ft = X[r][sel]
    single[str(r)] = {name: float(roc_auc_score(y, Ft[:, i] + 1e-12 * np.arange(len(y)) * 0)) for i, name in enumerate(FEATS[:3])}
res['single_feature_auc'] = single

# exact vs sketched, on seed-0 negatives
sel_all = M['sel']
mv, mt = in_val[sel_all], in_test[sel_all]
ya = M['y'][sel_all]
sk = {}
for name, rs in [('tuned', [cfg['tuned']]), ('ladder', cfg['ladder'])]:
    ex = run(np.hstack([fx(X[r][sel_all]) for r in rs]), mv, mt, ya)
    runs = [run(np.hstack([fx(Xs[(r, p)]) for r in rs]), mv, mt, ya) for p in range(P_SEEDS)]
    sk[name] = dict(exact=ex, sketched=float(np.mean(runs)), sketched_std=float(np.std(runs)))
res['sketch_loss'] = sk

# how well one sketch can resolve a single contact, by Proposition: separation a_v sqrt(d) / ||a||
pos_test = (M['y'] == 1) & in_test
res['resolution'] = {}
for r in sorted(set([cfg['tuned'], 0.0]), reverse=True):
    direct = X[r][pos_test, 0]
    norm = np.sqrt(X[r][pos_test, 3])
    ok = direct > 0
    sep = direct[ok] * math.sqrt(d_R) / norm[ok]
    res['resolution'][str(r)] = dict(share_nonzero=float(ok.mean()), median_sep=float(np.median(sep)) if ok.any() else 0.0, p10_sep=float(np.quantile(sep, 0.1)) if ok.any() else 0.0)
    a = state[r]['A1'][active]
    s2 = (a ** 2).sum(1)
    s1 = np.abs(a).sum(1)
    ok2 = s2 > 0
    res['resolution'][str(r)]['median_eff_contacts'] = float(np.median(s1[ok2] ** 2 / s2[ok2])) if ok2.any() else 0.0

# common-neighbour saturation just before the test period
pre = ts <= test_time
Ap = sp.csr_matrix((np.ones(pre.sum() * 2), (np.concatenate([src[pre], dst[pre]]), np.concatenate([dst[pre], src[pre]]))), shape=(N, N))
Ap = (Ap > 0).astype(np.float64)
Cn = (Ap @ Ap.T).toarray()
tst = M['sel'] & in_test
uu, vv, yy = M['u'][tst], M['v'][tst], M['y'][tst]
res['common_neighbours'] = dict(pos_any=float((Cn[uu[yy == 1], vv[yy == 1]] > 0).mean()), neg_any=float((Cn[uu[yy == 0], vv[yy == 0]] > 0).mean()),
                                pos_median=float(np.median(Cn[uu[yy == 1], vv[yy == 1]])), neg_median=float(np.median(Cn[uu[yy == 0], vv[yy == 0]])),
                                auc=float(roc_auc_score(yy, Cn[uu, vv])))
res['seconds'] = time.time() - t0
json.dump(res, open(f'{out_dir}/{ds}.json', 'w'), indent=1, default=float)
print(ds, 'done', round(res['seconds'], 1), flush=True)
