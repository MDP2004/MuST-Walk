"""Independent tests for the v3 additions. Run from local/TPNet_v3:  python ../verify/test_features.py

Lazy is compared with the original eager module. Cross-scale is compared with brute-force temporal walk matrices that
share no code with the sketch. Historical negatives are checked against their definition.
"""
import math
import sys
from argparse import Namespace

import numpy as np
import torch

sys.path.insert(0, '.')
from models.MuSTWalk import (CachedNegativeSampler, HistoricalTrainNegatives, LazyRandomProjectionModule,
                            MultiScaleRandomProjectionModule as MS, build_random_projection_module)
from models.TPNet import RandomProjectionModule as RP
from utils.utils import NegativeEdgeSampler

fails = 0


def check(name, ok, detail=''):
    global fails
    fails += not ok
    print(('PASS ' if ok else 'FAIL ') + name + (f'  [{detail}]' if detail else ''))


N, K, YEAR = 40, 3, 86400 * 365
rng = np.random.default_rng(0)
E = 4000
src = rng.integers(1, N + 1, E)
dst = (src + rng.integers(0, N - 1, E)) % N + 1
cont = np.sort(rng.uniform(0, 3e7, E))
years = np.sort(rng.integers(0, 40, E)).astype(np.float64) * YEAR
common = dict(node_num=N + 1, edge_num=E, dim_factor=10, num_layer=K, device='cpu', use_matrix=False,
              beginning_time=np.float64(0.0), not_scale=False, enforce_dim=-1)


def pair(lam, seed=1):
    torch.manual_seed(seed)
    a = RP(time_decay_weight=lam, **common)
    torch.manual_seed(seed)
    b = LazyRandomProjectionModule(time_decay_weight=lam, **common)
    b.load_state_dict(a.state_dict(), strict=False)
    return a, b


def true_layers(m):
    ids = np.arange(N + 1)
    return torch.stack(m.get_random_projections(ids))


def rel_err(a, b):
    x, y = true_layers(a), true_layers(b)
    return float(((x - y).abs().amax(dim=(1, 2)) / x.abs().amax(dim=(1, 2)).clamp(min=1e-30)).max())


print('--- lazy equals eager')
for name, times in (('continuous', cont), ('yearly', years)):
    for lam in (0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4):
        for batch in (1, 200):
            if batch == 1 and lam not in (1e-6, 1e-5):
                continue
            a, b = pair(lam)
            with torch.no_grad():
                for s in range(0, E, batch):
                    a.update(src[s:s + batch], dst[s:s + batch], times[s:s + batch])
                    b.update(src[s:s + batch], dst[s:s + batch], times[s:s + batch])
            e = rel_err(a, b)
            check(f'{name} lambda={lam:g} batch={batch}', e < 2e-4, f'worst relative difference {e:.2e}')

print('--- lazy state handling')
a, b = pair(1e-6)
with torch.no_grad():
    for s in range(0, 2000, 200):
        a.update(src[s:s + 200], dst[s:s + 200], cont[s:s + 200]); b.update(src[s:s + 200], dst[s:s + 200], cont[s:s + 200])
    ba, bb = a.backup_random_projections(), b.backup_random_projections()
    for s in range(2000, 3000, 200):
        a.update(src[s:s + 200], dst[s:s + 200], cont[s:s + 200]); b.update(src[s:s + 200], dst[s:s + 200], cont[s:s + 200])
    a.reload_random_projections(ba); b.reload_random_projections(bb)
    check('backup and reload', rel_err(a, b) < 2e-4, f'{rel_err(a, b):.2e}')
    for s in range(2000, 4000, 200):
        a.update(src[s:s + 200], dst[s:s + 200], cont[s:s + 200]); b.update(src[s:s + 200], dst[s:s + 200], cont[s:s + 200])
    check('continue after reload', rel_err(a, b) < 2e-4, f'{rel_err(a, b):.2e}')
    sd = b.state_dict()
    torch.manual_seed(9)
    c = LazyRandomProjectionModule(time_decay_weight=1e-6, **common)
    c.load_state_dict(sd)
    check('state_dict round trip', torch.equal(true_layers(b), true_layers(c)))
    check('layer_scale is saved', 'layer_scale' in sd)
    x, y = a.get_pair_wise_feature(src[:50], dst[:50]), b.get_pair_wise_feature(src[:50], dst[:50])
    b.reset_random_projections()
    check('reset clears scale', float(b.layer_scale.sum()) == K + 1 and float(b.now_time) == 0.0)

print('--- multi-scale with lazy, and strict + lazy')
for strict in (False, True):
    torch.manual_seed(2)
    m = MS(lambdas=[1e-5, 1e-6, 1e-7, 0.0], fusion='concat', base_index=0, strict_time=strict, **common)
    torch.manual_seed(2)
    l = MS(lambdas=[1e-5, 1e-6, 1e-7, 0.0], fusion='concat', base_index=0, strict_time=strict, lazy=True, **common)
    l.load_state_dict(m.state_dict(), strict=False)
    with torch.no_grad():
        for s in range(0, E, 200):
            m.advance_to(years[s:s + 200]); l.advance_to(years[s:s + 200])
            m.update(src[s:s + 200], dst[s:s + 200], years[s:s + 200]); l.update(src[s:s + 200], dst[s:s + 200], years[s:s + 200])
        m.flush(); l.flush()
    worst = max(rel_err(p, q) for p, q in zip(m.scales, l.scales))
    check(f'J=4 yearly strict={strict}', worst < 2e-4, f'{worst:.2e}')
    check(f'P still shared strict={strict}', all(s.random_projections[0] is l.scales[0].random_projections[0] for s in l.scales))

print('--- lazy touches only batch rows')
a, b = pair(1e-6)
with torch.no_grad():
    b.update(src[:200], dst[:200], cont[:200])
    before = b.random_projections[2].clone()
    b.update(src[200:201], dst[200:201], cont[200:201])
    changed = int((b.random_projections[2] != before).any(dim=1).sum())
check('one edge changes at most 2 stored rows', changed <= 2, f'{changed} rows')

print('--- cross-scale inner products against brute force')


def brute(edges, lam, now, n, k):
    A = [np.zeros((n + 1, n + 1)) for _ in range(k + 1)]
    A[0] = np.eye(n + 1)
    adj = {}
    for i, (u, v, t) in enumerate(edges):
        adj.setdefault(int(u), []).append((int(v), i, t))
        adj.setdefault(int(v), []).append((int(u), i, t))

    def dfs(start, node, depth, last, w):
        for nxt, i, t in adj.get(node, []):
            if i < last:
                w2 = w * math.exp(-lam * (now - t))
                A[depth + 1][start, nxt] += w2
                if depth + 1 < k:
                    dfs(start, nxt, depth + 1, i, w2)

    for a_ in range(1, n + 1):
        dfs(a_, a_, 0, len(edges), 1.0)
    return A


n, e = 8, 20
r2 = np.random.default_rng(1)
s2 = r2.integers(1, n + 1, e)
d2 = (s2 + r2.integers(0, n - 1, e)) % n + 1
t2 = np.sort(r2.uniform(0, 1e6, e))
lams = [1e-5, 1e-6, 0.0]
m = MS(lambdas=lams, fusion='concat', base_index=0, cross=True, node_num=n + 1, edge_num=e, dim_factor=10, num_layer=K, device='cpu',
       use_matrix=False, beginning_time=np.float64(0.0), not_scale=False, enforce_dim=n + 1)
m.scales[0].random_projections[0].data = torch.eye(n + 1)
with torch.no_grad():
    for i in range(e):
        m.update(s2[i:i + 1], d2[i:i + 1], t2[i:i + 1])
A = [brute(list(zip(s2, d2, t2)), lam, t2[-1], n, K) for lam in lams]
u, v = np.array([1, 3]), np.array([2, 7])
got = m.cross_scale_inner_products(u, v).detach().numpy()
J, L = len(lams), K + 1
exp = []
for q in range(2):
    row = []
    for a_ in range(J):
        for i in range(L):
            for b_ in range(J):
                for j in range(L):
                    if a_ != b_:
                        row.append(A[a_][i][u[q]] @ A[b_][j][v[q]])
    exp.append(row)
exp = np.array(exp)
err = np.abs(got - exp).max() / np.abs(exp).max()
check('every cross entry matches brute-force A_a[u] dot A_b[v]', err < 1e-5, f'relative error {err:.2e}, {got.shape[1]} entries')
check('cross dim = J(J-1)(k+1)^2', got.shape[1] == J * (J - 1) * L ** 2)
f = m.get_pair_wise_feature(u, v)
check('feature dim adds up', f.shape[1] == m.pair_wise_feature_dim == J * (2 * K + 2) ** 2 + got.shape[1], str(f.shape))
check('scales are not interchangeable', not np.allclose(A[0][1], A[1][1]))

print('--- flags off keep the job-1 random draws')
args = Namespace(rp_dim_factor=10, rp_num_layer=K, device='cpu', rp_use_matrix=False, rp_not_scale=False, enforce_dim=-1, rp_lambdas=[1e-6],
                 rp_fusion='concat', rp_base_index=0, rp_time_decay_weight=1e-6)
torch.manual_seed(5)
a = RP(time_decay_weight=1e-6, **common)
torch.manual_seed(5)
b = build_random_projection_module(args, N + 1, E, np.float64(0.0))
check('same P and MLP weights as original TPNet', all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.scales[0].state_dict().values())))


class D:
    pass


print('--- historical training negatives')
d = D()
idx = np.arange(1, 601)
d.src_node_ids, d.dst_node_ids = src[:600], dst[:600]
d.node_interact_times = np.sort(rng.integers(0, 30, 600)).astype(np.float64) * YEAR
h = HistoricalTrainNegatives(d)
bad, used, total = 0, 0, 0
np.random.seed(0)
for s in range(100, 600, 50):
    bs, bd, bt = d.src_node_ids[s:s + 50], d.dst_node_ids[s:s + 50], d.node_interact_times[s:s + 50]
    ns, nd = h.mix(bs, bd, np.full(len(bs), -1), bt, 0.5)  # -1 marks a negative mix() left alone
    t0, t1 = bt[0], bt[-1]
    cur = set(zip(*[x[(d.node_interact_times >= t0) & (d.node_interact_times <= t1)].tolist() for x in (d.src_node_ids, d.dst_node_ids)]))
    past = set(zip(*[x[d.node_interact_times < t0].tolist() for x in (d.src_node_ids, d.dst_node_ids)]))
    for a_, b_ in zip(ns.tolist(), nd.tolist()):
        total += 1
        if b_ != -1:
            used += 1
            bad += (a_, b_) in cur or (a_, b_) not in past
check('every replaced negative was seen before and is not active now', bad == 0, f'{bad} of {used} historical')
check('about half are historical', 0.3 < used / total < 0.8, f'{used}/{total}')

print('--- cached sampler')
s3, d3, t3 = src[:500], dst[:500], np.sort(rng.uniform(0, 1e6, 500))
plain = NegativeEdgeSampler(s3, d3, interact_times=t3, last_observed_time=t3[200], negative_sample_strategy='historical', seed=0)
cached = CachedNegativeSampler(s3, d3, interact_times=t3, last_observed_time=t3[200], negative_sample_strategy='historical', seed=0)
same = True
for epoch in range(2):
    plain.reset_random_state(); cached.reset_random_state()
    for s in range(300, 500, 50):
        kw = dict(size=50, batch_src_node_ids=s3[s:s + 50], batch_dst_node_ids=d3[s:s + 50], current_batch_start_time=t3[s], current_batch_end_time=t3[s + 49])
        x, y = plain.sample(**kw), cached.sample(**kw)
        same &= bool((x[0] == y[0]).all() and (x[1] == y[1]).all())
check('same negatives as the plain sampler across epochs', same)
print(f'\n{fails} failure(s)')
sys.exit(fails > 0)
