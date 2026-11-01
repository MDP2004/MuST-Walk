"""Check the sketch against ground truth computed WITHOUT any of its code: enumerate temporal walks by depth-first search.

With P = identity, layer k of the sketch must equal the temporal walk matrix A^(k) itself. For a new edge (u, v) the update
A^(k)(u, .) += w * A^(k-1)(v, .) means a counted walk leaves u over its NEWEST edge and then moves to strictly older edges.
Weight of a walk = product over its edges of exp(-lambda * (now - t_edge)). Both directions of every edge count.
Run from the sandbox root: python check_sketch.py
"""
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
from models.MuSTWalk import MultiScaleRandomProjectionModule as MS

N, K = 8, 3
rng = np.random.default_rng(1)
E = 20
src = rng.integers(1, N + 1, E)
dst = (src + rng.integers(0, N - 1, E)) % N + 1
times = np.sort(rng.uniform(0, 1e6, E))


def brute(edges, lam, now):
    """A[k][a, b] = sum over walks a -> b of exactly k edges, with edge order index strictly decreasing along the walk."""
    A = [np.zeros((N + 1, N + 1)) for _ in range(K + 1)]
    A[0] = np.eye(N + 1)
    adj = {}
    for i, (u, v, t) in enumerate(edges):
        adj.setdefault(int(u), []).append((int(v), i, t))
        adj.setdefault(int(v), []).append((int(u), i, t))

    def dfs(start, node, depth, last, weight):
        for nxt, i, t in adj.get(node, []):
            if i < last:
                w = weight * math.exp(-lam * (now - t))
                A[depth + 1][start, nxt] += w
                if depth + 1 < K:
                    dfs(start, nxt, depth + 1, i, w)

    for a in range(1, N + 1):
        dfs(a, a, 0, len(edges), 1.0)
    return A


def sketch(lams, batch):
    m = MS(lambdas=lams, fusion='concat', base_index=0, node_num=N + 1, edge_num=E, dim_factor=10, num_layer=K, device='cpu',
           use_matrix=False, beginning_time=np.float64(0.0), not_scale=False, enforce_dim=N + 1)
    m.scales[0].random_projections[0].data = torch.eye(N + 1)
    for s in range(0, E, batch):
        m.update(src[s:s + batch], dst[s:s + batch], times[s:s + batch])
    return m


edges = list(zip(src, dst, times))
print('edges', [(int(u), int(v)) for u, v, _ in edges])
worst = 0.0
for lams in ([1e-6], [1e-5, 1e-6], [1e-5, 1e-6, 1e-7, 0.0]):
    m = sketch(lams, 1)
    now = times[-1]
    for j, lam in enumerate(lams):
        A = brute(edges, lam, now)
        for k in range(1, K + 1):
            got = m.scales[j].random_projections[k].detach().numpy().astype(np.float64)
            err = np.abs(got - A[k]).max() / max(1e-12, np.abs(A[k]).max())
            worst = max(worst, err)
            print(f'batch=1 J={len(lams)} lambda={lam:<8g} depth {k}: max |sketch-brute| / max = {err:.2e}  (max entry {np.abs(A[k]).max():.4g})')
print('worst relative error, batch=1:', f'{worst:.2e}', 'PASS' if worst < 1e-3 else 'FAIL')

print('\nBatch 5: inside one batch edges cannot chain, so depth>=2 is expected to differ a little from batch=1.')
m1, m5 = sketch([1e-6], 1), sketch([1e-6], 5)
for k in range(1, K + 1):
    a, b = m1.scales[0].random_projections[k], m5.scales[0].random_projections[k]
    print(f'depth {k}: relative difference batch 5 vs batch 1 = {float((a - b).abs().max() / a.abs().max()):.3f}')

print('\nPairwise feature: expected entries are inner products of [src layers, dst layers], then clamp at 0 and log(1+x).')
m = sketch([1e-6], 1)
s = m.scales[0]
u, v = np.array([1]), np.array([2])
R = torch.cat([torch.stack([s.random_projections[i][u] for i in range(K + 1)], 1), torch.stack([s.random_projections[i][v] for i in range(K + 1)], 1)], 1)
exp = (R @ R.transpose(1, 2)).reshape(1, -1).clamp(min=0)
exp = torch.log(exp + 1)
seen = {}
h = s.mlp.register_forward_hook(lambda mod, inp, oup: seen.update(x=inp[0].detach()))
m.get_pair_wise_feature(u, v)
h.remove()
print('feature length', seen['x'].shape[1], '= (2k+2)^2 =', (2 * K + 2) ** 2, '| max difference from hand formula', float((seen['x'] - exp).abs().max()))

print('\nJ=1 through the multi-scale wrapper vs original module, same seed:')
from models.TPNet import RandomProjectionModule as RP
common = dict(node_num=N + 1, edge_num=E, dim_factor=10, num_layer=K, device='cpu', use_matrix=False, beginning_time=np.float64(0.0), not_scale=False, enforce_dim=-1)
torch.manual_seed(3); a = RP(time_decay_weight=1e-6, **common)
torch.manual_seed(3); b = MS(lambdas=[1e-6], fusion='concat', base_index=0, **common)
for s_ in range(0, E, 4):
    a.update(src[s_:s_ + 4], dst[s_:s_ + 4], times[s_:s_ + 4]); b.update(src[s_:s_ + 4], dst[s_:s_ + 4], times[s_:s_ + 4])
print('identical sketch:', all(torch.equal(x, y) for x, y in zip(a.random_projections, b.scales[0].random_projections)))
