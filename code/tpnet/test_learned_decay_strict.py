"""Learned decay inside the strict multi-scale module. Run from a patched repo root: python test_learned_decay_strict.py"""
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
from models.MuSTWalk import MultiScaleRandomProjectionModule as MS  # noqa: E402

fails = 0


def check(name, cond, extra=''):
    global fails
    fails += not cond
    print('PASS' if cond else 'FAIL', name, extra)


N, EDGES = 30, 240
rng = np.random.default_rng(0)
src, dst = rng.integers(1, N, EDGES), rng.integers(1, N, EDGES)
times = np.sort(rng.integers(0, 60, EDGES)).astype(np.float64) * 1e4  # many ties, as on yearly data


def make(lams, learn='none', strict=True, seed=0):
    torch.manual_seed(seed)
    return MS(lambdas=lams, fusion='concat', base_index=0, node_num=N, edge_num=EDGES, dim_factor=10, num_layer=2, device='cpu', use_matrix=False,
              beginning_time=np.float64(0.0), not_scale=False, enforce_dim=-1, strict_time=strict, learn_decay=learn).double()


def stream(m, upto=EDGES, b=16):
    for i in range(0, upto, b):
        j = min(i + b, upto)
        m.advance_to(times[i:j])
        m.update(src[i:j], dst[i:j], times[i:j])
    m.advance_to(times[upto:upto + 1] if upto < EDGES else np.array([times[-1] + 1.0]))


Q = np.arange(1, 9)
for mode in ('shared', 'per_depth'):
    a, b = make([1e-5, 1e-6, 0.0]), make([1e-5, 1e-6, 0.0], mode)
    for x, y in zip(a.scales, b.scales):
        x.random_projections[0].data.copy_(y.random_projections[0].data)
    stream(a), stream(b)
    with torch.no_grad():
        err = max((x - y).abs().max().item() for s, t in zip(a.scales, b.scales) for x, y in zip(s.get_random_projections(Q), t.get_random_projections(Q)))
    check(f'{mode}: same sketch as fixed rates at the initial values', err < 1e-3, f'max abs diff {err:.2e}')

m = make([1e-5, 1e-6], 'per_depth')
check('P stays shared', m.scales[1].random_projections[0] is m.scales[0].random_projections[0])
stream(m)
check('tangents are non-zero after a stream', float(m.scales[0].tangent(1, 1).abs().sum()) > 0)
m.reset_random_projections()
check('reset clears the tangents of every scale', all(float(s.tangent(i, k).abs().sum()) == 0 for s in m.scales for i in range(1, 3) for k in range(1, i + 1)))

m = make([1e-5, 1e-6], 'shared')
stream(m, 120)
saved = m.backup_random_projections()
want = [x.clone() for s in m.scales for x in s.get_random_projections(Q)]
wt = m.scales[0].tangent(2, 1).clone()
stream(m)
m.reload_random_projections(saved)
got = [x for s in m.scales for x in s.get_random_projections(Q)]
check('backup and reload restore sketch and tangents', all(torch.equal(x, y) for x, y in zip(want, got)) and torch.equal(wt, m.scales[0].tangent(2, 1)))


def feature(m):
    return torch.stack([x.sum() for s in m.scales for x in s.get_random_projections(Q)]).sum()


m = make([3e-6, 1e-6], 'per_depth')
stream(m)
feature(m).backward()
grad = torch.cat([s.log_lambda.grad for s in m.scales]).clone()
h = 1e-5
for k in range(4):
    vals = []
    for sign in (1, -1):
        n = make([3e-6, 1e-6], 'per_depth')
        with torch.no_grad():
            n.scales[k // 2].log_lambda[k % 2] += sign * h
        stream(n)
        vals.append(float(feature(n)))
    fd = (vals[0] - vals[1]) / (2 * h)
    check(f'gradient {k} matches finite difference under strict time', abs(fd - float(grad[k])) <= 1e-4 * max(1.0, abs(fd)), f'exact {float(grad[k]):.6g} fd {fd:.6g}')

try:
    make_lazy = MS(lambdas=[1e-5], fusion='concat', base_index=0, node_num=N, edge_num=EDGES, dim_factor=10, num_layer=2, device='cpu', use_matrix=False,
                   beginning_time=np.float64(0.0), not_scale=False, enforce_dim=-1, lazy=True, learn_decay='shared')
    check('lazy with learned rates is refused', False)
except AssertionError:
    check('lazy with learned rates is refused', True)
print('failed:', fails)
sys.exit(fails > 0)
