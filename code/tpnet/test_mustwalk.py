"""Unit tests for models/MuSTWalk.py. Run from the TPNet repo root:  python ../tests/test_mustwalk.py"""
import copy
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
from models.TPNet import RandomProjectionModule  # noqa: E402
from models.MuSTWalk import MultiScaleRandomProjectionModule  # noqa: E402

torch.manual_seed(0)
rng = np.random.default_rng(0)
N, M, K = 30, 400, 3
LAMS = [1e-1, 1e-2, 0.0]
common = dict(node_num=N + 1, edge_num=M + 1, dim_factor=10, num_layer=K, device='cpu', use_matrix=False,
              beginning_time=np.float64(0.0), not_scale=False, enforce_dim=-1)
src = rng.integers(1, N + 1, M).astype(np.int64)
dst = rng.integers(1, N + 1, M).astype(np.int64)
dst = np.where(dst == src, src % N + 1, dst)
ts = np.sort(rng.uniform(0, 50, M)).astype(np.float64)


def stream(module, bs=20):
    for s in range(0, M, bs):
        module.update(src[s:s + bs], dst[s:s + bs], ts[s:s + bs])


failures = 0


def check(name, cond):
    global failures
    print(('PASS ' if cond else 'FAIL ') + name)
    failures += 0 if cond else 1


# 1. each scale equals an independent single-scale TPNet module that uses the same P
ms = MultiScaleRandomProjectionModule(lambdas=LAMS, fusion='concat', base_index=0, **common)
stream(ms)
for j, lam in enumerate(LAMS):
    single = RandomProjectionModule(time_decay_weight=lam, **common)
    single.random_projections[0].data = ms.scales[0].random_projections[0].data.clone()
    stream(single)
    err = max((single.random_projections[i] - ms.scales[j].random_projections[i]).abs().max().item()
              for i in range(K + 1))
    check(f'scale {j} (lambda={lam}) matches single-scale TPNet, max abs err {err:.2e}', err == 0.0)

# 2. P shared, also after reset
check('P shared at init', all(s.random_projections[0] is ms.scales[0].random_projections[0] for s in ms.scales))
p_before = ms.scales[0].random_projections[0].clone()
ms.reset_random_projections()
check('P shared after reset', all(s.random_projections[0] is ms.scales[0].random_projections[0] for s in ms.scales))
check('P re-drawn on reset (like TPNet)', not torch.equal(p_before, ms.scales[0].random_projections[0]))
check('higher layers zeroed after reset',
      all(s.random_projections[i].abs().sum().item() == 0 for s in ms.scales for i in range(1, K + 1)))

# 3. backup / reload restores the exact state of every scale
stream(ms)
backup = ms.backup_random_projections()
snapshot = [[s.random_projections[i].clone() for i in range(K + 1)] for s in ms.scales]
times = [s.now_time.clone() for s in ms.scales]
ms.update(src[:50], dst[:50], ts[:50] + 100.0)
ms.reload_random_projections(backup)
ok = all(torch.equal(ms.scales[j].random_projections[i], snapshot[j][i]) for j in range(len(LAMS)) for i in range(K + 1))
ok = ok and all(torch.equal(ms.scales[j].now_time, times[j]) for j in range(len(LAMS)))
check('backup/reload restores all scales and times', ok)

# 4. feature dimensions and residual fusion start exactly at the base scale
q_src, q_dst = src[:16], dst[:16]
f_concat = ms.get_pair_wise_feature(q_src, q_dst)
check(f'concat dim {f_concat.shape[1]} == J*D', f_concat.shape[1] == len(LAMS) * ms.per_scale_dim)
for fusion in ['avg', 'gate']:
    torch.manual_seed(1)
    m2 = MultiScaleRandomProjectionModule(lambdas=LAMS, fusion=fusion, base_index=1, **common)
    stream(m2)
    out = m2.get_pair_wise_feature(q_src, q_dst)
    base = m2.scales[1].get_pair_wise_feature(q_src, q_dst)
    check(f'{fusion}: beta=0 -> output == base-scale feature', torch.allclose(out, base))
    check(f'{fusion}: dim == D', out.shape[1] == m2.per_scale_dim)
    # gradient reaches beta (and the gate) once beta moves
    loss = out.sum()
    loss.backward()
    check(f'{fusion}: beta receives gradient', m2.beta.grad is not None and m2.beta.grad.abs().item() > 0)
    if fusion == 'gate':
        with torch.no_grad():
            m2.beta.fill_(0.5)
        m2.zero_grad()
        m2.get_pair_wise_feature(q_src, q_dst).sum().backward()
        g = sum(p.grad.abs().sum().item() for p in m2.gate.parameters() if p.grad is not None)
        check('gate: gate MLP receives gradient when beta != 0', g > 0)
        a = m2.last_alpha
        check('gate: alpha rows sum to 1', torch.allclose(a.sum(1), torch.ones(a.shape[0])))

# 5. state_dict round trip (checkpoint save/load used by the scripts)
sd = copy.deepcopy(ms.state_dict())
ms3 = MultiScaleRandomProjectionModule(lambdas=LAMS, fusion='concat', base_index=0, **common)
missing = ms3.load_state_dict(sd, strict=True)
check('state_dict loads strictly', True)
check('P shared after load_state_dict', all(s.random_projections[0] is ms3.scales[0].random_projections[0] for s in ms3.scales))

print(f'\n{failures} failure(s)')
sys.exit(1 if failures else 0)
