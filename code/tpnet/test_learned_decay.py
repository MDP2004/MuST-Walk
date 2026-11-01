"""Unit tests for models/LDTPNet.py (learnable decay with exact streaming gradients).
Run from the TPNet repo root:  python ../tests/test_learned_decay.py
"""
import copy
import sys
import types

import numpy as np
import torch

sys.path.insert(0, '.')
from models.TPNet import RandomProjectionModule  # noqa: E402
from models.LDTPNet import LearnableDecayRandomProjectionModule, add_decay_param_group  # noqa: E402
from models.MuSTWalk import MultiScaleRandomProjectionModule  # noqa: E402

rng = np.random.default_rng(0)
N, M, K, BS = 12, 120, 3, 10
src = rng.integers(1, N + 1, M).astype(np.int64)
dst = rng.integers(1, N + 1, M).astype(np.int64)
dst = np.where(dst == src, src % N + 1, dst)
ts = np.sort(rng.uniform(0, 20, M)).astype(np.float64)
common = dict(node_num=N + 1, edge_num=M + 1, dim_factor=10, num_layer=K, device='cpu',
              beginning_time=np.float64(0.0), not_scale=False, enforce_dim=-1)
failures = 0


def check(name, cond):
    global failures
    print(('PASS ' if cond else 'FAIL ') + name)
    failures += 0 if cond else 1


def stream(module, upto=M):
    for s in range(0, upto, BS):
        module.update(src[s:min(s + BS, upto)], dst[s:min(s + BS, upto)], ts[s:min(s + BS, upto)])


def reference(lams, P):
    """Independent float64 dense recursion (oldest-edge-first per-depth rates), same batching as TPNet."""
    H = [P.copy()] + [np.zeros_like(P) for _ in range(K)]
    cum = np.cumsum(lams)
    t_prev = 0.0
    for s in range(0, M, BS):
        bs = slice(s, min(s + BS, M)); tb = ts[bs][-1]
        for i in range(1, K + 1):
            H[i] = H[i] * np.exp(-cum[i - 1] * (tb - t_prev))
        for i in range(K, 0, -1):
            w = np.exp(-lams[i - 1] * (tb - ts[bs]))[:, None]
            mu, mv = w * H[i - 1][dst[bs]], w * H[i - 1][src[bs]]
            np.add.at(H[i], src[bs], mu); np.add.at(H[i], dst[bs], mv)
        t_prev = tb
    return H


def make(mode, lam, use_matrix=False, double=False, seed=0):
    torch.manual_seed(seed)
    m = LearnableDecayRandomProjectionModule(decay_mode=mode, init_lambda=lam, use_matrix=use_matrix, **common)
    return m.double() if double else m


# 1. shared mode reproduces TPNet's own module (float32, same P)
lam0 = 0.15
ld = make('shared', lam0)
tp = RandomProjectionModule(time_decay_weight=lam0, use_matrix=False, **common)
tp.random_projections[0].data = ld.random_projections[0].data.clone()
stream(ld); stream(tp)
err = max(((ld.random_projections[i] - tp.random_projections[i]).abs().max() /
           tp.random_projections[i].abs().max()).item() for i in range(1, K + 1))
check(f'shared mode matches TPNet state (max rel err {err:.1e})', err < 1e-5)

# 2. per-depth rates: exact walk matrices (use_matrix, float64) equal the independent reference
lams = [0.05, 0.4, 1.2]
ex = make('per_depth', 1.0, use_matrix=True, double=True)
with torch.no_grad():
    ex.log_lambda.copy_(torch.log(torch.tensor(lams, dtype=torch.float64)))
stream(ex)
ref = reference(np.array(lams), np.eye(N + 1))
err = max(np.abs(ex.random_projections[i].detach().numpy() - ref[i]).max() / np.abs(ref[i]).max() for i in range(1, K + 1))
check(f'per-depth walk matrices match independent reference (max rel err {err:.1e})', err < 1e-12)

# 3. tangents == finite differences of the state
eps = 1e-6
worst = 0.0
for mi in range(K):
    plus, minus = list(lams), list(lams)
    plus[mi] += eps; minus[mi] -= eps
    rp, rm = reference(np.array(plus), np.eye(N + 1)), reference(np.array(minus), np.eye(N + 1))
    for i in range(1, K + 1):
        fd = (rp[i] - rm[i]) / (2 * eps)
        tan = ex.tangent(i, mi + 1).numpy() if mi + 1 <= i else np.zeros_like(fd)
        scale = max(np.abs(fd).max(), 1e-12)
        worst = max(worst, np.abs(tan - fd).max() / scale)
check(f'tangent states match finite differences (max rel err {worst:.1e})', worst < 1e-6)

# 4. end-to-end gradient of a loss on TPNet's pairwise features == finite difference over re-streaming
q_src, q_dst = src[:15], dst[:15]


def loss_at(log_lams, need_grad):
    torch.manual_seed(3)
    m = make('per_depth', 1.0, double=True, seed=7)
    with torch.no_grad():
        m.log_lambda.copy_(log_lams)
    stream(m)
    gw = torch.Generator().manual_seed(11)
    weights = torch.randn(len(q_src), m.pair_wise_feature_dim, generator=gw, dtype=torch.float64)
    with torch.set_grad_enabled(need_grad):
        f = m.get_pair_wise_feature(q_src, q_dst)
        loss = (f * weights).sum()
    return m, loss


phi = torch.log(torch.tensor([0.08, 0.3, 0.9], dtype=torch.float64))
m, loss = loss_at(phi, True)
loss.backward()
g_auto = m.log_lambda.grad.clone()
g_fd = torch.zeros(K, dtype=torch.float64)
for mi in range(K):
    e = torch.zeros(K, dtype=torch.float64); e[mi] = 1e-6
    g_fd[mi] = (loss_at(phi + e, False)[1] - loss_at(phi - e, False)[1]) / 2e-6
rel = ((g_auto - g_fd).abs().max() / g_fd.abs().max()).item()
check(f'end-to-end d loss / d log_lambda: autograd {g_auto.numpy().round(4)} vs finite diff {g_fd.numpy().round(4)} (rel err {rel:.1e})', rel < 1e-5)

# 5. reading the sketch with gradients does not change its value
m2, _ = loss_at(phi, False)
with torch.no_grad():
    a = m2.get_random_projections(q_src)
b = m2.get_random_projections(q_src)
check('gradient injection is value-preserving', all(torch.equal(x, y.detach()) for x, y in zip(a, b)))

# 6. TPNet updates the state AFTER the forward pass and BEFORE backward: gradients must be unaffected
m3, _ = loss_at(phi, False)
m4 = copy.deepcopy(m3)
w = torch.randn(len(q_src), m3.pair_wise_feature_dim, dtype=torch.float64)
l3 = (m3.get_pair_wise_feature(q_src, q_dst) * w).sum(); l3.backward()
l4 = (m4.get_pair_wise_feature(q_src, q_dst) * w).sum()
m4.update(src[:20], dst[:20], ts[:20] + 30.0)  # in-place state change between forward and backward
l4.backward()
check('in-place update between forward and backward leaves gradients unchanged',
      torch.allclose(m3.log_lambda.grad, m4.log_lambda.grad))

# 7. reset / backup / reload
m5 = make('per_depth', 0.2)
stream(m5)
backup = m5.backup_random_projections()
snap = {n: b.clone() for n, b in m5.named_buffers()}
layers = [m5.random_projections[i].clone() for i in range(K + 1)]
m5.update(src[:30], dst[:30], ts[:30] + 50.0)
m5.reload_random_projections(backup)
ok = all(torch.equal(b, snap[n]) for n, b in m5.named_buffers()) and \
     all(torch.equal(m5.random_projections[i], layers[i]) for i in range(K + 1))
check('backup/reload restores layers, tangents and time', ok)
m5.reset_random_projections()
check('reset zeroes layers and tangents',
      all(b.abs().sum().item() == 0 for n, b in m5.named_buffers() if n.startswith('tan_')) and
      all(m5.random_projections[i].abs().sum().item() == 0 for i in range(1, K + 1)))

# 8. multi-scale + learnable decay: shared P, separate rates per scale, zero rate handled
ms = MultiScaleRandomProjectionModule(lambdas=[0.2, 0.0], fusion='concat', base_index=0, use_matrix=False,
                                      learn_decay='per_depth', **common)
check('multi-scale learnable: P shared', ms.scales[1].random_projections[0] is ms.scales[0].random_projections[0])
check('multi-scale learnable: zero rate starts at 1e-12',
      abs(torch.exp(ms.scales[1].log_lambda).max().item() - 1e-12) < 1e-15)
stream(ms)
ms.reset_random_projections()
check('multi-scale learnable: reset zeroes every scale\'s tangents',
      all(b.abs().sum().item() == 0 for s in ms.scales for n, b in s.named_buffers() if n.startswith('tan_')))
check('decay_summary prints', 'scale 1' in ms.decay_summary())

# 9. optimizer groups and checkpoint round trip
net = torch.nn.ModuleList([ms, torch.nn.Linear(3, 3)])
args = types.SimpleNamespace(learning_rate=1e-4, rp_decay_lr=1e-2, weight_decay=0.0)
opt = add_decay_param_group(torch.optim.Adam(net.parameters(), lr=1e-4), net, args)
check('optimizer: decay params in their own group with their own lr',
      len(opt.param_groups) == 2 and opt.param_groups[1]['lr'] == 1e-2 and len(opt.param_groups[1]['params']) == 2)
plain = torch.nn.Linear(3, 3)
opt2 = torch.optim.Adam(plain.parameters(), lr=1e-4)
check('optimizer unchanged when nothing is learnable', add_decay_param_group(opt2, plain, args) is opt2)
sd = copy.deepcopy(ms.state_dict())
ms2 = MultiScaleRandomProjectionModule(lambdas=[0.2, 0.0], fusion='concat', base_index=0, use_matrix=False,
                                       learn_decay='per_depth', **common)
ms2.load_state_dict(sd, strict=True)
check('checkpoint round trip (strict) keeps P shared', ms2.scales[1].random_projections[0] is ms2.scales[0].random_projections[0])

print(f'\n{failures} failure(s)')
sys.exit(1 if failures else 0)
