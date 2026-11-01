"""The padded-negative fill used in apply_tgb_features_patch.py: padded slots must not change a rank, whatever the score scale. python test_pad_mask.py"""
import numpy as np
import torch


def mrr(pos, neg):
    opt = (neg > pos[:, None]).sum(1).float()
    pes = (neg >= pos[:, None]).sum(1).float()
    return (1.0 / (0.5 * (opt + pes) + 1)).numpy()


rng = np.random.default_rng(0)
for scale in (1.0, 50.0):
    pos = torch.tensor(rng.normal(0, scale, 8), dtype=torch.float32)
    real = [rng.normal(0, scale, int(k)) for k in rng.integers(3, 7, 8)]
    longest = max(len(x) for x in real)
    ids = np.array([[1] * len(x) + [0] * (longest - len(x)) for x in real]).reshape(-1)  # 0 = padded after the +1 shift
    neg = torch.tensor(np.array([list(x) + [1e6] * (longest - len(x)) for x in real]).reshape(-1), dtype=torch.float32)  # padded scores set high on purpose
    masked = neg.masked_fill(torch.from_numpy(ids == 0), pos.min().item() - 1.0).reshape(-1, longest)
    want = np.array([1.0 / (0.5 * ((x > float(p)).sum() + (x >= float(p)).sum()) + 1) for x, p in zip(real, pos)])
    print('PASS' if np.allclose(mrr(pos, masked), want) else 'FAIL', 'scale', scale, 'mrr', round(float(want.mean()), 4))
