"""Controlled test of WHY several decay rates can help: python make_synthetic.py <repo_root>   (written as dataset 'uci')

Each source node has 12 candidate partners, all with the same long-run number of events. In any phase half of a node's partners are
"active" and half "lapsed", and the sets swap every PERIOD. So a pair's all-time count says nothing about its current state; only the
balance between recent and long-run activity does. Background noise edges keep it from being trivial.
Ground truth per edge is known, so the task can be solved only with recency relative to history, which is the claim under test.
"""
import os
import sys

import numpy as np
import pandas as pd

root = sys.argv[1]
rng = np.random.default_rng(11)
N_SRC, N_DST, K, E, PERIOD, GAP = 60, 240, 12, 30000, 4000, 60.0
partners = [rng.choice(np.arange(N_SRC + 1, N_SRC + N_DST + 1), K, replace=False) for _ in range(N_SRC)]
rows, t = [], 0.0
for e in range(E):
    t += rng.exponential(GAP)
    u = int(rng.integers(N_SRC))
    phase = int(e / PERIOD) % 2
    if rng.random() < 0.9:
        v = int(partners[u][int(rng.integers(K // 2)) * 2 + phase])  # even slots in one phase, odd slots in the other
    else:
        v = int(rng.integers(N_SRC + 1, N_SRC + N_DST + 1))
    rows.append((u + 1, v, t))
df = pd.DataFrame(rows, columns=['u', 'i', 'ts'])
used = np.unique(np.r_[df.u.values, df.i.values])
remap = {o: n + 1 for n, o in enumerate(used.tolist())}
df['u'] = df.u.map(remap)
df['i'] = df.i.map(remap)
df['label'] = 0.0
df['idx'] = np.arange(1, len(df) + 1)
out = os.path.join(root, 'processed_data', 'uci')
os.makedirs(out, exist_ok=True)
df.to_csv(os.path.join(out, 'ml_uci.csv'))
np.save(os.path.join(out, 'ml_uci.npy'), np.zeros((len(df) + 1, 172)))
np.save(os.path.join(out, 'ml_uci_node.npy'), np.zeros((len(used) + 1, 172)))
span = df.ts.max() - df.ts.min()
print(f'{len(df)} edges, {len(used)} nodes, span {span:.3g}, phase length {PERIOD * GAP:.3g} time units')
print(f'half-life of one phase corresponds to lambda about {np.log(2) / (PERIOD * GAP):.2e}')
