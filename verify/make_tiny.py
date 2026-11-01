"""Write two tiny, hand-checkable temporal graphs into <sandbox>/processed_data.

USLegis (fake, yearly): 12 nodes, 16 years, 6 edges per year, tied timestamps, inside a year sorted by source id.
uci (fake, continuous): 12 nodes, 96 edges, all timestamps distinct.
Pairs recur on purpose, so historical negatives exist. Edge feature = edge id, so a wrong row is visible in the trace.
Node ids are 1..12 and edge ids start at 1, as DyGLib's preprocessing writes them.
"""
import os
import sys

import numpy as np
import pandas as pd

root = sys.argv[1]
rng = np.random.default_rng(7)
N = 12
PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (1, 3), (2, 4), (5, 7), (6, 8), (9, 11), (10, 12), (1, 5), (2, 6)]


def write(name, rows):
    d = os.path.join(root, 'processed_data', name)
    os.makedirs(d, exist_ok=True)
    df = pd.DataFrame(rows, columns=['u', 'i', 'ts'])
    df['label'] = 0.0
    df['idx'] = np.arange(1, len(df) + 1)
    df.to_csv(os.path.join(d, f'ml_{name}.csv'))
    edge = np.zeros((len(df) + 1, 1))
    edge[1:, 0] = df['idx'].values
    np.save(os.path.join(d, f'ml_{name}.npy'), edge)
    np.save(os.path.join(d, f'ml_{name}_node.npy'), np.zeros((N + 1, 172)))
    print(name, len(df), 'edges', df.ts.nunique(), 'distinct times')


yearly = []
for year in range(16):
    idx = rng.choice(len(PAIRS), 6, replace=False)
    edges = sorted(PAIRS[k] for k in idx)
    yearly += [(u, v, float(year)) for u, v in edges]
write('USLegis', yearly)

times = np.cumsum(rng.integers(50, 5000, 96)).astype(float)
cont = [(*PAIRS[i % len(PAIRS)], t) for i, t in enumerate(times)]  # every node appears in every period
write('uci', cont)
