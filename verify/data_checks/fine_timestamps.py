"""Exploratory, no model trained: pair-level checks for fine-timestamp datasets, using last-seen time.
Same test split and sampler settings as evaluate_link_prediction.py (historical, seed 2, batch 200).
Run from local/TPNet_ms: python <this file> uci enron
"""
import bisect
import logging
import sys

import numpy as np
from sklearn.metrics import average_precision_score as aps

sys.path.insert(0, '.')
from utils.DataLoader import get_link_prediction_data
from utils.utils import NegativeEdgeSampler

log = logging.getLogger('x')
log.addHandler(logging.NullHandler())
B = 200

for ds in sys.argv[1:]:
    node_f, edge_f, full, train, val, test, nnv, nnt = get_link_prediction_data(ds, 0.15, 0.15, log, convert_time=True)
    src, dst, ts = full.src_node_ids.tolist(), full.dst_node_ids.tolist(), full.node_interact_times.tolist()
    directed, both = {}, {}
    for u, v, t in zip(src, dst, ts):
        directed.setdefault((u, v), []).append(t)
        both.setdefault((min(u, v), max(u, v)), []).append(t)
    for table in (directed, both):
        for k in table:
            table[k].sort()

    def last_seen(table, pair, t0):
        times = table.get(pair)
        if not times:
            return None
        i = bisect.bisect_left(times, t0)
        return times[i - 1] if i else None

    def any_between(table, pair, a, b):
        times = table.get(pair)
        if not times:
            return False
        i = bisect.bisect_left(times, a)
        return i < len(times) and times[i] <= b

    hist = NegativeEdgeSampler(full.src_node_ids, full.dst_node_ids, interact_times=full.node_interact_times,
                               last_observed_time=val.node_interact_times[-1], negative_sample_strategy='historical', seed=2)
    hist.reset_random_state()
    p = pos_seen = pos_seen_dir = h = h_same = h_rev = 0
    ap_blind, ap_dir, ap_seen = [], [], []
    n = len(test.src_node_ids)
    t_min = min(ts)
    for k in range(0, n, B):
        a, b, tt = test.src_node_ids[k:k + B], test.dst_node_ids[k:k + B], test.node_interact_times[k:k + B]
        t0, t1 = float(tt[0]), float(tt[-1])
        hs, hd = hist.sample(size=len(a), batch_src_node_ids=a, batch_dst_node_ids=b, current_batch_start_time=tt[0], current_batch_end_time=tt[-1])
        pos = list(zip(a.tolist(), b.tolist()))
        neg = list(zip(hs.tolist(), hd.tolist()))
        for q, ti in zip(pos, tt.tolist()):
            p += 1
            pos_seen += last_seen(both, (min(q), max(q)), ti) is not None
            pos_seen_dir += last_seen(directed, q, ti) is not None
        for q in neg:
            h += 1
            h_same += any_between(directed, q, t0, t1)
            h_rev += any_between(directed, (q[1], q[0]), t0, t1)

        def score(table, q, undirected):
            s = last_seen(table, (min(q), max(q)) if undirected else q, t0)
            return (s - t_min + 1.0) if s is not None else 0.0

        y = [1] * len(pos) + [0] * len(neg)
        pairs = pos + neg
        ap_blind.append(aps(y, [score(both, q, True) for q in pairs]))
        ap_dir.append(aps(y, [score(directed, q, False) for q in pairs]))
        ap_seen.append(aps(y, [float(score(both, q, True) > 0) for q in pairs]))
    days = (max(ts) - min(ts)) / 86400
    print(f'\n===== {ds}: {full.num_interactions:,} edges, {full.num_unique_nodes} nodes, {len(set(ts)):,} distinct times, span {days:.1f} days, test edges {n:,}')
    print(f'real test pairs: seen before (either direction) {pos_seen / p:.1%}, never seen {1 - pos_seen / p:.1%}, seen in same direction {pos_seen_dir / p:.1%}')
    print(f'hard fake pairs: same direction active in the batch interval {h_same / h:.1%}, reversed direction active {h_rev / h:.1%}')
    print(f'hard test AP, rules: seen before {100 * np.mean(ap_seen):.2f}, last-seen direction-blind {100 * np.mean(ap_blind):.2f}, last-seen direction-aware {100 * np.mean(ap_dir):.2f}')
    sys.stdout.flush()
