"""Exploratory, no model trained: does stored edge direction drive the hard test on the yearly datasets?
Same test split and the same sampler settings as evaluate_link_prediction.py (historical, seed 2, batch 200).
Run from local/TPNet_ms: python <this file> USLegis CanParl UNtrade UNvote
"""
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
    times = np.unique(full.node_interact_times)
    index = {t: i for i, t in enumerate(times.tolist())}
    directed, both = {}, {}
    for u, v, t in zip(full.src_node_ids.tolist(), full.dst_node_ids.tolist(), full.node_interact_times.tolist()):
        directed.setdefault((u, v), set()).add(index[t])
        both.setdefault((min(u, v), max(u, v)), set()).add(index[t])

    # how often are both directions of a pair stored at the same time step?
    pairs_at = two_way = 0
    for (a, b), steps in both.items():
        pairs_at += len(steps)
        two_way += len(directed.get((a, b), set()) & directed.get((b, a), set()))
    # share of neighbouring-step repeats that keep the same stored direction
    keep = flip = 0
    for (a, b), steps in both.items():
        ab, ba = directed.get((a, b), set()), directed.get((b, a), set())
        for s in steps:
            if s + 1 in steps:
                for x, y in ((ab, ab), (ba, ba)):
                    keep += s in x and s + 1 in y
                flip += (s in ab and s + 1 in ba) + (s in ba and s + 1 in ab)

    def rec(table, p, step):
        return sum(0.5 ** (step - 1 - i) for i in table.get(p, ()) if i < step)

    hist = NegativeEdgeSampler(full.src_node_ids, full.dst_node_ids, interact_times=full.node_interact_times,
                               last_observed_time=val.node_interact_times[-1], negative_sample_strategy='historical', seed=2)
    hist.reset_random_state()
    ap_dir, ap_und, ap_prev = [], [], []
    h = h_rev_now = h_same_prev = h_same_now = p_same_prev = p = 0
    n = len(test.src_node_ids)
    for k in range(0, n, B):
        src, dst, tt = test.src_node_ids[k:k + B], test.dst_node_ids[k:k + B], test.node_interact_times[k:k + B]
        steps = [index[x] for x in tt.tolist()]
        hs, hd = hist.sample(size=len(src), batch_src_node_ids=src, batch_dst_node_ids=dst,
                             current_batch_start_time=tt[0], current_batch_end_time=tt[-1])
        pos = list(zip(src.tolist(), dst.tolist()))
        neg = list(zip(hs.tolist(), hd.tolist()))
        for (a, b), st in zip(neg, steps):
            h += 1
            h_same_now += st in directed.get((a, b), ())
            h_rev_now += st in directed.get((b, a), ())
            h_same_prev += (st - 1) in directed.get((a, b), ())
        for q, st in zip(pos, steps):
            p += 1
            p_same_prev += (st - 1) in directed.get(q, ())
        y = [1] * len(pos) + [0] * len(neg)
        ap_dir.append(aps(y, [rec(directed, q, st) for q, st in zip(pos, steps)] + [rec(directed, q, st) for q, st in zip(neg, steps)]))
        und = lambda q: (min(q), max(q))
        ap_und.append(aps(y, [rec(both, und(q), st) for q, st in zip(pos, steps)] + [rec(both, und(q), st) for q, st in zip(neg, steps)]))
        ap_prev.append(aps(y, [float((st - 1) in directed.get(q, ())) for q, st in zip(pos, steps)] + [float((st - 1) in directed.get(q, ())) for q, st in zip(neg, steps)]))

    print(f'\n===== {ds}')
    print(f'pair-steps with both directions stored: {two_way / pairs_at:.1%}; next-step repeats keeping stored direction: {keep / max(keep + flip, 1):.1%}')
    print(f'real pairs: same direction active one step ago {p_same_prev / p:.1%}')
    print(f'hard fake pairs: same direction active now {h_same_now / h:.1%}, reversed direction active now {h_rev_now / h:.1%}, same direction active one step ago {h_same_prev / h:.1%}')
    print(f'hard test AP, simple rules: direction-blind recency {100 * np.mean(ap_und):.2f}, direction-aware recency {100 * np.mean(ap_dir):.2f}, direction-aware one step ago {100 * np.mean(ap_prev):.2f}')
    sys.stdout.flush()
print('\nfinished')
