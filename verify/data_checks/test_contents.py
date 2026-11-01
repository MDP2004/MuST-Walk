"""Exploratory, no model trained: what do the random and hard tests actually contain on each yearly dataset?
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


def key(u, v):
    return (u, v) if u < v else (v, u)


for ds in sys.argv[1:]:
    node_f, edge_f, full, train, val, test, nnv, nnt = get_link_prediction_data(ds, 0.15, 0.15, log, convert_time=True)
    times = np.unique(full.node_interact_times)
    index = {t: i for i, t in enumerate(times.tolist())}
    active = {}
    for u, v, t in zip(full.src_node_ids.tolist(), full.dst_node_ids.tolist(), full.node_interact_times.tolist()):
        active.setdefault(key(u, v), set()).add(index[t])

    def recency(pair, step):
        return sum(0.5 ** (step - 1 - i) for i in active.get(pair, ()) if i < step)

    rnd = NegativeEdgeSampler(full.src_node_ids, full.dst_node_ids, seed=2)
    rnd.reset_random_state()
    hist = NegativeEdgeSampler(full.src_node_ids, full.dst_node_ids, interact_times=full.node_interact_times,
                               last_observed_time=val.node_interact_times[-1], negative_sample_strategy='historical', seed=3)
    hist.reset_random_state()

    s = dict(pos_prev=0, pos_seen=0, pos=0, rnd_now=0, rnd_seen=0, rnd=0, h_prev=0, h_now=0, h_seen=0, h=0)
    ap_oracle, ap_rule_rnd, ap_prev, ap_rec, ap_seen = [], [], [], [], []
    n = len(test.src_node_ids)
    for k in range(0, n, B):
        src, dst, tt = test.src_node_ids[k:k + B], test.dst_node_ids[k:k + B], test.node_interact_times[k:k + B]
        step = index[float(tt[0])]
        steps = [index[x] for x in tt.tolist()]
        _, neg_r = rnd.sample(size=len(src))
        hs, hd = hist.sample(size=len(src), batch_src_node_ids=src, batch_dst_node_ids=dst,
                             current_batch_start_time=tt[0], current_batch_end_time=tt[-1])
        pos = [key(a, b) for a, b in zip(src.tolist(), dst.tolist())]
        rn = [key(a, b) for a, b in zip(src.tolist(), neg_r.tolist())]
        hn = [key(a, b) for a, b in zip(hs.tolist(), hd.tolist())]

        def prev(p, st):
            return (st - 1) in active.get(p, ())

        def seen(p, st):
            return any(i < st for i in active.get(p, ()))

        s['pos'] += len(pos); s['rnd'] += len(rn); s['h'] += len(hn)
        pp = [prev(p, st) for p, st in zip(pos, steps)]
        s['pos_prev'] += sum(pp)
        s['pos_seen'] += sum(seen(p, st) for p, st in zip(pos, steps))
        rnow = [st in active.get(p, ()) for p, st in zip(rn, steps)]
        s['rnd_now'] += sum(rnow)
        s['rnd_seen'] += sum(seen(p, st) for p, st in zip(rn, steps))
        hp = [prev(p, st) for p, st in zip(hn, steps)]
        s['h_prev'] += sum(hp)
        s['h_now'] += sum(st in active.get(p, ()) for p, st in zip(hn, steps))
        s['h_seen'] += sum(seen(p, st) for p, st in zip(hn, steps))

        y = [1] * len(pos) + [0] * len(pos)
        ap_oracle.append(aps(y, [1.0] * len(pos) + [float(x) for x in rnow]))
        ap_prev.append(aps(y, [float(x) for x in pp] + [float(x) for x in hp]))
        ap_rec.append(aps(y, [recency(p, st) for p, st in zip(pos, steps)] + [recency(p, st) for p, st in zip(hn, steps)]))
        ap_seen.append(aps(y, [float(seen(p, st)) for p, st in zip(pos, steps)] + [float(seen(p, st)) for p, st in zip(hn, steps)]))
        ap_rule_rnd.append(aps(y, [recency(p, st) for p, st in zip(pos, steps)] + [recency(p, st) for p, st in zip(rn, steps)]))

    print(f'\n===== {ds}: {full.num_interactions:,} edges, {full.num_unique_nodes} nodes, {len(times)} distinct times, test edges {n:,}')
    print(f"real test pairs: active one step ago {s['pos_prev'] / s['pos']:.1%}, seen at any earlier time {s['pos_seen'] / s['pos']:.1%}")
    print(f"random fake pairs: truly active at that time {s['rnd_now'] / s['rnd']:.1%}, seen earlier {s['rnd_seen'] / s['rnd']:.1%}")
    print(f"hard fake pairs: active one step ago {s['h_prev'] / s['h']:.1%}, truly active at that time {s['h_now'] / s['h']:.1%}, seen earlier {s['h_seen'] / s['h']:.1%}")
    print(f'random test, perfect-knowledge rule AP {100 * np.mean(ap_oracle):.2f}; recency rule AP {100 * np.mean(ap_rule_rnd):.2f}')
    print(f'hard test, rule AP: seen-before {100 * np.mean(ap_seen):.2f}, active one step ago {100 * np.mean(ap_prev):.2f}, recency {100 * np.mean(ap_rec):.2f}')
    sys.stdout.flush()

# clean training negatives: what does cleaning remove on US Legis?
node_f, edge_f, full, train, val, test, nnv, nnt = get_link_prediction_data('USLegis', 0.15, 0.15, log, convert_time=True)
rng = np.random.default_rng(0)
pool = np.unique(train.dst_node_ids)
at_time = set()
seen_at = {}
for u, v, t in zip(train.src_node_ids.tolist(), train.dst_node_ids.tolist(), train.node_interact_times.tolist()):
    at_time.add((key(u, v), t))
    seen_at.setdefault(key(u, v), t)
raw_now = raw_seen = clean_seen = clean_n = tot = 0
for u, t in zip(train.src_node_ids.tolist(), train.node_interact_times.tolist()):
    v = int(rng.choice(pool))
    tot += 1
    now = (key(u, v), t) in at_time or u == v
    before = seen_at.get(key(u, v), np.inf) < t
    raw_now += now
    raw_seen += before
    tries = 0
    while now and tries < 20:
        v = int(rng.choice(pool)); tries += 1
        now = (key(u, v), t) in at_time or u == v
    clean_n += 1
    clean_seen += seen_at.get(key(u, v), np.inf) < t
print(f'\nUS Legis training negatives: raw truly active {raw_now / tot:.1%}, raw seen earlier {raw_seen / tot:.1%}; after cleaning seen earlier {clean_seen / clean_n:.1%}')
