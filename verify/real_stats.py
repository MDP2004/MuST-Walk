"""Measure on the real data how much each thing seen in the tiny trace matters. No model is trained.

Run from local/TPNet_ms:  python ../verify/real_stats.py USLegis UNtrade UNvote
"""
import logging
import math
import random
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
from models.TPNet import RandomProjectionModule as RP
from utils.DataLoader import get_link_prediction_data
from utils.utils import NegativeEdgeSampler

logger = logging.getLogger('x')
logger.addHandler(logging.NullHandler())
TUNED = {'USLegis': 1e-6, 'UNtrade': 1e-7, 'UNvote': 1e-5, 'CanParl': 1e-5, 'uci': 1e-7, 'enron': 1e-6}
B = 200
for ds in sys.argv[1:]:
    node_f, edge_f, full, train, val, test, nnv, nnt = get_link_prediction_data(ds, 0.15, 0.15, logger, convert_time=True)
    print(f'\n===== {ds}: {full.num_interactions:,} edges, {full.num_unique_nodes} nodes, {len(np.unique(full.node_interact_times))} distinct times')
    lam = TUNED[ds]

    # 1. same-time edges in the sketch when a test batch is predicted (seen by the sketch, not by the neighbor sampler)
    tt, n = test.node_interact_times, len(test.node_interact_times)
    vt = val.node_interact_times
    all_t = np.concatenate([train.node_interact_times, vt, tt])
    offset = len(train.node_interact_times) + len(vt)
    same, share, gap = [], [], []
    for s in range(0, n, B):
        t0 = tt[s]
        before = all_t[:offset + s]
        k = int((before == t0).sum())
        w = np.exp(-lam * (t0 - before)).sum()
        same.append(k)
        share.append(k / w if w > 0 else 0.0)
    same = np.array(same)
    print(f'[same-time] test batches {len(same)}: batches holding same-time edges {np.mean(same > 0):.1%}, mean same-time edges in sketch {same.mean():,.0f}, '
          f'share of tuned-lambda ({lam:g}) walk-1 weight from same-time edges {np.mean(share):.1%}')

    # 2. order inside one timestamp
    if len(np.unique(full.node_interact_times)) < 1000:
        mono = np.mean([np.all(np.diff(full.src_node_ids[full.node_interact_times == t]) >= 0) for t in np.unique(full.node_interact_times)])
        print(f'[order] timestamps whose edges are sorted by source id: {mono:.1%}')

    # 3. random negatives that are real edges
    sampler = NegativeEdgeSampler(full.src_node_ids, full.dst_node_ids, seed=2)
    sampler.reset_random_state()
    by_time = {}
    for u, v, t in zip(full.src_node_ids.tolist(), full.dst_node_ids.tolist(), full.node_interact_times.tolist()):
        by_time.setdefault(t, set()).add((u, v))
    bad = self_loop = tot = 0
    for s in range(0, n, B):
        src, dst, t = test.src_node_ids[s:s + B], test.dst_node_ids[s:s + B], tt[s:s + B]
        _, neg = sampler.sample(size=len(src))
        for a, b, c in zip(src.tolist(), neg.tolist(), t.tolist()):
            tot += 1
            self_loop += a == b
            bad += (a, b) in by_time[c] or (b, a) in by_time[c]
    print(f'[negatives] random test negatives that are true edges at that same time: {bad / tot:.1%}, self-loops: {self_loop / tot:.1%}')

    # 4. edges missing from the sketch because they touch held-out nodes
    pre_val = (full.node_interact_times <= val.node_interact_times[0] - 1e-9).sum()
    print(f'[held-out] pre-validation edges {pre_val:,}, in the sketch {train.num_interactions:,}, missing {1 - train.num_interactions / pre_val:.1%}')

    # 5. edge feature scale
    e = edge_f[1:]
    nz = e[:, np.abs(e).max(0) > 0]
    if nz.size:
        print(f'[edge features] non-zero dims {nz.shape[1]}, min {nz.min():.3g}, median {np.median(nz):.3g}, p99 {np.percentile(nz, 99):.3g}, max {nz.max():.3g}')

    # 6. overflow on the real stream, float32 as in training
    d = min(int(math.log(edge_f.shape[0] * 2)) * 10, node_f.shape[0])
    for l in (lam, 0.0):
        torch.manual_seed(0)
        m = RP(node_num=node_f.shape[0], edge_num=edge_f.shape[0], dim_factor=10, num_layer=3, time_decay_weight=l, device='cpu', use_matrix=False,
               beginning_time=np.float64(train.node_interact_times[0]), not_scale=False, enforce_dim=-1)
        with torch.no_grad():
            for s in range(0, full.num_interactions, B):
                m.update(full.src_node_ids[s:s + B], full.dst_node_ids[s:s + B], full.node_interact_times[s:s + B].copy())
            maxes = [float(m.random_projections[i].abs().max()) for i in range(4)]
            u = torch.randint(1, node_f.shape[0], (2000,))
            v = torch.randint(1, node_f.shape[0], (2000,))
            R = torch.cat([torch.stack([m.random_projections[i][u] for i in range(4)], 1), torch.stack([m.random_projections[i][v] for i in range(4)], 1)], 1)
            F = R @ R.transpose(1, 2)
            bad_vals = int((~torch.isfinite(F)).sum())
            F = torch.log(F.clamp(min=0) + 1)
        print(f'[overflow] lambda={l:g} d={d}: max per layer {[f"{x:.3g}" for x in maxes]}, non-finite inner products {bad_vals}/{F.numel()}, log-feature max {float(F[torch.isfinite(F)].max()):.1f}')
