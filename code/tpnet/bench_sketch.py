"""Efficiency benchmark of the temporal-walk sketch (update cost, memory, streaming vs recompute).

Run from the patched TPNet repo root:  python bench_sketch.py --device cuda:0 [--sizes 10000 100000 1000000]
For each graph size n (edges E = 10 n, d_R = TPNet's 10*int(ln 2E)) and number of scales J = 1..4 it measures:
  - sketch update time per batch of 200 edges (ms) and per edge (us)
  - pairwise-feature time for 200 query pairs (ms)
  - sketch state memory (MB) and peak GPU memory (MB)
each with the eager update (TPNet, every node written every batch) and the lazy update (--rp_lazy, only the batch's rows), and, for one size, the cost of rebuilding the sketch from scratch (periodic recompute) vs one streaming update.
Synthetic random edges; numbers are about speed/memory only.
"""
import argparse
import math
import time

import numpy as np
import torch

from models.TPNet import RandomProjectionModule
from models.MuSTWalk import LazyRandomProjectionModule, MultiScaleRandomProjectionModule

p = argparse.ArgumentParser()
p.add_argument('--device', default='cuda:0')
p.add_argument('--sizes', type=int, nargs='*', default=[10_000, 100_000, 1_000_000])
p.add_argument('--batches', type=int, default=200)
args = p.parse_args()
dev = args.device
LAMS = [1e-6, 1e-7, 1e-8, 0.0]
B = 200


def sync():
    if dev.startswith('cuda'):
        torch.cuda.synchronize()


def build(J, n, E, lazy):
    common = dict(node_num=n + 1, edge_num=E, dim_factor=10, num_layer=3, device=dev, use_matrix=False,
                  beginning_time=np.float64(0.0), not_scale=False, enforce_dim=-1)
    if J == 1:
        m = (LazyRandomProjectionModule if lazy else RandomProjectionModule)(time_decay_weight=LAMS[0], **common)
    else:
        m = MultiScaleRandomProjectionModule(lambdas=LAMS[:J], fusion='concat', base_index=0, lazy=lazy, **common)
    return m.to(dev)


def batches(n, count, rng, t0=0.0):
    out, t = [], t0
    for _ in range(count):
        src = rng.integers(1, n + 1, B)
        dst = rng.integers(1, n + 1, B)
        ts = np.sort(t + rng.uniform(0, 100, B))
        t = ts[-1]
        out.append((src, dst, ts))
    return out


print(f'device {dev}' + (f' ({torch.cuda.get_device_name(0)})' if dev.startswith('cuda') else ''))
print(f"{'nodes':>9} {'mode':>5} {'J':>2} {'d_R':>4} | {'update ms/batch':>15} {'us/edge':>8} | {'pairwise ms':>11} | {'state MB':>9} {'peak MB':>8}")
for n, lazy, J in [(n, lazy, J) for n in args.sizes for lazy in (False, True) for J in (1, 2, 3, 4)]:
    E = 10 * n
    rng = np.random.default_rng(n + J)  # eager and lazy see the same edges
    torch.manual_seed(0)
    if dev.startswith('cuda'):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    m = build(J, n, E, lazy)
    bl = batches(n, 20 + args.batches, rng)
    with torch.no_grad():
        for src, dst, ts in bl[:20]:
            m.update(src, dst, ts)
        sync(); t0 = time.perf_counter()
        for src, dst, ts in bl[20:]:
            m.update(src, dst, ts)
        sync(); upd = (time.perf_counter() - t0) / args.batches
        q_src, q_dst = rng.integers(1, n + 1, B), rng.integers(1, n + 1, B)
        m.get_pair_wise_feature(q_src, q_dst); sync(); t0 = time.perf_counter()
        for _ in range(20):
            m.get_pair_wise_feature(q_src, q_dst)
        sync(); pw = (time.perf_counter() - t0) / 20
    unique = {t.data_ptr(): t for t in list(m.parameters()) + list(m.buffers())}  # shared P counted once
    state = sum(t.numel() * t.element_size() for t in unique.values()) / 2 ** 20
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if dev.startswith('cuda') else float('nan')
    d = m.dim if J == 1 else m.scales[0].dim
    print(f"{n:>9} {'lazy' if lazy else 'eager':>5} {J:>2} {d:>4} | {upd * 1e3:>15.3f} {upd / B * 1e6:>8.2f} | {pw * 1e3:>11.3f} | {state:>9.1f} {peak:>8.0f}", flush=True)
    del m

# streaming vs periodic recompute: rebuilding the sketch from all past edges vs updating with one new batch
n = args.sizes[min(1, len(args.sizes) - 1)]
E = 10 * n
rng = np.random.default_rng(1)
hist = batches(n, E // B, rng)
for J, lazy in [(1, False), (1, True), (4, False), (4, True)]:
    m = build(J, n, E, lazy)
    with torch.no_grad():
        sync(); t0 = time.perf_counter()
        for src, dst, ts in hist:
            m.update(src, dst, ts)
        sync(); rebuild = time.perf_counter() - t0
        src, dst, ts = batches(n, 1, rng, t0=hist[-1][2][-1])[0]
        sync(); t0 = time.perf_counter()
        m.update(src, dst, ts)
        sync(); one = time.perf_counter() - t0
    print(f'streaming vs recompute (n={n}, {E:,} past edges, J={J}, {'lazy' if lazy else 'eager'}): rebuild from scratch {rebuild:.2f} s, '
          f'one streaming update {one * 1e3:.2f} ms -> recompute is {rebuild / one:,.0f}x the cost of one update')
    del m
