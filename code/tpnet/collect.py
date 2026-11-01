"""
Collect job results into results_units.csv and print the summary + the pre-registered test.

Per unit (dataset, config, seed) it reads:
  - validation AP (random negatives) of the kept checkpoint: from unit_logs/<unit>_train.txt
  - test AP/AUC random negatives: saved_results/<unit>_link_<dataset>_TPNet_seed<seed>.json
  - test AP historical / inductive: saved_results/<unit>_link_<strategy>_<dataset>_TPNet_seed<seed>.json
Pre-registered test (pilot records): for each of C1..C5 vs C0, primary metric = historical test AP,
paired over the (dataset, seed) pairs that both have; one-sided exact Wilcoxon signed-rank; Holm correction over
the 5 configs; claim needs adjusted p < 0.05 AND mean gain >= 1.0 AP.
Usage (from the TPNet repo root): python collect.py --runlist runlist_job1.txt --out <job dir>
"""
import argparse
import csv
import itertools
import json
import os
import re
import statistics as st

p = argparse.ArgumentParser()
p.add_argument('--runlist', required=True)
p.add_argument('--out', required=True)
args = p.parse_args()


def read_units(path):
    units = []
    for raw in open(path):
        line = raw.split('#', 1)[0].strip()
        if line:
            uid, ds, seed, flags = [x.strip() for x in line.split('|')]
            units.append((uid, ds, int(seed), flags))
    return units


def kept_val_ap(log_path):
    if not os.path.exists(log_path):
        return None
    last, kept = None, None
    for line in open(log_path, errors='replace'):
        m = re.match(r'INFO:root:validate average_precision, ([0-9.]+)', line)
        if m:
            last = float(m.group(1))
        elif 'INFO:root:save model' in line:
            kept = last
    return kept


def json_metric(path, block, key):
    if not os.path.exists(path):
        return None
    return float(json.load(open(path))[block][key])


rows = []
for uid, ds, seed, flags in read_units(args.runlist):
    config = uid.split('_')[-2]  # <dataset>_<config>_s<seed>, optionally with a prefix such as V2_
    r = dict(unit=uid, dataset=ds, config=config, seed=seed,
             done=os.path.exists(os.path.join(args.out, 'state', 'done', uid + '.ok')),
             val_ap=kept_val_ap(os.path.join(args.out, 'unit_logs', f'{uid}_train.txt')),
             test_ap_rnd=json_metric(f'saved_results/{uid}_link_{ds}_TPNet_seed{seed}.json', 'test metrics', 'average_precision'),
             test_auc_rnd=json_metric(f'saved_results/{uid}_link_{ds}_TPNet_seed{seed}.json', 'test metrics', 'roc_auc'),
             test_ap_hist=json_metric(f'saved_results/{uid}_link_historical_{ds}_TPNet_seed{seed}.json', 'test metrics', 'average_precision'),
             test_ap_ind=json_metric(f'saved_results/{uid}_link_inductive_{ds}_TPNet_seed{seed}.json', 'test metrics', 'average_precision'))
    rows.append(r)

with open(os.path.join(args.out, 'results_units.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

metrics = ['val_ap', 'test_ap_rnd', 'test_auc_rnd', 'test_ap_hist', 'test_ap_ind']


def ms(vals):
    vals = [100 * v for v in vals if v is not None]
    if not vals:
        return 'n/a'
    return f'{st.mean(vals):.2f} ± {st.stdev(vals):.2f} (n={len(vals)})' if len(vals) > 1 else f'{vals[0]:.2f} (n=1)'


print(f"{'dataset':8s} {'config':8s} | " + ' | '.join(f'{m:22s}' for m in metrics))
groups = {}
for r in rows:
    groups.setdefault((r['dataset'], r['config']), []).append(r)
for (ds, cfg), rs in groups.items():
    print(f'{ds:8s} {cfg:8s} | ' + ' | '.join(f'{ms([x[m] for x in rs]):22s}' for m in metrics))


def wilcoxon_greater(diffs):
    """Exact one-sided Wilcoxon signed-rank p-value for H1: median diff > 0 (zeros dropped, average ranks)."""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return 1.0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    w_obs = sum(r for r, x in zip(ranks, d) if x > 0)
    count = sum(1 for signs in itertools.product([0, 1], repeat=n)
                if sum(r for r, s in zip(ranks, signs) if s) >= w_obs - 1e-12)
    return count / 2 ** n


print('\nPre-registered test: historical test AP, Ci vs C0, paired over (dataset, seed); exact one-sided Wilcoxon, Holm')
base = {(r['dataset'], r['seed']): r['test_ap_hist'] for r in rows if r['config'] == 'C0' and r['test_ap_hist'] is not None}
results = []
for cfg in ['C1', 'C2', 'C3', 'C4', 'C5']:
    diffs = [100 * (r['test_ap_hist'] - base[(r['dataset'], r['seed'])]) for r in rows
             if r['config'] == cfg and r['test_ap_hist'] is not None and (r['dataset'], r['seed']) in base]
    results.append((cfg, diffs, wilcoxon_greater(diffs) if diffs else None))
valid = sorted([x for x in results if x[2] is not None], key=lambda x: x[2])
m = len(valid)
adj, running_max = {}, 0.0
for i, (cfg, diffs, pval) in enumerate(valid):
    running_max = max(running_max, min(1.0, (m - i) * pval))
    adj[cfg] = running_max
for cfg, diffs, pval in results:
    if pval is None:
        print(f'  {cfg}: no paired data yet')
        continue
    mean = st.mean(diffs)
    claim = adj[cfg] < 0.05 and mean >= 1.0
    print(f'  {cfg}: n={len(diffs)} pairs, mean gain {mean:+.2f} AP, wins {sum(x > 0 for x in diffs)}/{len(diffs)}, '
          f'p={pval:.4f}, Holm-adjusted p={adj[cfg]:.4f} -> {"SUPPORTED" if claim else "not supported"}')
done = sum(r['done'] for r in rows)
print(f'\nunits done: {done}/{len(rows)}  (results_units.csv written to {args.out})')
