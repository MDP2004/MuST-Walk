"""Final tables and verdicts from every source: python final_analysis.py   (run from local/, writes analysis/final_tables.txt and analysis/units.csv)

Reads the Kaggle sessions, the laptop and both cluster accounts. A run with no result is shown as missing, never estimated.
Rules come from main_experiment.md and large_datasets.md and their addenda.
"""
import csv
import glob
import itertools
import os
import pathlib
import re
import statistics as st

ROOT = pathlib.Path(__file__).resolve().parent
os.makedirs(ROOT / 'analysis', exist_ok=True)
lines = []


def say(text=''):
    lines.append(text)
    print(text)


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


def holm(pvals):
    order = sorted(pvals, key=pvals.get)
    adj, top = {}, 0.0
    for i, k in enumerate(order):
        top = max(top, min(1.0, (len(order) - i) * pvals[k]))
        adj[k] = top
    return adj


# ---------- main experiment (V3) ----------
SOURCES = [(pathlib.Path(p).parents[1] if p.endswith('.csv') else None, p) for p in
           glob.glob(str(ROOT / 'kaggle/v3/*_out/out/results_units.csv')) +
           [str(ROOT / 'out/vote/results_units.csv')]]
SOURCES = [(pathlib.Path(p).parent, p) for _, p in SOURCES]
for name in ('vote1', 'vote2'):
    p = ROOT / f'cluster/{name}/results/results_units.csv'
    if p.exists():
        SOURCES.append((p.parent, str(p)))


def hardware(folder):
    f = folder / 'gpu.txt'
    if f.exists():
        return open(f, errors='replace').readline().split(',')[0].strip()
    return 'RTX 3050 laptop' if 'out' in folder.parts and 'vote' in folder.parts and 'kaggle' not in folder.parts else 'Tesla T4'


def epochs_of(folder, unit):
    f = folder / 'unit_logs' / f'{unit}_train.txt'
    if not f.exists():
        return None, None
    last, best, cur = None, None, None
    for line in open(f, errors='replace'):
        m = re.search(r'INFO:root:Epoch: (\d+), learning rate', line)
        if m:
            cur = last = int(m.group(1))
        elif 'INFO:root:save model' in line:
            best = cur
    return last, best


v3, seen = {}, {}
for folder, path in SOURCES:
    log = open(folder / 'runner.log', errors='replace').read() if (folder / 'runner.log').exists() else ''
    for r in csv.DictReader(open(path)):
        if r['done'] != 'True':
            continue
        r['truncated'] = f"{r['unit']} in" in log and bool(re.search(re.escape(r['unit']) + r' in [0-9.]+ min \(truncated\)', log))
        r['hardware'] = hardware(folder)
        r['epochs'], r['best'] = epochs_of(folder, r['unit'])
        r['source'] = folder.parent.name if folder.name == 'out' else folder.parent.name
        seen.setdefault(r['unit'], []).append(r)
syn_mixed = {u: rs[-1] for u, rs in seen.items() if u.startswith('SY_')}  # six synthetic runs came back through the mix sessions
for unit, rs in seen.items():
    if unit.startswith('SY_'):
        continue
    vals = {r['test_ap_hist'] for r in rs}
    if len(vals) > 1:
        say(f'NOTE {unit} has two different results from {[r["source"] for r in rs]}: {sorted(vals)}; the last read is used')
    v3[unit] = rs[-1]

DATA = ['USLegis', 'CanParl', 'UNtrade', 'UNvote', 'uci', 'enron']
CFG = ['B', 'M', 'X', 'E', 'HB', 'HM', 'HX']
NAMES = {'test_ap_hist': 'Historical test AP (primary)', 'test_ap_rnd': 'Random test AP', 'test_ap_ind': 'Inductive test AP'}


def val(d, c, s, key='test_ap_hist'):
    r = v3.get(f'V3_{d}_{c}_s{s}')
    return None if r is None or not r[key] else float(r[key]) * 100


say(f'MAIN EXPERIMENT: {len(v3)} of {len(DATA) * len(CFG) * 3} runs have results')
for key, title in NAMES.items():
    say(title)
    for d in DATA:
        cells = []
        for c in CFG:
            x = [v for v in (val(d, c, s, key) for s in range(3)) if v is not None]
            cells.append(f'{c} {st.mean(x):.2f}+-{st.stdev(x) if len(x) > 1 else 0:.2f}' + ('' if len(x) == 3 else f'(n={len(x)})') if x else f'{c} missing')
        say(f'  {d:8} ' + ' | '.join(cells))
say('Runs labelled truncated, or not on a T4')
for u, r in sorted(v3.items()):
    if r['truncated'] or r['hardware'] != 'Tesla T4':
        say(f"  {u:20} {r['hardware']:32} last epoch {r['epochs']}, best {r['best']}{'  TRUNCATED' if r['truncated'] else ''}")


def gains(a, b, d):
    pairs = [(val(d, a, s), val(d, b, s)) for s in range(3)]
    return [x - y for x, y in pairs if x is not None and y is not None]


def verdict(g):
    return 'open' if len(g) < 3 else 'SUPPORTED' if st.mean(g) >= 1.0 and sum(x > 0 for x in g) >= 2 else 'not supported'


say('Per-dataset questions: supported = mean gain >= +1.0 AP and at least 2 of 3 seeds won')
for name, a, b in [('Q1', 'M', 'B'), ('Q2', 'X', 'M'), ('Q3', 'E', 'B'), ('Q4', 'HB', 'B'), ('Q4', 'HM', 'M')]:
    for d in DATA:
        g = gains(a, b, d)
        v = verdict(g)
        if name == 'Q3' and v == 'SUPPORTED':
            v += ', keeps half of M' if st.mean(g) >= 0.5 * st.mean(gains('M', 'B', d)) else ', loses over half of M'
        say(f'  {name} {a:2} vs {b:2} {d:8} mean {st.mean(g):+6.2f} wins {sum(x > 0 for x in g)}/{len(g)}  ' + ' '.join(f'{x:+.2f}' for x in g) + f'  -> {v}')

say('Q5 across datasets: exact one-sided Wilcoxon on per-dataset mean gains, Holm over 4; claim needs adjusted p < 0.05 and mean gain >= +1.0')
Q5 = [('M', 'B'), ('X', 'M'), ('HM', 'M'), ('E', 'B')]
per = {q: [st.mean(gains(*q, d)) for d in DATA] for q in Q5}
raw = {q: wilcoxon_greater(per[q]) for q in Q5}
adj = holm(raw)
for q in Q5:
    m = st.mean(per[q])
    say(f"  {q[0]:2} vs {q[1]:2} gains {' '.join(f'{x:+.2f}' for x in per[q])} mean {m:+.2f} p {raw[q]:.4f} Holm {adj[q]:.4f} -> {'SUPPORTED' if adj[q] < 0.05 and m >= 1.0 else 'not supported'}")
say(f'  Note: with 6 datasets and 4 tests the smallest possible Holm-adjusted p is 4/64 = {4 / 64:.4f}, so no comparison could reach 0.05. This was not noticed when the rule was written.')
say('Exploratory, not pre-registered')
for q in Q5 + [('HB', 'B')]:
    g = [x for d in DATA for x in gains(*q, d)]
    say(f"  {q[0]:2} vs {q[1]:2} over (dataset, seed) pairs, seeds not independent: n={len(g)} wins {sum(x > 0 for x in g)} mean {st.mean(g):+.2f} exact one-sided p {wilcoxon_greater(g):.6f}")
g = [st.mean(gains('HB', 'B', d)) for d in DATA]
say(f"  HB vs B across datasets: gains {' '.join(f'{x:+.2f}' for x in g)} mean {st.mean(g):+.2f} exact one-sided p {wilcoxon_greater(g):.4f}")
for d in DATA:
    g = gains('HM', 'HB', d)
    say(f"  HM vs HB {d:8} mean {st.mean(g):+6.2f} wins {sum(x > 0 for x in g)}/{len(g)}  " + ' '.join(f'{x:+.2f}' for x in g))

# ---------- large datasets (TGB) ----------
LANES = ROOT / 'kaggle/tgb/lanes'
order = ['small_out2', 'small_r2_out', 'review_out', 'review_r2_out', 'wiki_out', 'wiki_r2_out', 'wiki_r3_out', 'wiki_r4_out', 'wiki_r5_out']
tgb = {}
for name in order:  # later sessions replace earlier rows: wiki_r5 holds the M_s1 rerun the addendum names
    folder = LANES / name / 'out'
    if not (folder / 'results.csv').exists():
        continue
    hw = open(folder / 'gpu.txt', errors='replace').readline().split(',')[0].strip() if (folder / 'gpu.txt').exists() else 'Tesla T4'
    for row in csv.reader(open(folder / 'results.csv')):
        if len(row) >= 7 and row[0] != 'unit':
            tgb.setdefault(row[0], []).append(dict(val=row[3], test=row[4], epochs=row[5], state=row[6], hardware=hw, source=name))
for row in csv.reader(open(ROOT / 'cluster/job3/results/results.csv')):
    if len(row) >= 7 and row[0] != 'unit':
        tgb.setdefault(row[0], []).append(dict(val=row[3], test=row[4], epochs=row[5], state=row[6], hardware='NVIDIA RTX 6000 Ada Generation', source='job3'))
final = {u: rs[-1] for u, rs in tgb.items()}
say()
say(f'LARGE DATASETS: {len(final)} units have results (52 planned)')
for u, rs in sorted(tgb.items()):
    if len(rs) > 1:
        say(f"  {u} has {len(rs)} results: " + ', '.join(f"{r['source']} test {r['test']} epochs {r['epochs']} {r['state']}" for r in rs) + ' (last one used)')
TDATA = ['tgbl-enron', 'tgbl-uci', 'tgbl-wiki', 'tgbl-review']


def tv(d, c, s):
    r = final.get(f'TGB_{d}_{c}_s{s}')
    return None if r is None or r['test'] in ('', 'None') else float(r['test'])


for d in TDATA:
    default = [r for u, r in final.items() if u.startswith(f'TGB_{d}_D')]
    say(f"  {d}: default config {default[0]['test'] if default else 'missing'}")
    for c in ['B', 'M', 'X', 'HM']:
        x = [(s, tv(d, c, s), final.get(f'TGB_{d}_{c}_s{s}')) for s in range(3)]
        got = [v for _, v, _ in x if v is not None]
        detail = ' '.join(f"s{s} {v:.4f}({r['epochs']}ep{',T' if r['state'] == 'truncated' else ''})" if v is not None else f's{s} missing' for s, v, r in x)
        say(f"    {c:3} mean {st.mean(got):.4f} std {st.stdev(got) if len(got) > 1 else 0:.4f}  {detail}")
say('TGB verdicts: supported = mean gain >= +0.005 MRR and at least 2 of 3 seeds won')
for a, b in [('M', 'B'), ('X', 'M'), ('HM', 'M')]:
    for d in TDATA:
        g = [tv(d, a, s) - tv(d, b, s) for s in range(3) if tv(d, a, s) is not None and tv(d, b, s) is not None]
        v = 'open' if len(g) < 3 else 'SUPPORTED' if st.mean(g) >= 0.005 and sum(x > 0 for x in g) >= 2 else 'not supported'
        say(f"  {a:2} vs {b} {d:12} mean {st.mean(g):+.4f} wins {sum(x > 0 for x in g)}/{len(g)}  " + ' '.join(f'{x:+.4f}' for x in g) + f'  -> {v}')

# ---------- completion check ----------
a1 = list(csv.DictReader(open(ROOT / 'out/a1/results_units.csv')))
ld = list(csv.DictReader(open(ROOT / 'kaggle/ld_out/out/results_units.csv'))) if (ROOT / 'kaggle/ld_out/out/results_units.csv').exists() else []
syn = {}
for f in glob.glob(str(ROOT / 'kaggle/syn*_out/out/results_units.csv')) + glob.glob(str(ROOT / 'kaggle/syn*_out/out/*/results_units.csv')):
    for r in csv.DictReader(open(f)):
        if r['done'] == 'True':
            syn[r['unit']] = r
syn.update(syn_mixed)
n_a1, n_ld, n_syn = sum(r['done'] == 'True' for r in a1), sum(r['done'] == 'True' for r in ld), len(syn)
say()
say('SYNTHETIC: historical test AP, seeds 0 and 1')
sv = {c: [float(syn[f'SY_uci_{c}_s{s}']['test_ap_hist']) * 100 for s in range(2)] for c in ['S4', 'S5', 'S6', 'S0', 'M', 'X']}
for c, x in sv.items():
    say(f'  {c:3} {x[0]:.2f} {x[1]:.2f}')
best = [max(sv[c][s] for c in ['S4', 'S5', 'S6', 'S0']) for s in range(2)]
g1 = [sv['M'][s] - best[s] for s in range(2)]
g2 = [sv['X'][s] - sv['M'][s] for s in range(2)]
say(f"  M vs best single rate {g1[0]:+.2f} {g1[1]:+.2f} -> {'PASSED' if min(g1) >= 2.0 else 'failed'}   X vs M {g2[0]:+.2f} {g2[1]:+.2f} -> {'PASSED' if min(g2) >= 1.0 else 'failed'}")
trunc_v3 = sum(r['truncated'] for r in v3.values())
trunc_tgb = sum(r['state'] == 'truncated' for r in final.values())
say()
say('COMPLETION CHECK')
say(f'  Main experiment     {len(v3):3} of 126  ({trunc_v3} truncated)')
say(f'  Large datasets      {len(final):3} of  52  ({trunc_tgb} truncated)')
say(f'  Ablations (A1)      {n_a1:3} of  24')
say(f'  Learned decay (LD)  {n_ld:3} of  18')
say(f'  Synthetic           {n_syn:3} of  12')
say(f'  Total               {len(v3) + len(final) + n_a1 + n_ld + n_syn:3} of 232')
missing = [f'V3_{d}_{c}_s{s}' for d in DATA for c in CFG for s in range(3) if f'V3_{d}_{c}_s{s}' not in v3]
say(f'  Missing main runs: {missing or "none"}')

(ROOT / 'analysis/final_tables.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
with open(ROOT / 'analysis/units.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['unit', 'set', 'hardware', 'epochs_reached', 'best_epoch', 'truncated', 'test_primary', 'test_random', 'test_inductive', 'source'])
    for u, r in sorted(v3.items()):
        w.writerow([u, 'main', r['hardware'], r['epochs'], r['best'], r['truncated'], r['test_ap_hist'], r['test_ap_rnd'], r['test_ap_ind'], r['source']])
    for u, r in sorted(final.items()):
        w.writerow([u, 'large', r['hardware'], r['epochs'], '', r['state'] == 'truncated', r['test'], '', '', r['source']])
