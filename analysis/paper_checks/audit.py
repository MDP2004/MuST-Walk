import csv, statistics as st, random, re, sys
from collections import defaultdict

import os
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "..", "results", "")
rows = list(csv.DictReader(open(R + "runs_main_and_large.csv")))
main = [r for r in rows if r["set"] == "main"]
ds_names = ["USLegis", "CanParl", "UNtrade", "UNvote", "uci", "enron"]


def parse(u):
    m = re.match(r"V3_(\w+?)_([A-Z]+)_s(\d)", u)
    return m.group(1), m.group(2), int(m.group(3))


data = {}
for r in main:
    d, c, s = parse(r["unit"])
    data[(d, c, s)] = (float(r["test_primary"]) * 100, int(r["best_epoch"]), int(r["epochs_reached"]))

print("== A. per-seed historical AP and best epoch, random-validation configs")
for d in ds_names:
    for c in ["B", "M", "X", "E"]:
        v = [data[(d, c, s)] for s in range(3)]
        print(f"{d:8s} {c}  " + "  ".join(f"{a:6.2f} (ep {b:3d})" for a, b, _ in v))

print("\n== B. within-dataset rank link: best epoch vs historical AP, configs B,M,X,E (12 runs each)")


def pearson(x, y):
    mx, my = st.mean(x), st.mean(y)
    sx = sum((a - mx) ** 2 for a in x) ** .5
    sy = sum((b - my) ** 2 for b in y) ** .5
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else float("nan")


def ranks(x):
    o = sorted(range(len(x)), key=lambda i: x[i])
    r = [0] * len(x)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2
        i = j + 1
    return r


for d in ds_names:
    ep, ap = [], []
    for c in ["B", "M", "X", "E"]:
        for s in range(3):
            a, b, _ = data[(d, c, s)]
            ep.append(b)
            ap.append(a)
    print(f"{d:8s} spearman {pearson(ranks(ep), ranks(ap)):+.2f}  best-epoch range {min(ep)}-{max(ep)}")

print("\n== C. false-positive rate of 'mean>=+1 and >=2/3 wins' with zero true effect")
random.seed(0)
for d in ds_names:
    diffs = []
    for s in range(3):
        diffs.append(data[(d, "M", s)][0] - data[(d, "B", s)][0])
    sa = st.stdev([data[(d, "B", s)][0] for s in range(3)])
    sb = st.stdev([data[(d, "M", s)][0] for s in range(3)])
    # sign-flip of observed pairs: every assignment of arm labels is equally likely under the null
    n = 0
    tot = 0
    for m in range(8):
        f = [diffs[i] * (1 if m >> i & 1 else -1) for i in range(3)]
        tot += 1
        if st.mean(f) >= 1 and sum(x > 0 for x in f) >= 2:
            n += 1
    # gaussian null using the two arms' own seed spread
    sd = (sa ** 2 + sb ** 2) ** .5
    N = 200000
    hit = 0
    for _ in range(N):
        f = [random.gauss(0, sd) for _ in range(3)]
        if sum(f) / 3 >= 1 and sum(x > 0 for x in f) >= 2:
            hit += 1
    print(f"{d:8s} seedSD B {sa:5.2f} M {sb:5.2f} | gaussian-null FP {hit/N:.2f} | sign-flip {n}/8")

print("\n== D. multi-scale under hard training, the arm with informative selection")
g = []
w = 0
for d in ds_names:
    x = [data[(d, "HM", s)][0] - data[(d, "HB", s)][0] for s in range(3)]
    g += x
    w += sum(v > 0 for v in x)
print(f"HM-HB mean {st.mean(g):+.2f}, wins {w}/18, per-dataset {[round(st.mean(g[i:i+3]),2) for i in range(0,18,3)]}")
print("seed SD of HN arms vs random-val arms (mean over datasets):")
for c in ["B", "M", "HB", "HM"]:
    print(f"  {c:3s}", round(st.mean(st.stdev([data[(d, c, s)][0] for s in range(3)]) for d in ds_names), 2))

print("\n== E. fixed code: learned shared rate vs ladder vs tuned")
ld = list(csv.DictReader(open(R + "learned_decay.csv")))
print(list(ld[0].keys()))
agg = defaultdict(list)
for r in ld:
    agg[(r.get("dataset"), r.get("config"))].append(float(r["test_ap_hist"]) * 100)
for k, v in sorted(agg.items()):
    print(k, [round(a, 2) for a in v], "mean", round(st.mean(v), 2))
for d in ["USLegis", "CanParl"]:
    print(d, "B", round(st.mean(data[(d, 'B', s)][0] for s in range(3)), 2), "M", round(st.mean(data[(d, 'M', s)][0] for s in range(3)), 2))

print("\n== F. original code: single rates vs tuned vs ladder (validation picks the single rate)")
oc = list(csv.DictReader(open(R + "original_code_ablations.csv")))
agg = defaultdict(list)
for r in oc:
    agg[(r["dataset"], r["config"])].append(r)
for d in ["USLegis", "UNtrade", "UNvote"]:
    print(d)
    single = {}
    for (dd, c), v in sorted(agg.items()):
        if dd != d:
            continue
        va = st.mean(float(x["val_ap"]) * 100 for x in v)
        h = st.mean(float(x["test_ap_hist"]) * 100 for x in v)
        print(f"   {c:9s} n={len(v)} val {va:6.2f} hist {h:6.2f}")
        if c.startswith("G") or c == "C0":
            single[c] = (va, h)
    b = max(single, key=lambda k: single[k][0])
    print(f"   val-picked single rate: {b}, hist {single[b][1]:.2f}; ladder C3 hist {st.mean(float(x['test_ap_hist'])*100 for x in agg[(d,'C3')]):.2f}")

print("\n== G. job-1 vs job-2 on C3 vs C0: see original_code_confirmation_summary.txt")
