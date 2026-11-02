"""Recompute the research paper's figures from the result files and analysis output, confirm each appears in the LaTeX source, and scan scope."""
import csv
import json
import re
import statistics as st
import subprocess

import os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "")
RES = ROOT + "results/"
tex = open(ROOT + "docs/MuST-Walk_Research_Paper.tex", encoding="utf-8").read()
ft = open(RES + "final_tables.txt", encoding="utf-8").read()
DS = ["USLegis", "CanParl", "UNtrade", "UNvote", "uci", "enron"]
J = {d: json.load(open(ROOT + f"analysis/paper_checks/structure_out/{d}.json")) for d in DS}
miss, ok = [], 0


def need(x, why):
    global ok
    if x in tex:
        ok += 1
    else:
        miss.append((why, x))


def sg(v, d=2):
    return f"+{v:.{d}f}" if v >= 0 else f"$-{abs(v):.{d}f}$"


sect = {}
for name, head in [("hist", "Historical test AP"), ("rnd", "Random test AP"), ("ind", "Inductive test AP")]:
    body = ft.split(head)[1]
    for d in DS:
        line = re.search(rf"^\s+{d}\s+(.*)$", body, re.M).group(1)
        for c in ["B", "M", "HM"]:
            m = re.search(rf"\b{c} ([\d.]+)\+-([\d.]+)", line)
            sect[(name, d, c)] = (float(m.group(1)), float(m.group(2)))
runs = {}
for r in csv.DictReader(open(RES + "runs_main_and_large.csv")):
    m = re.match(r"V3_(\w+?)_([A-Z]+)_s(\d)", r["unit"])
    if m and r["set"] == "main":
        runs[(m[1], m[2], int(m[3]))] = [float(r[k]) * 100 for k in ("test_primary", "test_random", "test_inductive")]
gain = lambda d, a, i: st.mean(runs[(d, a, s)][i] - runs[(d, "B", s)][i] for s in range(3))
for d in DS:
    for c in ["B", "M", "HM"]:
        a, b = sect[("hist", d, c)]
        need(f"{a:.2f}$\\pm${b:.2f}", f"main {d} {c}")
    need(sg(round(gain(d, "HM", 0), 2)), f"HM gain {d}")
    wins = sum(runs[(d, "HM", s)][0] > runs[(d, "B", s)][0] for s in range(3))
    assert wins == 3, d
    for a in ("M", "HM"):
        need(sg(round(gain(d, a, 1), 2)), f"rnd {a} {d}")
        need(sg(round(gain(d, a, 2), 2)), f"ind {a} {d}")
for ds, mean, seeds in re.findall(r"Q1 M  vs B  (\w+)\s+mean\s+(\S+) wins \d/3\s+(\S+ \S+ \S+)", ft):
    need(sg(float(mean)), f"gain M {ds}")
    for v in seeds.split():
        need(v.lstrip("+"), f"seed {ds}")
bseeds = [57.20, 63.26, 78.30]
base = 66.25
for line in open(RES + "ablations_fixed_code_table.txt", encoding="utf-8"):
    m = re.match(r"(\w+)\s+([\d.]+) \+/-\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
    if m and m.group(1) not in ("CN", "EL"):
        v = float(m.group(2))
        need(f"{v:.2f}$\\pm${float(m.group(3)):.2f}", f"ablation {m.group(1)}")
        if m.group(1) != "M":
            need(f"+{v - base:.2f}", f"ablation gain {m.group(1)}")
        print(m.group(1), "seeds won vs TPNet", sum(float(m.group(4 + i)) > bseeds[i] for i in range(3)))
for ds in ["tgbl-wiki", "tgbl-review", "tgbl-enron", "tgbl-uci"]:
    block = ft.split(f"  {ds}: default config ")[1]
    need(block.split()[0], f"{ds} no strict")
    for c in ["B", "M", "HM"]:
        m = re.search(rf"^\s+{c}\s+mean ([\d.]+) std ([\d.]+)", block, re.M)
        need(f"{m.group(1)}$\\pm${m.group(2)}", f"{ds} {c}")
for d in DS:
    r = J[d]
    f = r["feature_level"]
    s = r["structure"]
    t = str(r["tuned"])
    for k in [t, "picked_ap", "ladder", "seen_tuned", "seen_ladder", "tuned_random_train", "ladder_random_train", "rule_novel_aware"]:
        need(f"{f[k]['mean'] * 100:.2f}", f"{d} {k}")
    need(f"{s['density']:.3f}", f"{d} density")
    need(f"{s['step_persistence']:.3f}", f"{d} persistence")
    need(f"{r['nodes']['stay_next_period']:.3f}", f"{d} stay")
    need(f"{r['positives']['never_seen_stored_direction'] * 100:.1f}\\%", f"{d} free")
    need(f"{(1 + r['positives']['never_seen_stored_direction']) * 50:.1f}", f"{d} floor")
    need(f"{r['resolution'][t]['median_sep']:.2f}", f"{d} sep")
    sk = r["sketch_loss"]["ladder"]
    need(f"{sk['exact'] * 100:.2f}", f"{d} exact")
    need(f"{sk['sketched'] * 100:.2f}", f"{d} sketched")
    for a, p in r["memory_kernel"].items():
        if p["p"] > 0:
            need(f"{p['p']:.4f}".rstrip("0"), f"{d} kernel {a}")
print("checked", ok, "missing", len(miss))
for m in miss:
    print("  MISSING", m)
banned = ["MuST-Cross", "MuST-Equal", "Cross-HN", "TPNet-HN", "learned decay", "EdgeBank", "GraphMixer", "erases"]
print("scope hits:", {b: tex.count(b) for b in banned if tex.count(b)})
