import csv, re, statistics as st
import os
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "..", "results", "")
rows = list(csv.DictReader(open(R + "runs_main_and_large.csv")))
d = {}
for r in rows:
    m = re.match(r"V3_(\w+?)_([A-Z]+)_s(\d)", r["unit"])
    if m and r["set"] == "main":
        d[(m[1], m[2], int(m[3]))] = (float(r["test_primary"]) * 100, float(r["test_random"]) * 100, float(r["test_inductive"]) * 100)
tot = 0
for ds in ["USLegis", "CanParl", "UNtrade", "UNvote", "uci", "enron"]:
    g = [d[(ds, "HM", s)][0] - d[(ds, "B", s)][0] for s in range(3)]
    w = sum(x > 0 for x in g); tot += w
    print(ds, "hist gain", round(st.mean(g), 2), [round(x, 2) for x in g], "wins", w,
          "rnd", round(st.mean(d[(ds, "HM", s)][1] - d[(ds, "B", s)][1] for s in range(3)), 2),
          "ind", round(st.mean(d[(ds, "HM", s)][2] - d[(ds, "B", s)][2] for s in range(3)), 2))
print("wins", tot, "of 18")
ft = open(R + "final_tables.txt", encoding="utf-8").read()
for ds in ["tgbl-wiki", "tgbl-review", "tgbl-enron", "tgbl-uci"]:
    b = ft.split(f"  {ds}: default config ")[1]
    get = lambda c: [float(x) for x in re.findall(r"s\d ([\d.]+)\(", re.search(rf"^\s+{c}\s+mean.*$", b, re.M).group(0))]
    B, H = get("B"), get("HM")
    print(ds, "HM-B", [round(h - x, 4) for h, x in zip(H, B)], "mean", round(st.mean(H) - st.mean(B), 4))
