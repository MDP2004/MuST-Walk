# Pre-registration - synthetic mechanism demo (written 30 Oct 2026, before any result exists)

Claim under test: when a pair's long-run count carries no information and only recency relative to history does, several decay rates beat any
single rate, and cross-scale features add to that.

Data: `verify/make_synthetic.py`, 30,000 edges, each source's partners swap between active and lapsed every 4,000 edges.
Model: TPNet with `--not_encode`, so scores come from the sketch alone. 100 epochs / patience 20, seeds 0 and 1, strict + lazy.
Configs: single rate S4 (1e-4), S5 (1e-5), S6 (1e-6), S0 (0); M = concat {1e-4, 1e-5, 1e-6, 0}; X = M + cross-scale.
Metric: test AP with historical negatives.

Rules: "several rates help" if M beats the best single rate by >= 2.0 AP in both seeds. "Cross-scale helps" if X beats M by >= 1.0 AP in both seeds.
This is a constructed task, so a pass shows the mechanism can work, not that it works on real data.

## Addendum, 31 Oct 2026 (before these results existed)
SY_uci_M_s0, M_s1, S4_s1 and S5_s1 did not finish in their CPU session, so they run on a Kaggle T4 in session tpnet-v3-mix2, as X_s0 and X_s1 did in tpnet-v3-mix. Code, flags and seeds do not change. The report states which units ran on CPU and which on GPU.
