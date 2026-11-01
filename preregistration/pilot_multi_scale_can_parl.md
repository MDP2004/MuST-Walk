# Pre-registration — D3 (written 2026-10-28, before any D3 result exists)

Dataset: Can. Parl. (DyGLib protocol, transductive). Code: official TPNet + MuST-Walk patch (models/MuSTWalk.py).
Settings for all configs: --sample_neighbor_strategy uniform --num_epochs 30 --patience 5 --num_runs 3 (seeds 0,1,2).

Configurations
- C0 TPNet, lambda = 1e-5 (TPNet's tuned value), through the MS code path with J = 1 (proven identical to original TPNet on CPU).
- C1 concat {1e-5, 0}; C2 concat {1e-5, 2.2e-8}; C3 concat {1e-5, 1e-7, 2.2e-8, 0}  (concatenation)
- C4 residual average over C3's scales (fixed-average ablation)
- C5 residual softmax gate over C3's scales, beta = 0 init (gate)
- C6 TPNet lambda = 1e-5 with 64 neighbours (confound check: context length vs decay)

Metrics
- Primary: test AP with historical negatives (mean over 3 seeds).
- Secondary: test AP/AUC with random negatives; test AP with inductive negatives.
- Model selection inside each run: TPNet's early stopping on validation (random negatives).
- Choosing among C1–C5: highest mean validation AP (random negatives). Test numbers are never used to choose.
- Per-epoch test lines printed by TPNet's training loop are ignored.

Decision rules
- Sanity: C0 must reproduce D1 (random-neg test AP 89.04 ± 0.41) within 1.0 AP. Otherwise stop and investigate.
- "Multi-scale helps on Can. Parl.": let C* be the C1–C5 config with the best validation AP.
  Pass if C* − C0 ≥ +2.0 historical AP AND ≥ +0.5 random AP AND C* beats C0 on historical AP in ≥ 2 of 3 seeds.
- "Gate helps (gate)": C5 − C3 ≥ +1.0 historical AP and better in ≥ 2 of 3 seeds.
- "Context length is the confound": C6 − C0 ≥ +2.0 random AP.
- 3 seeds make D3 diagnostic only. Final claims require 5 seeds, all datasets and a paired test with Holm correction.

If "multi-scale helps" fails, that is reported as a null result for Can. Parl.; the same comparison is then run on
US Legis., UN Trade and UN Vote (different tuned lambdas) before any overall conclusion.
