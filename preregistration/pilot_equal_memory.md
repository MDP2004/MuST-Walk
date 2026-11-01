# Pre-registration - V2 Kaggle lanes (written 30 Oct 2026, before any V1 or V2 result exists)

Protocol for all V2 units: authors' defaults, 100 epochs / patience 20, same code as job 1 (commit 7dd0cf4 + job-1 patch).
Models are selected on validation AP. Test numbers are not used to choose anything.

Configurations
- C0: TPNet with its tuned decay (load_configs.py): US Legis 1e-6, UN Trade 1e-7, Can. Parl. 1e-5, UCI 1e-7, Enron 1e-6.
- C3: yearly datasets, the same ladder as job 1. L: new datasets, ladder fixed by one rule, {10x tuned, tuned, tuned/10, 0}:
  UCI {1e-6, 1e-7, 1e-8, 0}, Enron {1e-5, 1e-6, 1e-7, 0}.
- EQ: the same ladder as C3 or L with sketch dimension d/J via --enforce_dim, so sketch state equals TPNet's.
  d = TPNet's 10*int(ln 2E): US Legis 110 (J=5 -> 22), UN Trade 130 (J=4 -> 32), Can. Parl. 110 (27), UCI 110 (27), Enron 120 (30).

Primary metric: test AP with historical negatives. Secondary: random-negative AP, inductive AP, seconds per run.

Questions and rules (reported per dataset, seeds 0-2, mean +/- std; 3 seeds = descriptive, no significance claim per dataset)
Q1 Does multi-scale help at full training? Support = mean gain >= +1.0 AP on the dataset and wins in at least 2 of 3 seeds.
Q2 Does equal-memory multi-scale keep the gain? Support = EQ gain >= half of C3/L's gain and EQ beats C0 on the dataset.
Q3 Across datasets: exact one-sided Wilcoxon signed-rank on per-dataset mean gains vs C0, for C3/L and for EQ separately,
   alpha 0.05, Holm over these 2 tests. Runs only if at least 5 datasets finish. Claim needs adjusted p < 0.05 and mean gain >= +1.0 AP.

Limits: units not finished by a lane's stop time are reported as missing, never filled in or estimated.
