# Pre-registration - V3 main experiment (written 30 Oct 2026, before any V3 result exists)

Replaces V1 and V2. Known before writing this: job 1, the tiny-graph audit, one finished pilot unit (strict C0 on US Legis),
SNAM's published tables. C3 under strict time has not been seen on any dataset.

Protocol for every unit: authors' defaults 100 epochs / patience 20, `--rp_strict_time --rp_lazy`, code = commit 7dd0cf4 + 3 patches.
Lazy gives the same values as eager (tests pass to about 1e-7), so it changes speed only. Seeds 0, 1, 2.

Configurations (ladder = C3's ladder on yearly data, {10x tuned, tuned, tuned/10, 0} on UCI and Enron)
- B   TPNet with tuned decay
- M   ladder, concat
- X   M + cross-scale features (--rp_cross)
- E   M with sketch dimension d/J (equal memory)
- HB  B + historical-aware training (--train_neg_hist 0.5 --select_hist)
- HM  M + historical-aware training
- HX  X + historical-aware training

Primary metric: test AP with historical negatives. Secondary: random and inductive AP, seconds per run.
Selection on validation only: random-negative validation for B, M, X, E, historical validation for HB, HM, HX.

Questions (per dataset, mean +/- std over seeds; "supported" = mean gain >= +1.0 AP and wins in at least 2 of 3 seeds)
Q1 Multi-scale without the leak: M vs B.
Q2 Cross-scale features: X vs M.
Q3 Equal memory: E vs B, and E keeps at least half of M's gain.
Q4 Historical-aware training: HB vs B, HM vs M.
Q5 Across datasets (at least 5 finished): exact one-sided Wilcoxon on per-dataset mean gains for M vs B, X vs M, HM vs M, E vs B, alpha 0.05,
   Holm over these 4. Claim needs adjusted p < 0.05 and mean gain >= +1.0 AP. With 6 datasets the smallest possible p is 1/64.

Reference columns, not run by us: SNAM's published results (Zhang et al., arXiv 2610.04869, same DyGLib protocol): TPNet strict, EdgeBank, SNAM.
They are compared by mean only, because seeds and code differ.

Reporting rules: units not finished are reported as missing, never estimated. Anything added after seeing V3 results is labelled exploratory.
Cutting any dataset, seed or configuration needs the user's explicit permission.

Addendum, 31 Oct 2026, 22:40 IST (no result of this unit exists; a partial best validation value of 0.7532 was seen): UN Vote HB seed 2 did not finish in tpnet-v3-vote-r2. It restarts from scratch on the laptop RTX 3050 with the same code and flags. B seeds 0 and 2 also ran on the laptop; the report states the hardware of each unit.

Addendum, 31 Oct 2026, 23:03 IST (no complete result exists for UN Vote M_s1 or M_s2; partial validation values up to 0.5738 were seen, no test value):
in sessions of 11.4 h, M_s1 and M_s2 reached only 87 epochs, and a T4 needs about 12.8 h for 100, so they cannot finish in any Kaggle session. They restart from scratch on another account, whose last 7.2 h allows 6.5 h of training, about 50 epochs. M_s0 stopped at epoch 51 under patience 20. At the limit the best checkpoint so far, chosen on validation as usual, gets all three test evaluations.
These two units are labelled truncated with their epoch count and reported apart from complete units. Q1 and Q2 on UN Vote are computed with them and the labels are shown beside the verdict. Other unfinished UN Vote units stay missing unless a later addendum says otherwise.

Addendum, 31 Oct 2026, 23:20 IST (UN Vote HM_s1 has no result; HM_s2 has partial validation values up to 0.7569 and no test value):
tpnet-v3-mix2 ended early, which left 4.33 h of quota on another account. HM_s1 and HM_s2 restart from scratch there with 13700 s (3.8 h) of training, about 29 epochs, then the best checkpoint gets all three test evaluations. HM_s0 had its best epoch at 30 of 50. Both units are labelled truncated with their epoch count, as in the 23:03 addendum, and Q4 HM vs M on UN Vote is computed with them and the labels shown.

Addendum, 1 Nov 2026, 00:25 IST (UN Vote X_s1 has partial validation values and no test value):
Manas agreed to the use of the second cluster account cluster account for X_s1, which no Kaggle quota can hold. X_s1 restarts from scratch in job vote1 on account second cluster account, on an RTX 6000 Ada with torch 2.14.1, with the same code, patches, tests and flags as the Kaggle sessions, at 100 epochs and patience 20. Training stops at 13:00 IST, or after 4 h if the job starts too late for that, and the best checkpoint then gets all three test evaluations. If it stops at the limit, the unit is labelled truncated with its epoch count. The report states this unit's hardware.

Addendum, 1 Nov 2026, 03:40 IST (UN Vote HX_s1 has partial validation values up to 0.7689 and no test value; HX_s0 is known, historical test AP 68.99):
HX_s1 reached epoch 59 on another account, still improving, and the session hit its 41000 s stop before any test evaluation. The checkpoint did not survive the session, so the unit restarts from nothing. No Kaggle account has 10 h free before 7 Nov, and the laptop is ruled out for HX.
HX_s1 therefore runs as job vote2 on the cluster account cluster account, RTX 6000 Ada, torch 2.14.1, same code, flags and tests as the Kaggle V3 sessions, one unit on its own GPU. Training stops after 36000 s and the best checkpoint gets all test evaluations, labelled truncated if it hits that limit. The report states the hardware. HX is not used by any pre-registered question, so this changes no verdict.
