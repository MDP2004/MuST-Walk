# Pre-registration - TGB experiment (written 30 Oct 2026, before any real TGB result of these configurations exists)

Known before writing: all of `main_experiment.md`, pilot P1, the V3 smoke test, one earlier reproduction of TPNet on tgbl-wiki (test MRR 0.8259,
leaderboard 0.827), the size of each dataset. No V3 lane has returned. The 12-step CPU rehearsal used a fake dataset only.

Code: TGB_TPNet commit 86d84e2 + `apply_tgb_patch.py` + `apply_tgb_features_patch.py`, py-tgb 2.3.0.
Protocol: the authors' `--load_best_configs`, 30 epochs, patience 5, seeds 0, 1, 2. Every config uses `--val_fraction 0.2`: per-epoch validation on the
first 20% of validation edges in time order, best epoch chosen on that, then one full validation and test. The official TGB evaluator gives MRR.
If a unit hits its time limit, the best epoch saved so far gets the full validation and test, and the unit is labelled truncated with its epoch count.

Decay rates (seconds)
- tgbl-wiki:   tuned 1e-6 (authors), ladder {1e-5, 1e-6, 1e-7, 0}
- tgbl-review: tuned 1e-7 (authors), ladder {1e-6, 1e-7, 1e-8, 0}
- tgbl-uci:    tuned 1e-7 (DyGLib UCI), ladder {1e-6, 1e-7, 1e-8, 0}
- tgbl-enron:  tuned 1e-6 (DyGLib Enron), ladder {1e-5, 1e-6, 1e-7, 0}

Configurations
- D   TPNet with tuned decay, default protocol (no strict time), seed 0 only: how much strict time matters on second-resolution data
- B   TPNet with tuned decay, `--rp_strict_time --rp_lazy`
- M   ladder, concat, strict, lazy
- X   M + `--rp_cross`
- HM  M + `--train_neg_hist 0.5`. Validation stays TGB's fixed negatives, so there is no `--select_hist` here

Primary metric: test MRR. Selection on validation only.

Questions (per dataset, mean +/- std over seeds; "supported" = mean gain >= +0.005 MRR and wins in at least 2 of 3 seeds)
T1 M vs B. T2 X vs M. T3 HM vs M.
T4 sanity: B on tgbl-wiki should sit within 0.02 of the leaderboard's 0.827. If not, strict time or the 20% rule changed the result, and T1 to T3 on that
dataset are reported with that warning.

Reporting rules: 4 datasets are too few for a significance claim (smallest one-sided Wilcoxon p = 1/16), so the TGB result is descriptive.
The `--val_fraction` rule applies equally to all configs but is not the authors' protocol, so these numbers are not leaderboard submissions.
Units not finished are reported as missing, never estimated. Anything added after seeing results is labelled exploratory.
Cutting any dataset, seed or configuration needs the user's explicit permission.
Order of work: seed 0 of every config first; enron, uci, wiki, then review.

Addendum, 31 Oct 2026, 08:50 IST (no multi-scale tgbl-wiki result existed when this was written; B and D results did):
every M, X and HM unit on tgbl-wiki ran out of GPU memory at its first validation, because 20 edges x 999 negatives are scored in one call. These units
are rerun with evaluation in pieces of 4,000 node ids, which leaves the sketch and the configs unchanged. The rerun checks on a fake dataset that whole and
pieces give the same MRR. The wiki B and D units ran before the padded-slot mask, so the rerun leaves the mask out and all wiki units share one evaluation.

Addendum, 31 Oct 2026, 12:05 IST (no tgbl-review M, X or HM result existed when this was written; the seed 0 B and D results did):
the user asked to keep all tgbl-review seeds, and Kaggle's quota cannot cover them, so seeds 1 and 2 of B, M, X and HM run on the department cluster as job 3.
All four configs of a seed share one GPU and one time budget (5.3 h of training per unit), so each seed's comparison stays on equal terms. Seed 0 stays on Kaggle T4s.
A faster GPU can reach more epochs inside the budget, so seeds 1 and 2 may not match seed 0's epoch count. Results are reported per seed with the epoch count and hardware.

## Addendum, 31 Oct 2026 (before this result existed)
tgbl-review HM seed 0 runs on another account in session tpnet-tgb-review-r3. Its quota leaves about 10 h, so the session stops at 10 h instead of 11.4 h. The training limit stays 19000 s, as for every review unit. Evaluation has about 4.4 h, and the review evaluation of B took 2.75 h.

## Addendum, 31 Oct 2026, 21:35 IST (M_s0, X_s0, HM_s0 and M_s1 tgbl-wiki results exist; none of the units below has one)
Session tpnet-tgb-wiki-r2 hit its limit with 4 of 9 units returned. M_s1 got 0.4 h of training and stopped after 2 epochs, and X_s1 ran out of time during its final evaluation.
The units with no result (X_s1, X_s2, M_s2, HM_s1, HM_s2) run again with the same flags, code, evaluation and 25000 s training limit.
M_s1 also runs again, because a 2-epoch stop is not a fair run, whatever its score. The rerun is decided before its result exists and is the number used for the verdict. The 2-epoch number (val 0.8321, test 0.8158) is reported beside it. If the rerun does not finish, the 2-epoch number stays, labelled truncated.
X_s0 is labelled truncated but stays as it is: it reached 24 epochs with its best at epoch 19, so its stop made no difference under patience 5.
X seeds 1 and 2 run in tpnet-tgb-wiki-r3 on another account. Its 9.47 h of quota means the session stops at 33000 s instead of 41000 s, which leaves about 7900 s for evaluation; X seed 0 took 7320 s.

Addendum, 31 Oct 2026, 22:40 IST (no result of the units below exists): tgbl-wiki M_s2, HM_s1 and HM_s2 run in tpnet-tgb-wiki-r4 on another account, with the same flags, code and evaluation.
Quota there leaves about 7 h beside another session, so the session stops at 24800 s. The training limit stays 25000 s on paper, but the stop and a 6000 s reserve cap it at about 18800 s. M seed 0 trained in about 12960 s. Any unit that hits the stop is labelled truncated with its epoch count.

Addendum, 31 Oct 2026, 23:00 IST (tgbl-review M_s0 is known, val 0.3067, test 0.3996; X_s0 and HM_s0 have no result; seed 1 of M, X, HM has none yet):
tgbl-review X_s0 trained 3 epochs on a T4 and then timed out in final evaluation, at 52% of the test pass. X needs about 7.9 h to evaluate on a T4, so 5.3 h of training plus evaluation does not fit a Kaggle session. M took 5.0 h to evaluate, not the 2.75 h B took, so HM_s0 in tpnet-tgb-review-r3 has about 4.7 h and will probably time out too.
X_s0 and HM_s0 therefore also run as two extra lanes inside cluster job 3, on the same GPU, with the same code, flags, 19000 s training limit and evaluation as seeds 1 and 2. They start when lane 0 ends.
If HM_s0 finishes on Kaggle, that result is used, because every other seed 0 unit ran on a T4, and the cluster result is reported beside it. Otherwise the cluster result is used. This is fixed before either exists.
X_s0 and possibly HM_s0 are then seed 0 units on different hardware from B_s0 and M_s0, so the report states hardware and epoch count per unit.

Addendum, 1 Nov 2026, 02:35 IST (tgbl-wiki X_s1 and X_s2 are known, test 0.8212 and 0.8197; HM_s1, HM_s2, M_s2 and the M_s1 rerun have no result):
tpnet-tgb-wiki-r4 split its run list by line across its two GPUs, so HM_s2 sits behind M_s2 on GPU 0 and is unlikely to start before that session's 16000 s start limit. The script's description said otherwise; that was a mistake in my build.
another account's session ended early and left 5.15 h of quota. tpnet-tgb-wiki-r5 therefore runs HM_s2 on one T4 and the M_s1 rerun on the other, with the same flags, code and evaluation. The session stops at 18000 s with a 5400 s reserve, so training stops by about 12600 s. M seed 0 trained in about 12960 s, so a unit that hits the stop is labelled truncated and evaluated as usual.
If HM_s2 also starts and finishes in tpnet-tgb-wiki-r4, the wiki-r5 result is used and the other is reported beside it. This is fixed before either exists.

Addendum, 1 Nov 2026, 07:35 IST (tgbl-review M_s2, X_s2, HM_s2, X_s0 and HM_s0 have no result):
Job 3 ends when its four main lanes end, and that also kills the two extra lanes, so their 06:43:30 UTC stop was only reachable if a main lane lasted that long. X_s2 itself had to stop at about 06:01 UTC. Lane 2's supervising process is therefore paused while X_s2's evaluation keeps running, and a small script on the login node resumes it once X_s2, X_s0 and HM_s0 have all ended, or at 06:43 UTC at the latest. X_s2 thus gets up to 42 more minutes of evaluation. No flag, checkpoint or evaluation changes. If X_s2 still does not finish, its checkpoint is evaluated later with the same code, and the report states this.
