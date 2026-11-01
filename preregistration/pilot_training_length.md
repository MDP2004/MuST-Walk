# Pre-registration - V1 validity test (written 30 Oct 2026, before any V1 result exists)

Why: job 1 used 30 epochs / patience 5. The authors' code defaults are 100 / 20. 73 of 90 job-1 units were stopped by
patience 5, and all three UN Trade C3 units hit the epoch cap. Our C0 on UN Trade is 78.01 random / 71.98 historical,
the paper reports 87.24 / 85.22. Hypothesis: truncated training explains the gap and may have inflated C3's gain.
Checked and rejected already: the random.sample(sorted(...)) fix (held-out node sets are identical on all 3 datasets).

Setup: UN Trade, seed 0, same code as job 1 (commit 7dd0cf4 + job-1 patch), laptop RTX 3050. Only epochs/patience change:
100 / 20. Configs: C0 (--rp_lambdas 1e-07), C3 (--rp_lambdas 1e-05 1e-07 2.2e-08 0 --rp_fusion concat).
Model selected on validation AP only. Test numbers are read only after training ends.

Decision rules (one seed, so this is a screening test, not a claim):
- Truncation confirmed as the main cause if C0 test random AP >= 85.0. Rejected if <= 80.0. In between = partial.
- C3's gain survives if C3 - C0 historical AP >= +1.0 at 100/20. It is "convergence speed only" if the gain is < +1.0.
- Next step either way: seeds 1 and 2 of both configs at 100/20 before any statement in the report.
- If truncation is rejected, look for another cause of the gap before building on UN Trade numbers.

Not changed by this test: job 2's own pre-registration (30/5) stays as written; its result is reported under that protocol.
