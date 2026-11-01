# Pre-registration - P1 strict-time pilot (written 30 Oct 2026, before any P1 result exists)

Why: tiny-graph verification showed same-time edges reach the sketch (96% of 1-hop weight on US Legis at the tuned decay).
Question: how much does removing them change real results?

Setup: US Legis, seed 0, 100 epochs / patience 20, laptop RTX 3050, same code as job 1 plus apply_strict_patch.py.
Units: C0 (--rp_lambdas 1e-06), C3 (--rp_lambdas 1e-05 1e-06 1e-07 2.2e-08 0 --rp_fusion concat), each with and without --rp_strict_time.
Model selection on validation only. Primary metric: test AP with historical negatives.

Decision rules (one seed on one dataset, so this picks a protocol, it is not a claim):
- The leak matters if default and strict differ by >= 2.0 historical AP for C0 or for C3.
- If it matters: the Kaggle lanes use --rp_strict_time as the primary protocol, and default runs are kept only to compare with the paper.
- If it does not: the Kaggle lanes keep the default protocol and the report states the issue with this measurement.
- Separately noted, not decided here: C3 minus C0 under strict.
