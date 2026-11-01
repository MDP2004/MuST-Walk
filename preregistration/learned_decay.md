# Pre-registration - learnable decay under strict time (written 30 Oct 2026, before any result of these configurations exists)

Known before writing: D4 (negative on Can. Parl., 30/5, leak present), the V3 rules, pilot P1, the V3 smoke test. No V3 lane has returned.
Purpose: a fair second test of learnable decay, because D4 ran on the leaky default protocol and with truncated training.

Protocol as V3: 100 epochs / patience 20, `--rp_strict_time`, seeds 0, 1, 2, US Legis and Can. Parl. Lazy does not apply, because tangent states are kept for
every node. Code: V3 + `apply_learned_decay_patch.py`. Rates are learned in log-space with Adam at 1e-2, as in D4.

Configurations
- LS  one learned rate, starting at a 1-year half-life (2.2e-8)
- LP  one learned rate per hop depth, same start
- LM  the dataset's V3 ladder, each scale learned per depth (0 starts at 1e-12)

Compared with V3 on the same seeds: LS and LP vs B, LM vs M. Primary metric: test AP with historical negatives. Selection on random-negative validation.
"Supported" = mean gain >= +1.0 AP and wins in at least 2 of 3 seeds. Descriptive only. Learned rates are reported per seed.
Units not finished are reported as missing. Cutting any part needs the user's explicit permission.
