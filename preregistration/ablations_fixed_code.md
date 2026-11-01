# Pre-registration - A1 ablations (written 30 Oct 2026, before any A1 result exists)

Known before writing: pilot P1 on US Legis seed 0, and a 1-epoch smoke test of V3 configs.
Protocol: strict + lazy, 100 epochs / patience 20, US Legis, seeds 0, 1, 2. Compared with V3 unit M on the same seeds (full ladder, concat).

Configs
- J3   ladder {1e-05, 1e-07, 0}
- UNI  the uniform ladder {1e-04, 1e-05, 1e-06, 1e-07}
- LOG  log-grid half-lives from one year, {1e-05 (previous step only), 2.2e-08, 5.5e-09, 0}
- AVG  average fusion
- GATE gate fusion
- GN   gate fusion + per-scale LayerNorm
- CN   M + --clean_train_neg
- EL   M + --edge_feat_log

Reporting: mean +/- std of historical AP, descriptive only, no significance claim. Differences under +1.0 AP are reported as no effect.
