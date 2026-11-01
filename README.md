# MuST-Walk: Multi-Scale Temporal Walks for Efficient Link Prediction

Course project for DAL 518 (Graph Machine Learning), Project 4.3, "Efficient temporal link prediction using temporal walk encodings".

Abhi Raj, Manas Dnyaneshvar Patil, Kamranjeet Singh.

- Report: [docs/MuST-Walk_Report.pdf](docs/MuST-Walk_Report.pdf)
- Slides: [docs/MuST-Walk_Presentation.pptx](docs/MuST-Walk_Presentation.pptx)
- Every run and its result: [results/runs_main_and_large.csv](results/runs_main_and_large.csv)

## Summary

TPNet encodes temporal walks with a random-projection sketch and one decay rate per dataset. We built MuST-Walk, which keeps several decay rates on one shared projection and fuses them. We also audited the data and the pipeline before running, then reran everything on corrected code.

| | |
|---|---|
| Runs | 232 of 232 pre-registered runs finished |
| Datasets | US Legis., Can. Parl., UN Trade, UN Vote, UCI, Enron, tgbl-wiki, tgbl-review, tgbl-uci, tgbl-enron |
| Compute | 534.7 hours, about 45 elapsed hours |
| Tests | 53 unit tests, plus brute-force walk checks |

Main findings:

- Multi-scale beats tuned TPNet on 3 of 6 DyGLib datasets, where the tuned decay erases the previous time step. It is not a general improvement.
- Training on historical negatives gives +9.44 to +25.62 AP on all 6 datasets, 18 of 18 seeds, at a cost on random-negative tests and with mixed TGB results. This idea is not new; see Credits.
- The lazy update reaches a true O(k·dR) per edge and is 24 times faster at 1M nodes with J = 4.
- The same-timestamp leak moves baseline scores substantially. Zhang et al. reported it first.
- On UN Trade and UN Vote a simple direction-aware recency rule beats every model we trained.

## Layout

| Path | Contents |
|---|---|
| `code/tpnet/` | TPNet (commit 7dd0cf4) with our changes. `models/MuSTWalk.py` and `models/LDTPNet.py` are ours. Keeps the upstream MIT licence |
| `code/tgb/` | Our patches and runner for the TGB experiments |
| `verify/` | Tiny-graph traces, brute-force sketch check, data statistics, audit findings |
| `preregistration/` | Decision rules written before each experiment, with dated addenda |
| `results/` | Result tables, per-run CSV files, benchmarks, data checks |
| `analysis/` | Script that built the final tables |
| `docs/` | Report and slides |

## Model names

Run IDs and `results/final_tables.txt` keep short labels. They map to the names used in the report and slides:

| Short label | Name | Meaning |
|---|---|---|
| B | TPNet | Tuned single decay rate, fixed code |
| M | MuST-Walk | Rate ladder, concatenation |
| X | MuST-Cross | MuST-Walk plus cross-scale features |
| E | MuST-Equal | MuST-Walk with width dR/J |
| HB, HM, HX | TPNet-HN, MuST-HN, Cross-HN | The same models trained on historical negatives |
| D | TPNet, no strict rule | TPNet without the strict-time buffer |
| C0 to C5 | Original-code ablations | J = 1; J = 2 (two ladders); J = 4 with concat, average, gate |
| UNI | Uniform ladder | {1e-4, 1e-5, 1e-6, 1e-7} |

## Setup

```bash
pip install -r requirements.txt
```

Datasets are not included. DyGLib datasets come from [Zenodo record 7213796](https://zenodo.org/records/7213796). Unzip into `code/tpnet/DG_data/`, then run `preprocess_data/preprocess_data.py`.

## Run

Baseline TPNet with the leak fix on one seed:

```bash
cd code/tpnet
python train_link_prediction.py --dataset_name USLegis --model_name TPNet --use_random_projection \
  --num_epochs 100 --patience 20 --num_runs 1 --run_start 0 --gpu 0 --prefix demo \
  --rp_strict_time --rp_lazy --rp_lambdas 1e-06
```

Multi-scale (M) uses the ladder and fusion flags:

```bash
python train_link_prediction.py --dataset_name USLegis --model_name TPNet --use_random_projection \
  --num_epochs 100 --patience 20 --num_runs 1 --run_start 0 --gpu 0 --prefix demo_m \
  --rp_strict_time --rp_lazy --rp_lambdas 1e-05 1e-06 1e-07 2.2e-08 0 --rp_fusion concat
```

| Flag | Effect |
|---|---|
| `--rp_lambdas` | One decay rate per scale |
| `--rp_fusion` | `concat`, `avg` or `gate` |
| `--rp_strict_time` | Sketch holds only edges strictly older than the query time |
| `--rp_lazy` | Lazy update, same values |
| `--rp_cross` | Cross-scale features |
| `--train_neg_hist 0.5 --select_hist` | Historical-negative training and selection |
| `--enforce_dim` | Set sketch width, used for equal memory |

The exact flags of every run are in `code/tpnet/runlist_*.txt` and `code/tgb/runlist_tgb.txt`.

## Tests

```bash
cd code/tpnet
python test_mustwalk.py
python test_features.py
```

## TGB

TGB_TPNet has no licence, so its code is not copied here. Clone `github.com/lxd99/TGB_TPNet` at commit 86d84e2, install `py-tgb==2.3.0` and `numba`, copy the files from `code/tgb/` into it, then run `apply_tgb_patch.py` and `apply_tgb_features_patch.py`.

## Notes on the records

- Account names, machine details and one ablation label in logs and pre-registrations were replaced with neutral text, and files were renamed for clarity. No result changed.
- `analysis/final_analysis.py` needs the raw run logs, about 2.6 GB, which are not in this repository. Its output is `results/final_tables.txt` and `results/runs_main_and_large.csv`.
- Earlier experiments on the original code are in `results/original_code_*`. They carry the same-timestamp leak.

## Credits

TPNet: Lu et al., NeurIPS 2024. TGB: Huang et al., 2023. DyGLib: Yu et al., 2023. Historical negatives: Poursafaei et al., 2022. Forward decay, used by the lazy update: Cormode et al., ICDE 2009. The same-timestamp leak was first reported by Zhang et al. (SNAM, arXiv 2610.04869). Similar historical-negative training appears in ENS (Gao et al., 2024), CurNM (Chen et al., 2025) and ATNSF (Jiang et al., 2026). Full references are in the report.

## Licence

Our code is under the MIT licence in `LICENSE`. `code/tpnet/` also keeps the upstream TPNet licence.
