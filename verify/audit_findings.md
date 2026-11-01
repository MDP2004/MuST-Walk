# Pipeline verification on tiny graphs (30 Oct 2026)

**Attribution, added after this audit:** the same-timestamp leak in TPNet was already found and fixed by Zhang et al. (SNAM,
arXiv 2610.04869, `baselines/TPNet_strict`), using the same buffering idea as our `--rp_strict_time`. Lampert et al. (TMLR 2026) describe
the general batching leak. Our audit confirms it independently; it is not our discovery. Their strict retraining: historical AP
Can. Parl. 72.07, US Legis 75.37, UN Trade 61.97, UN Vote 64.78, EdgeBank 63.84 / 63.22 / 81.32 / 84.89.
Their README also says the released loader sorted the node set, which matches our check that `sorted()` changes nothing against released TPNet.

Method: two hand-checkable graphs (`make_tiny.py`), the real scripts traced call by call (`trace_run.py`, traces in `traces/`),
the sketch checked against brute-force walk enumeration that shares no code with it (`check_sketch.py`), and effect sizes
measured on real data with no training (`real_stats.py`).

## Verified correct
- Sketch equals brute-force temporal walks, relative error about 1e-7, all depths, J = 1, 2, 4, lambda 1e-7 to 0.
- Pairwise feature equals the hand formula exactly. J = 1 through the wrapper is identical to the original module.
- No future edges in the sketch, no neighbor at or after the query time, train neighbors come only from train edges.
- Update happens after prediction. Time splits do not overlap. New-node evaluation starts from the train state, test from the val state.
- Validation and test negatives are the same in every epoch. Checkpoint reload reproduces the test AP exactly (0.8597).
- Same seed gives identical results, a different seed differs. No inf/NaN at lambda = 0 on the real streams.
- Historical negatives exclude every edge of the batch's time interval.

## Bug in our code, fixed
- `collect.py` read the config from `uid.split('_')[1]`, which breaks on any prefixed unit id. Now `[-2]`.

## Issues inherited from the authors' code (not bugs in our patch)
Real-data numbers: US Legis / UN Trade / UN Vote.
1. Same-time information: 96.0 / 99.7 / 99.9 % of test batches hold same-time edges in the sketch, and 96 / 85 / 99 % of the 1-hop
   walk weight comes from them at the tuned lambda. The neighbor sampler excludes these edges. Random-negative and historical AP are both exposed.
2. Random negatives that are true edges at that time: 44.0 / 59.9 / 95.7 %. Applies to evaluation and to training labels. Historical negatives do not have this problem.
3. Edges touching held-out nodes never enter the sketch during training: 31.8 / 27.8 / 20.7 % of pre-validation edges are missing, though the neighbor sampler has them.
4. Raw edge features are not scaled: max 314 / 52,504,548 / 150. UN Trade's first-epoch loss is 4-6, against about 0.53 on the others.
5. Early stopping saves only if AP and AUC both improve, so the kept epoch is not always the best-AP epoch. Our earlier "best epoch" figures used max AP.
6. P is redrawn every epoch, by design. It adds epoch-to-epoch noise and will matter more for equal-memory runs.

## Opt-in fixes (`apply_strict_patch.py`, all off by default)
`--rp_strict_time`, `--clean_train_neg`, `--edge_feat_log`.
Flags off: results identical to before (0.8597 / 0.7157). Strict on: same-time lines 38 -> 0, 0 violations.
On the continuous graph strict equals default exactly, as it must when no timestamps tie.

## Not known yet
- How much these issues change real AP or seed variance. Tiny-graph AP values mean nothing.
- Whether 100/20 closes the UN Trade gap. V1 was stopped by request after a few epochs, so it has no result.
- Job 1 numbers were produced with issues 1-5 present. The paper's use the same authors' code, so comparisons with it stay fair, but neither is free of them.
