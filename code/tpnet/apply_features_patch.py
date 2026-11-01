"""Third patch, applied after apply_strict_patch.py. Adds opt-in flags, all off by default:

  --rp_lazy          batch cost O(batch) instead of O(nodes); same values
  --rp_cross         inner products between different decay scales
  --rp_gate_norm     LayerNorm on each scale before avg/gate fusion
  --train_neg_hist p replace a fraction p of training negatives with historical negatives
  --select_hist      early stopping on validation with historical negatives

Every edit must match exactly once, otherwise it stops. Safe to run twice.
"""
import sys

MARK = '# [v3 patch]'


def patch(path, edits):
    text = open(path).read()
    if MARK in text:
        print(path, 'already patched')
        return
    for old, new in edits:
        if text.count(old) != 1:
            sys.exit(f'ERROR {path}: found {text.count(old)} times: {old[:70]}')
        text = text.replace(old, new)
    open(path, 'w').write(text + f'\n{MARK}\n')
    print(path, 'patched')


patch('utils/load_configs.py', [(
    "    parser.add_argument('--run_start', type=int, default=0,\n",
    "    parser.add_argument('--rp_lazy', action='store_true', help='lazy sketch update, same values')\n"
    "    parser.add_argument('--rp_cross', action='store_true', help='inner products between decay scales')\n"
    "    parser.add_argument('--rp_gate_norm', action='store_true', help='LayerNorm per scale before avg/gate')\n"
    "    parser.add_argument('--train_neg_hist', type=float, default=0.0, help='fraction of historical training negatives')\n"
    "    parser.add_argument('--select_hist', action='store_true', help='early stopping on historical validation')\n"
    "    parser.add_argument('--run_start', type=int, default=0,\n")])
patch('train_link_prediction.py', [
    ('from models.MuSTWalk import build_random_projection_module, replace_false_negatives',
     'from models.MuSTWalk import build_random_projection_module, replace_false_negatives, HistoricalTrainNegatives, CachedNegativeSampler'),
    ("    val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids,\n"
     "                                               seed=0)\n",
     "    val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids,\n"
     "                                               seed=0)\n"
     "    if args.select_hist:\n"
     "        val_neg_edge_sampler = CachedNegativeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids,\n"
     "                                                     interact_times=full_data.node_interact_times,\n"
     "                                                     last_observed_time=train_data.node_interact_times[-1],\n"
     "                                                     negative_sample_strategy='historical', seed=0)\n"
     "    historical_train_negatives = HistoricalTrainNegatives(train_data) if args.train_neg_hist > 0 else None\n"),
    ("                batch_neg_src_node_ids = batch_src_node_ids\n"
     "                batch_neg_node_interact_times = batch_node_interact_times\n",
     "                batch_neg_src_node_ids = batch_src_node_ids\n"
     "                if historical_train_negatives is not None:\n"
     "                    batch_neg_src_node_ids, batch_neg_dst_node_ids = historical_train_negatives.mix(\n"
     "                        batch_src_node_ids, batch_dst_node_ids, batch_neg_dst_node_ids, batch_node_interact_times,\n"
     "                        args.train_neg_hist)\n"
     "                batch_neg_node_interact_times = batch_node_interact_times\n")])
