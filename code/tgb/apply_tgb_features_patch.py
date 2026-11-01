"""Applied to TGB_TPNet after handoff/tgb/apply_tgb_patch.py. Adds, all off by default:

  --val_fraction f   per-epoch validation on the first f of the validation edges in time order (the sketch stream stays correct);
                     the final evaluation still runs the full validation and test
  --rp_strict_time --rp_lazy --rp_cross --rp_gate_norm --train_neg_hist p
and support for tgbl-uci and tgbl-enron, whose sizes the repo does not know.

Every edit must match exactly once, otherwise it stops. Safe to run twice.
"""
import sys

MARK = '# [tgb v3 patch]'


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


SMALL = "args.dataset_name in ('tgbl-wiki', 'tgbl-review', 'tgbl-uci', 'tgbl-enron')"
INDENT = ' ' * 69

patch('utils/load_configs.py', [
    ('            "tgbl-wiki",\n            "tgbl-review",\n', '            "tgbl-wiki",\n            "tgbl-uci",\n            "tgbl-enron",\n            "tgbl-review",\n'),
    ('    parser.add_argument(\n        "--rp_use_matrix",',
     "    parser.add_argument('--rp_strict_time', action='store_true', help='sketch holds only strictly older edges')\n"
     "    parser.add_argument('--rp_lazy', action='store_true', help='lazy sketch update, same values')\n"
     "    parser.add_argument('--rp_cross', action='store_true', help='inner products between decay scales')\n"
     "    parser.add_argument('--rp_gate_norm', action='store_true', help='LayerNorm per scale before avg/gate')\n"
     "    parser.add_argument('--train_neg_hist', type=float, default=0.0, help='fraction of historical training negatives')\n"
     "    parser.add_argument('--val_fraction', type=float, default=1.0, help='fraction of validation edges used each epoch')\n"
     '    parser.add_argument(\n        "--rp_use_matrix",')])

patch('utils/DataLoader.py', [
    ("    assert num_edges == data_num_edges_map[dataset_name], 'Number of edges are not matched!'\n",
     "    assert dataset_name not in data_num_edges_map or num_edges == data_num_edges_map[dataset_name], 'Number of edges are not matched!'\n"),
    ("    assert num_nodes == data_num_nodes_map[dataset_name], f'Number of nodes are not matched! {num_nodes} vs {data_num_nodes_map[dataset_name]}'\n",
     "    assert dataset_name not in data_num_nodes_map or num_nodes == data_num_nodes_map[dataset_name], f'Number of nodes are not matched! {num_nodes}'\n"
     "    if dataset_name not in data_num_nodes_map:  # tgbl-uci and tgbl-enron leave some ids unused, so size the arrays by the largest id\n"
     "        num_nodes = max(src_node_ids.max(), dst_node_ids.max()) + 1\n"),
    ('    assert src_node_ids.min() == 0 or dst_node_ids.min(\n    ) == 0, "Node index should start from 0!"\n',
     '    assert dataset_name not in data_num_nodes_map or src_node_ids.min() == 0 or dst_node_ids.min() == 0, "Node index should start from 0!"\n')])

patch('models/TPNet.py', [(
    "        if self.not_embedding:\n            node_embeddings = torch.zeros((len(src_node_ids)+len(dst_node_ids),",
    "        if hasattr(self.random_projections, 'advance_to'):\n"
    "            self.random_projections.advance_to(node_interact_times)\n"
    "        if self.not_embedding:\n            node_embeddings = torch.zeros((len(src_node_ids)+len(dst_node_ids),"), (
    # 999 negatives per edge do not fit in GPU memory with four scales. The sketch is not touched here, so the pieces see the same state
    "            node_embeddings = self.embedding_module.compute_node_temporal_embeddings(\n"
    "                node_ids=np.concatenate([src_node_ids, dst_node_ids]),\n"
    "                src_node_ids=np.tile(src_node_ids, 2),\n"
    "                dst_node_ids=np.tile(dst_node_ids, 2),\n"
    "                node_interact_times=np.tile(node_interact_times, 2))\n",
    "            node_ids, pair_src, pair_dst, times = np.concatenate([src_node_ids, dst_node_ids]), np.tile(src_node_ids, 2), np.tile(dst_node_ids, 2), np.tile(node_interact_times, 2)\n"
    "            size = len(node_ids) if torch.is_grad_enabled() else self.eval_chunk\n"
    "            node_embeddings = torch.cat([self.embedding_module.compute_node_temporal_embeddings(\n"
    "                node_ids=node_ids[i:i + size], src_node_ids=pair_src[i:i + size], dst_node_ids=pair_dst[i:i + size],\n"
    "                node_interact_times=times[i:i + size]) for i in range(0, len(node_ids), size)], dim=0)\n"), (
    "class TPNet(torch.nn.Module):\n",
    "class TPNet(torch.nn.Module):\n    eval_chunk = 4000\n")])

patch('utils/evaluate_models_utils.py', [(
    "            if dataset_name == 'tgbl-wiki':\n"
    "                batch_neg_dst_node_ids = [x if len(x)==999 else x+[-1] for x in batch_neg_dst_node_ids]\n",
    "            longest = max(len(x) for x in batch_neg_dst_node_ids)\n"
    "            batch_neg_dst_node_ids = [list(x) + [-1] * (longest - len(x)) for x in batch_neg_dst_node_ids]\n"),
    ('                "y_pred_neg": negative_probabilities.reshape(-1, num_negative_samples_per_node),\n',
     '                "y_pred_neg": negative_probabilities.masked_fill(torch.from_numpy(batch_neg_dst_node_ids == 0).to(negative_probabilities.device), positive_probabilities.min().item() - 1.0).reshape(-1, num_negative_samples_per_node),  # padded slots never outrank\n')])

patch('train_link_prediction.py', [
    ('from models.MuSTWalk import build_random_projection_module', 'from models.MuSTWalk import build_random_projection_module, HistoricalTrainNegatives'),
    ("    if args.dataset_name == \"tgbl-wiki\" or args.dataset_name == 'tgbl-review':\n", f'    if {SMALL}:\n'),
    ("    evaluator = Evaluator(name=args.dataset_name)\n",
     "    epoch_val_idx_data_loader = get_idx_data_loader(\n"
     "        indices_list=list(range(max(1, int(len(val_data.src_node_ids) * args.val_fraction)))),\n"
     "        batch_size=val_idx_data_loader.batch_size, shuffle=False)\n"
     "    historical_train_negatives = HistoricalTrainNegatives(train_data) if args.train_neg_hist > 0 else None\n"
     "    evaluator = Evaluator(name=args.dataset_name)\n"),
    ("                batch_neg_node_interact_times = np.repeat(batch_node_interact_times, args.train_neg_num)\n",
     "                if historical_train_negatives is not None:\n"
     "                    batch_neg_src_node_ids, batch_neg_dst_node_ids = historical_train_negatives.mix(\n"
     "                        batch_src_node_ids, batch_dst_node_ids, batch_neg_dst_node_ids, batch_node_interact_times,\n"
     "                        args.train_neg_hist)\n"
     "                batch_neg_node_interact_times = np.repeat(batch_node_interact_times, args.train_neg_num)\n"),
    (f"model=model, dtype='val',\n{INDENT}eval_metric_name=eval_metric_name,\n{INDENT}neighbor_sampler=full_neighbor_sampler,\n"
     f"{INDENT}evaluate_idx_data_loader=val_idx_data_loader,\n{INDENT}evaluate_neg_edge_sampler=eval_neg_edge_sampler,\n{INDENT}evaluator=evaluator,\n{INDENT}evaluate_data=val_data,\n{INDENT}loss_func=loss_func,\n{INDENT}num_neighbors=args.num_neighbors,\n{INDENT}time_gap=args.time_gap, logger=logger)\n\n            # reload the train memory",
     f"model=model, dtype='val',\n{INDENT}eval_metric_name=eval_metric_name,\n{INDENT}neighbor_sampler=full_neighbor_sampler,\n"
     f"{INDENT}evaluate_idx_data_loader=epoch_val_idx_data_loader,\n{INDENT}evaluate_neg_edge_sampler=eval_neg_edge_sampler,\n{INDENT}evaluator=evaluator,\n{INDENT}evaluate_data=val_data,\n{INDENT}loss_func=loss_func,\n{INDENT}num_neighbors=args.num_neighbors,\n{INDENT}time_gap=args.time_gap, logger=logger)\n\n            # reload the train memory")])
