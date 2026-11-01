"""Second patch, applied after apply_mustwalk_patch.py. Adds three opt-in flags, all off by default:

  --rp_strict_time   the sketch holds only edges strictly older than the query time, like the neighbor sampler
  --clean_train_neg  redraw training negatives that are self-loops or true edges at that time
  --edge_feat_log    sign(x) * log(1 + |x|) on edge features

Every edit must match exactly once, otherwise it stops. Safe to run twice.
"""
import sys

MARK = '# [strict patch]'


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


LOAD = "logger=logger, convert_time=(args.use_random_projection or args.model_name == 'PINT'))\n"
LOG = LOAD + ("    if args.edge_feat_log:\n"
              "        edge_raw_features = np.sign(edge_raw_features) * np.log1p(np.abs(edge_raw_features))\n")

patch('models/TPNet.py', [(
    "        node_embeddings = self.embedding_module.compute_node_temporal_embeddings(\n            node_ids=np.concatenate([src_node_ids, dst_node_ids]),\n",
    "        if hasattr(self.random_projections, 'advance_to'):\n"
    "            self.random_projections.advance_to(node_interact_times)\n"
    "        node_embeddings = self.embedding_module.compute_node_temporal_embeddings(\n            node_ids=np.concatenate([src_node_ids, dst_node_ids]),\n")])
patch('utils/load_configs.py', [(
    "    parser.add_argument('--run_start', type=int, default=0,\n",
    "    parser.add_argument('--rp_strict_time', action='store_true', help='sketch holds only edges strictly older than the query time')\n"
    "    parser.add_argument('--clean_train_neg', action='store_true', help='redraw false training negatives')\n"
    "    parser.add_argument('--edge_feat_log', action='store_true', help='log-scale edge features')\n"
    "    parser.add_argument('--run_start', type=int, default=0,\n")])
patch('train_link_prediction.py', [
    (LOAD, LOG),
    ('from models.MuSTWalk import build_random_projection_module', 'from models.MuSTWalk import build_random_projection_module, replace_false_negatives'),
    ("                _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))\n",
     "                _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))\n"
     "                if args.clean_train_neg:\n"
     "                    batch_neg_dst_node_ids = replace_false_negatives(batch_src_node_ids, batch_neg_dst_node_ids,\n"
     "                                                                     batch_node_interact_times, train_data)\n")])
patch('evaluate_link_prediction.py', [(LOAD, LOG)])
