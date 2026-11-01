"""
Apply the MuST-Walk / learnable-decay changes to an unmodified clone of https://github.com/lxd99/TGB_TPNet (commit 86d84e2).

Run from the repo root:  python apply_tgb_patch.py      (models/MuSTWalk.py and models/LDTPNet.py must be present)
Safe to run more than once; every edit must match exactly once, otherwise it stops with an error.

Changes:
  1. utils/DataLoader.py  : tgbl-review node count of the CURRENT TGB release (v2: 352,637 nodes; the repo expects
                            651,226 from another release) + a check that node ids are contiguous 0..N-1.
  2. utils/load_configs.py: flags --rp_lambdas --rp_fusion --rp_base_index --rp_learn_decay --rp_decay_lr --run_start
  3. train/evaluate scripts: build the sketch via models/MuSTWalk.build_random_projection_module (returns TPNet's own
                            module when no new flag is given), seeds run_start.., decay-rate optimizer group + logging.
"""
import os
import re
import sys

MARK = '# [MuST-Walk patch]'


def patch_file(path, edits):
    text = open(path).read()
    if MARK in text:
        print(f'{path}: already patched, skipped')
        return
    new = text
    for pattern, repl, is_regex in edits:
        if is_regex:
            n = len(re.findall(pattern, new, flags=re.DOTALL))
            if n != 1:
                sys.exit(f'ERROR {path}: pattern matched {n} times (expected 1): {pattern[:70]}')
            new = re.sub(pattern, lambda m: repl, new, count=1, flags=re.DOTALL)
        else:
            if new.count(pattern) != 1:
                sys.exit(f'ERROR {path}: text found {new.count(pattern)} times (expected 1): {pattern[:70]}')
            new = new.replace(pattern, repl, 1)
    open(path, 'w').write(new + f'\n{MARK}\n')
    print(f'{path}: patched')


for f in ('models/MuSTWalk.py', 'models/LDTPNet.py'):
    if not os.path.exists(f):
        sys.exit(f'ERROR: {f} is missing')

patch_file('utils/DataLoader.py', [
    ('    "tgbl-review":651226,\n',
     '    "tgbl-review":352637,  # current TGB release (v2); the original repo expects 651226 from another release\n', False),
    ("    assert num_nodes == data_num_nodes_map[dataset_name], f'Number of nodes are not matched! {num_nodes} vs {data_num_nodes_map[dataset_name]}'\n",
     "    assert num_nodes == data_num_nodes_map[dataset_name], f'Number of nodes are not matched! {num_nodes} vs {data_num_nodes_map[dataset_name]}'\n"
     "    assert max(src_node_ids.max(), dst_node_ids.max()) + 1 == num_nodes, 'node ids are not contiguous 0..N-1'\n", False)])

patch_file('utils/load_configs.py', [(
    '    parser.add_argument(\n        "--rp_use_matrix",',
    "    parser.add_argument('--rp_lambdas', type=float, nargs='*', default=None,\n"
    "                        help='MuST-Walk: list of decay rates (one sketch per rate, shared projection P)')\n"
    "    parser.add_argument('--rp_fusion', type=str, default='concat', choices=['concat', 'avg', 'gate'],\n"
    "                        help='MuST-Walk: how per-scale pairwise features are fused')\n"
    "    parser.add_argument('--rp_base_index', type=int, default=0,\n"
    "                        help='MuST-Walk: index in --rp_lambdas of the base scale for avg/gate fusion')\n"
    "    parser.add_argument('--rp_learn_decay', type=str, default='none', choices=['none', 'shared', 'per_depth'],\n"
    "                        help='learn the decay rate(s) with exact streaming gradients (models/LDTPNet.py)')\n"
    "    parser.add_argument('--rp_decay_lr', type=float, default=1e-2,\n"
    "                        help='learning rate of the log decay rates when --rp_learn_decay is used')\n"
    "    parser.add_argument('--run_start', type=int, default=0,\n"
    "                        help='first run index = first seed (runs use seeds run_start .. run_start+num_runs-1)')\n"
    '    parser.add_argument(\n        "--rp_use_matrix",', False)])

CONSTRUCT = r"random_projections = RandomProjectionModule\(\s*node_num=node_raw_features\.shape\[0\],.*?enforce_dim=args\.enforce_dim,?\s*\)"
FACTORY = ("random_projections = build_random_projection_module(args, node_num=node_raw_features.shape[0], "
           "edge_num=edge_raw_features.shape[0], beginning_time=train_data.node_interact_times[0])")
LOOP = ('for run in range(args.num_runs):', 'for run in range(args.run_start, args.run_start + args.num_runs):', False)

patch_file('train_link_prediction.py', [
    ('from models.TPNet import TPNet, RandomProjectionModule',
     'from models.TPNet import TPNet, RandomProjectionModule\nfrom models.MuSTWalk import build_random_projection_module', False),
    (CONSTRUCT, FACTORY, True), LOOP,
    ('        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate,\n'
     '                                     weight_decay=args.weight_decay)\n',
     '        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate,\n'
     '                                     weight_decay=args.weight_decay)\n'
     '        from models.LDTPNet import add_decay_param_group\n'
     '        optimizer = add_decay_param_group(optimizer, model, args)\n', False),
    ("                f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0][\"lr\"]}, train loss: {np.mean(train_losses):.4f}')\n",
     "                f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0][\"lr\"]}, train loss: {np.mean(train_losses):.4f}')\n"
     "            if hasattr(random_projections, 'decay_summary'):\n"
     "                logger.info(random_projections.decay_summary())\n", False)])
patch_file('evaluate_link_prediction.py', [
    ('from models.TPNet import RandomProjectionModule, TPNet',
     'from models.TPNet import RandomProjectionModule, TPNet\nfrom models.MuSTWalk import build_random_projection_module', False),
    (CONSTRUCT, FACTORY, True), LOOP])
print('done')
