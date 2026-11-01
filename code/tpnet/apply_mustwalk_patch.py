"""
Apply the MuST-Walk changes to an unmodified clone of https://github.com/lxd99/TPNet.

Run from the repo root:  python apply_mustwalk_patch.py
Safe to run more than once. Every edit must match exactly once; otherwise it stops with an error
and changes nothing in that file.

Changes:
  1. utils/DataLoader.py   : Python >= 3.11 fix (random.sample needs a sequence, not a set).
  2. utils/load_configs.py : new flags --rp_lambdas, --rp_fusion, --rp_base_index.
  3. train_link_prediction.py, evaluate_link_prediction.py :
       build the sketch through models/MuSTWalk.build_random_projection_module, which returns
       TPNet's original module when --rp_lambdas is not given (default behaviour unchanged).
models/MuSTWalk.py must already be present.
"""
import os
import re
import sys

MARK = '# [MuST-Walk patch]'


def patch_file(path, edits):
    with open(path) as f:
        text = f.read()
    if MARK in text:
        print(f'{path}: already patched, skipped')
        return
    new = text
    for pattern, repl, is_regex in edits:
        if is_regex:
            matches = re.findall(pattern, new, flags=re.DOTALL)
            if len(matches) != 1:
                sys.exit(f'ERROR {path}: pattern matched {len(matches)} times (expected 1): {pattern[:60]}')
            new = re.sub(pattern, repl, new, count=1, flags=re.DOTALL)
        else:
            if new.count(pattern) != 1:
                sys.exit(f'ERROR {path}: text found {new.count(pattern)} times (expected 1): {pattern[:60]}')
            new = new.replace(pattern, repl, 1)
    with open(path, 'w') as f:
        f.write(new + f'\n{MARK}\n')
    print(f'{path}: patched')


if not os.path.exists('models/MuSTWalk.py'):
    sys.exit('ERROR: models/MuSTWalk.py is missing; create it first.')

# 1. Python 3.11+ fix
with open('utils/DataLoader.py') as f:
    dl = f.read()
if 'random.sample(test_node_set, ' in dl:
    patch_file('utils/DataLoader.py', [('random.sample(test_node_set, ', 'random.sample(sorted(test_node_set), ', False)])
else:
    print('utils/DataLoader.py: random.sample fix already present, skipped')

# 2. new flags
patch_file('utils/load_configs.py', [(
    "    parser.add_argument('--rp_use_matrix', action='store_true',",
    "    parser.add_argument('--rp_lambdas', type=float, nargs='*', default=None,\n"
    "                        help='MuST-Walk: list of decay rates (one sketch per rate, shared projection P)')\n"
    "    parser.add_argument('--rp_fusion', type=str, default='concat', choices=['concat', 'avg', 'gate'],\n"
    "                        help='MuST-Walk: how per-scale pairwise features are fused')\n"
    "    parser.add_argument('--rp_base_index', type=int, default=0,\n"
    "                        help='MuST-Walk: index in --rp_lambdas of the base scale for avg/gate fusion')\n"
    "    parser.add_argument('--run_start', type=int, default=0,\n"
    "                        help='first run index = first seed (runs use seeds run_start .. run_start+num_runs-1)')\n"
    "    parser.add_argument('--rp_use_matrix', action='store_true',", False)])

# 3. build the module through the factory in both scripts
CONSTRUCT = r"random_projections = RandomProjectionModule\(node_num=node_raw_features\.shape\[0\],.*?enforce_dim=args\.enforce_dim\)"
FACTORY = ("random_projections = build_random_projection_module(args, node_num=node_raw_features.shape[0], "
           "edge_num=edge_raw_features.shape[0], beginning_time=train_data.node_interact_times[0])")
LOOP = ('for run in range(args.num_runs):', 'for run in range(args.run_start, args.run_start + args.num_runs):', False)
patch_file('train_link_prediction.py', [
    ('from models.TPNet import TPNet, RandomProjectionModule',
     'from models.TPNet import TPNet, RandomProjectionModule\nfrom models.MuSTWalk import build_random_projection_module', False),
    (CONSTRUCT, FACTORY, True), LOOP])
patch_file('evaluate_link_prediction.py', [
    ('from models.TPNet import RandomProjectionModule, TPNet',
     'from models.TPNet import RandomProjectionModule, TPNet\nfrom models.MuSTWalk import build_random_projection_module', False),
    (CONSTRUCT, FACTORY, True), LOOP])
print('done')
