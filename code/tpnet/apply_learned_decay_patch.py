"""Fourth patch, applied after apply_features_patch.py. Adds, off by default:

  --rp_learn_decay m  learn the decay rates, 'shared' or 'per_depth', starting from the --rp_lambdas values
  --rp_decay_lr r     learning rate of the log decay rates (default 1e-2, as in D4)

Every edit must match exactly once, otherwise it stops. Safe to run twice.
"""
import sys

MARK = '# [ld patch]'


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


SELECT = "    parser.add_argument('--select_hist', action='store_true', help='early stopping on historical validation')\n"
OPTIMIZER = ("        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate,\n"
             "                                     weight_decay=args.weight_decay)\n")
EPOCH = ("                f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0][\"lr\"]}, train loss: {np.mean(train_losses):.4f}')\n")

patch('utils/load_configs.py', [(SELECT, SELECT +
    "    parser.add_argument('--rp_learn_decay', type=str, default='none', choices=['none', 'shared', 'per_depth'], help='learn the decay rates')\n"
    "    parser.add_argument('--rp_decay_lr', type=float, default=1e-2, help='learning rate of the log decay rates')\n")])
patch('train_link_prediction.py', [
    (OPTIMIZER, OPTIMIZER + "        from models.LDTPNet import add_decay_param_group\n        optimizer = add_decay_param_group(optimizer, model, args)\n"),
    (EPOCH, EPOCH + "            if args.rp_learn_decay != 'none':\n                logger.info(random_projections.decay_summary())\n")])
