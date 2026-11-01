"""Run TGB units one after another on one GPU, resumably.   python run_tgb.py --runlist r.txt --gpu 0 --deadline <unix> --unit_limit <s> --out <dir>

A unit = train_link_prediction.py, which ends with the full validation and test. If it hits the time limit, evaluate_link_prediction.py scores the
best epoch saved so far and the unit is marked truncated. Run-list line: unit_id | dataset | seed | flags
"""
import argparse
import os
import re
import subprocess
import sys
import time

COMMON = '--model_name TPNet --use_random_projection --load_best_configs --num_epochs {epochs} --patience 5 --num_runs 1'

p = argparse.ArgumentParser()
p.add_argument('--runlist', required=True)
p.add_argument('--gpu', type=int, default=0)
p.add_argument('--epochs', type=int, default=30)
p.add_argument('--deadline', type=float, required=True, help='unix time: no unit starts after it')
p.add_argument('--hard_stop', type=float, required=True, help='unix time: every process must have ended by then')
p.add_argument('--unit_limit', type=int, default=36000, help='seconds of training per unit')
p.add_argument('--reserve', type=int, default=5400, help='seconds kept for the evaluation of a truncated unit')
p.add_argument('--wrapper', default='', help='script placed before the train/evaluate script, for the fake-dataset rehearsal')
p.add_argument('--out', required=True)
args = p.parse_args()
WRAP = [sys.executable] + ([args.wrapper] if args.wrapper else [])
os.makedirs(f'{args.out}/logs', exist_ok=True)
env = dict(os.environ, WANDB_MODE='disabled', CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1')


def log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    open(f'{args.out}/runner_gpu{args.gpu}.log', 'a').write(line + '\n')


def run(cmd, path, limit):
    with open(path, 'w') as f:
        try:
            return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, timeout=limit).returncode
        except subprocess.TimeoutExpired:
            return 'timeout'


def result(path):
    text = open(path, errors='replace').read()
    test = re.findall(r'test mrr[,:]?\s*([0-9.]+)', text)
    val = re.findall(r'validate mrr[,:]?\s*([0-9.]+)', text)
    return (float(val[-1]) if val else None), (float(test[-1]) if test else None), len(re.findall(r'Epoch: \d+, learning rate', text))


for raw in open(args.runlist):
    if not raw.strip():
        continue
    uid, ds, seed, flags = [x.strip() for x in raw.split('|')]
    if os.path.exists(f'{args.out}/{uid}.ok'):
        continue
    if time.time() > args.deadline:
        log(f'deadline: not starting {uid}')
        continue
    base = f'--dataset_name {ds} {COMMON.format(epochs=args.epochs)} --run_start {seed} --gpu {args.gpu} --prefix {uid} {flags}'.split()
    left = args.hard_stop - time.time()
    limit = min(args.unit_limit, left - args.reserve)
    if limit < 60:
        log(f'not enough time left for {uid}')
        continue
    log(f'start {uid} (training limit {limit / 3600:.1f} h)')
    t0 = time.time()
    train_log = f'{args.out}/logs/{uid}_train.txt'
    rc = run(WRAP + ['train_link_prediction.py'] + base, train_log, limit)
    val, test, epochs = result(train_log)
    state = 'complete'
    if rc not in (0, 'timeout'):
        log(f'FAILED {uid}: exit {rc}, see logs/{uid}_train.txt')
        print(''.join(open(train_log, errors='replace').readlines()[-12:]), flush=True)
        continue
    if test is None:
        eval_log = f'{args.out}/logs/{uid}_eval.txt'
        rc = run(WRAP + ['evaluate_link_prediction.py'] + base, eval_log, max(600, args.hard_stop - time.time() - 60))
        val, test, _ = result(eval_log)
        state = 'truncated'
        if rc != 0 or test is None:
            log(f'FAILED {uid}: evaluation exit {rc}, see logs/{uid}_eval.txt')
            continue
    open(f'{args.out}/{uid}.ok', 'w').write(f'{time.time() - t0:.0f}\n')
    with open(f'{args.out}/results.csv', 'a') as f:
        f.write(f'{uid},{ds},{seed},{val},{test},{epochs},{state},{time.time() - t0:.0f}\n')
    log(f'done  {uid} {state} val {val} test {test} epochs {epochs} in {(time.time() - t0) / 60:.1f} min')
