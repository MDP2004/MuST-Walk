"""
Run the units of a run list in parallel, independently and resumably.

A unit = one (dataset, configuration, seed):
    train_link_prediction.py  (1 seed)  ->  evaluate historical  ->  evaluate inductive
A unit is marked done (state/done/<unit>.ok) only after all three steps succeed.
Done units are skipped on the next job; failed units are retried at most MAX_ATTEMPTS times in total.
No new unit is started after --deadline (unix time), so the job can end cleanly before Slurm's limit.

Run-list line format (one unit per line, '#' comments allowed):
    unit_id | dataset | seed | flags shared by train and eval
Usage (from the TPNet repo root):
    python runner.py --runlist runlist_job1.txt --workers 8 --gpu 0 --deadline <unix_time> --out <dir>
"""
import argparse
import os
import subprocess
import sys
import time

MAX_ATTEMPTS = 2
COMMON = '--model_name TPNet --use_random_projection --num_epochs {epochs} --patience {patience} --num_runs 1'

p = argparse.ArgumentParser()
p.add_argument('--runlist', required=True)
p.add_argument('--workers', type=int, default=8)
p.add_argument('--gpu', type=int, default=0)
p.add_argument('--epochs', type=int, default=30)
p.add_argument('--patience', type=int, default=5)
p.add_argument('--deadline', type=float, default=float('inf'))
p.add_argument('--out', required=True, help='directory for unit logs and state')
p.add_argument('--python', default=sys.executable)
args = p.parse_args()

state_done = os.path.join(args.out, 'state', 'done')
state_fail = os.path.join(args.out, 'state', 'failed')
unit_logs = os.path.join(args.out, 'unit_logs')
for d in (state_done, state_fail, unit_logs):
    os.makedirs(d, exist_ok=True)
runner_log = open(os.path.join(args.out, 'runner.log'), 'a', buffering=1)


def mem_available_gb():
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return float('nan')


def log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    runner_log.write(line + '\n')


units = []
with open(args.runlist) as f:
    for raw in f:
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        unit_id, dataset, seed, flags = [x.strip() for x in line.split('|')]
        units.append((unit_id, dataset, int(seed), flags))
ids = [u[0] for u in units]
assert len(ids) == len(set(ids)), 'duplicate unit ids in run list'


def attempts(unit_id):
    path = os.path.join(state_fail, unit_id + '.count')
    return int(open(path).read()) if os.path.exists(path) else 0


todo = [u for u in units if not os.path.exists(os.path.join(state_done, u[0] + '.ok'))
        and attempts(u[0]) < MAX_ATTEMPTS]
log(f'run list: {len(units)} units, {len(units) - len(todo)} done or given up, {len(todo)} to run, '
    f'{args.workers} workers, deadline {time.strftime("%H:%M:%S", time.localtime(args.deadline)) if args.deadline < float("inf") else "none"}')

env = dict(os.environ, WANDB_MODE='disabled', CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1')


def steps(unit):
    unit_id, dataset, seed, flags = unit
    base = (f'--dataset_name {dataset} {COMMON.format(epochs=args.epochs, patience=args.patience)} --run_start {seed} '
            f'--gpu {args.gpu} --prefix {unit_id} {flags}').split()
    return [('train', [args.python, 'train_link_prediction.py'] + base),
            ('historical', [args.python, 'evaluate_link_prediction.py'] + base + ['--negative_sample_strategy', 'historical']),
            ('inductive', [args.python, 'evaluate_link_prediction.py'] + base + ['--negative_sample_strategy', 'inductive'])]


# each running unit: [unit, step_index, Popen, start_time, logfile]
running = []
queue = list(todo)
n_ok = n_fail = 0


def start_step(entry):
    unit, k = entry[0], entry[1]
    name, cmd = steps(unit)[k]
    logf = open(os.path.join(unit_logs, f'{unit[0]}_{name}.txt'), 'w')
    entry[2] = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    entry[4] = logf


while queue or running:
    while queue and len(running) < args.workers:
        if time.time() > args.deadline:
            log(f'deadline reached: not starting {len(queue)} remaining units (they will run in the next job)')
            queue = []
            break
        unit = queue.pop(0)
        entry = [unit, 0, None, time.time(), None]
        start_step(entry)
        running.append(entry)
        log(f'start {unit[0]}  (running {len(running)}, node RAM available {mem_available_gb():.1f} GB)')
    time.sleep(5)
    for entry in list(running):
        unit, k, proc, t0, logf = entry
        rc = proc.poll()
        if rc is None:
            continue
        logf.close()
        name = steps(unit)[k][0]
        if rc != 0:
            count = attempts(unit[0]) + 1
            with open(os.path.join(state_fail, unit[0] + '.count'), 'w') as f:
                f.write(str(count))
            hint = ' (killed by the system: usually out of memory -> lower --workers)' if rc in (-9, 137) else ''
            log(f'FAILED {unit[0]} at step {name} (exit {rc}{hint}, attempt {count}/{MAX_ATTEMPTS}); see unit_logs/{unit[0]}_{name}.txt')
            running.remove(entry)
            n_fail += 1
            continue
        if k + 1 < len(steps(unit)):
            entry[1] = k + 1
            start_step(entry)
            continue
        open(os.path.join(state_done, unit[0] + '.ok'), 'w').write(f'{time.time() - t0:.0f}\n')
        log(f'done  {unit[0]} in {(time.time() - t0) / 60:.1f} min')
        running.remove(entry)
        n_ok += 1

left = len([u for u in units if not os.path.exists(os.path.join(state_done, u[0] + '.ok'))])
log(f'runner finished: {n_ok} units done now, {n_fail} failed now, {left} of {len(units)} still not done')
