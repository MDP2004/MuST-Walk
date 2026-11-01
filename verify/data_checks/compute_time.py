"""Total compute time from the raw run logs: unit minutes, by block, by hardware, and calendar span."""
import csv, datetime as dt, glob, os, re, sys, collections
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(r'PATH')
done = {}
first, last = {}, {}
stamp = re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] (.*)')
logs = glob.glob('kaggle/**/runner.log', recursive=True) + glob.glob('out/**/runner.log', recursive=True) + glob.glob('cluster/**/runner.log', recursive=True) + glob.glob('kaggle/**/lane*.log', recursive=True) + glob.glob('cluster/job3/results/lane*.log')
for f in logs:
    for line in open(f, errors='replace'):
        m = stamp.match(line)
        if not m:
            continue
        t = dt.datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
        d = re.match(r'done\s+(\S+) in ([0-9.]+) min', m.group(2))
        if d:
            u, mins = d.group(1), float(d.group(2))
            if u not in done or mins > done[u][0]:
                done[u] = (mins, f)
            block = u.split('_')[0]
            first[block] = min(first.get(block, t), t)
            last[block] = max(last.get(block, t), t)
print('logs', len(logs), 'units with times', len(done))
by = collections.defaultdict(float)
cnt = collections.Counter()
for u, (m, f) in done.items():
    k = u.split('_')[0]
    by[k] += m
    cnt[k] += 1
for k in sorted(by):
    print(f'{k:8} units {cnt[k]:4} hours {by[k] / 60:8.1f} first {first.get(k)} last {last.get(k)}')
print('total hours in these logs', round(sum(by.values()) / 60, 1))
units = list(csv.DictReader(open('analysis/units.csv')))
missing = [r['unit'] for r in units if r['unit'] not in done]
print('units.csv rows without a time here', len(missing), missing[:12])
print('sample tgb results header:')
for f in glob.glob('kaggle/tgb/lanes/*_out/out/results.csv')[:2] + ['cluster/job3/results/results.csv']:
    print(f, open(f).readline().strip())
