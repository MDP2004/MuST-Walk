"""Run the REAL train/evaluate script with tracing around it. Nothing in the scripts is replaced, only wrapped.

Usage, from the sandbox root:  python trace_run.py <trace_file> <script.py> <script args...>
Writes one line per batch to <trace_file>: the phase, what is being predicted, what the sketch holds at that moment,
which negatives were drawn, and which neighbors were read. Violations are written as lines starting with VIOLATION.
"""
import os
import runpy
import sys

import numpy as np

sys.path.insert(0, '.')
trace_path, script = sys.argv[1], sys.argv[2]
sys.argv = [script] + sys.argv[3:]
out = open(trace_path, 'w', buffering=1, encoding='utf-8')
YEAR = 86400 * 365


def log(s):
    out.write(s + '\n')


S = dict(phase='setup', batch=0, epoch=0, sketch=[], vio=0, train_ids=None, backups={}, ckpt=None, neigh_calls=0, reads=0, val_end=None)


def violation(msg):
    S['vio'] += 1
    log('VIOLATION ' + msg)


def fmt_t(t):
    return f'y{t / YEAR:g}' if t >= YEAR or t == 0 else f'{t:g}'


import utils.DataLoader as DL  # noqa: E402

orig_get = DL.get_link_prediction_data


def traced_get(*a, **k):
    res = orig_get(*a, **k)
    _, _, full, train, val, test, nn_val, nn_test = res
    S['train_ids'] = set(train.edge_ids.tolist())
    u_, v_, t_ = full.src_node_ids.tolist(), full.dst_node_ids.tolist(), full.node_interact_times.tolist()
    S['true'] = set(zip(u_, v_, t_)) | set(zip(v_, u_, t_))
    log('=== DATA: edge id, src, dst, time, split')
    split = {}
    for name, d in (('train', train), ('val', val), ('test', test)):
        for e in d.edge_ids.tolist():
            split[e] = name
    for e, u, v, t in zip(full.edge_ids, full.src_node_ids, full.dst_node_ids, full.node_interact_times):
        log(f'  e{e:<3d} {u:>2d}-{v:<2d} {fmt_t(t):>6s} {split.get(int(e), "HELD-OUT-FROM-TRAIN")}')
    log(f'sizes: full {full.num_interactions} train {train.num_interactions} val {val.num_interactions} test {test.num_interactions} '
        f'new-node val {nn_val.num_interactions} new-node test {nn_test.num_interactions}')
    if set(train.edge_ids.tolist()) & (set(val.edge_ids.tolist()) | set(test.edge_ids.tolist())):
        violation('train shares edges with val/test')
    if len(train.node_interact_times) and len(val.node_interact_times) and train.node_interact_times.max() >= val.node_interact_times.min():
        violation('train times reach into val')
    if len(val.node_interact_times) and len(test.node_interact_times) and val.node_interact_times.max() >= test.node_interact_times.min():
        violation('val times reach into test')
    return res


DL.get_link_prediction_data = traced_get

import utils.utils as U  # noqa: E402

orig_sample = U.NegativeEdgeSampler.sample


def traced_sample(self, size, batch_src_node_ids=None, batch_dst_node_ids=None, current_batch_start_time=0.0, current_batch_end_time=0.0):
    a, b = orig_sample(self, size, batch_src_node_ids, batch_dst_node_ids, current_batch_start_time, current_batch_end_time)
    S['last_neg'] = list(zip(a.tolist(), b.tolist()))
    return a, b


U.NegativeEdgeSampler.sample = traced_sample
orig_neigh = U.NeighborSampler.get_historical_neighbors


def traced_neigh(self, node_ids, node_interact_times, num_neighbors=20):
    ids, eids, times = orig_neigh(self, node_ids, node_interact_times, num_neighbors)
    S['neigh_calls'] += 1
    mask = eids > 0
    if (times[mask] >= np.broadcast_to(node_interact_times[:, None], times.shape)[mask]).any():
        violation(f'{S["phase"]} b{S["batch"]}: a neighbor is not strictly earlier than the query time')
    if S['phase'] == 'train' and not set(eids[mask].tolist()) <= S['train_ids']:
        violation(f'train b{S["batch"]}: neighbor edge outside the training set')
    return ids, eids, times


U.NeighborSampler.get_historical_neighbors = traced_neigh

from models.MuSTWalk import MultiScaleRandomProjectionModule as MS  # noqa: E402

orig_update, orig_pair, orig_apply = MS.update, MS.get_pair_wise_feature, MS._apply_update
orig_reset, orig_backup, orig_reload = MS.reset_random_projections, MS.backup_random_projections, MS.reload_random_projections


def traced_apply(self, src_node_ids, dst_node_ids, node_interact_times):
    S['sketch'] += list(zip(src_node_ids.tolist(), dst_node_ids.tolist(), node_interact_times.tolist()))
    orig_apply(self, src_node_ids, dst_node_ids, node_interact_times)


def traced_update(self, src_node_ids, dst_node_ids, node_interact_times):
    orig_update(self, src_node_ids, dst_node_ids, node_interact_times)
    S['batch'] += 1


def traced_pair(self, src_node_ids, dst_node_ids):
    S['reads'] += 1
    return orig_pair(self, src_node_ids, dst_node_ids)


def traced_reset(self):
    S['sketch'] = []
    orig_reset(self)
    t = [float(s.now_time) for s in self.scales]
    p0 = self.scales[0].random_projections[0]
    log(f'RESET sketch: now_time {t}, walk layers all zero: {all(float(s.random_projections[i].abs().sum()) == 0 for s in self.scales for i in range(1, s.num_layer + 1))}, '
        f'P checksum {float(p0.sum()):.4f}, P shared by all scales: {all(s.random_projections[0] is p0 for s in self.scales)}')


def traced_backup(self):
    b = orig_backup(self)
    S['backups'][id(b)] = list(S['sketch'])
    log(f'BACKUP sketch: {len(S["sketch"])} edges, now {fmt_t(float(self.scales[0].now_time))}')
    return b


def traced_reload(self, b):
    orig_reload(self, b)
    S['sketch'] = list(S['backups'][id(b)])
    log(f'RELOAD sketch: {len(S["sketch"])} edges, now {fmt_t(float(self.scales[0].now_time))}')


MS.update, MS.get_pair_wise_feature, MS._apply_update = traced_update, traced_pair, traced_apply
MS.reset_random_projections, MS.backup_random_projections, MS.reload_random_projections = traced_reset, traced_backup, traced_reload

from models import TPNet as TP  # noqa: E402

orig_embed = TP.TPNet.compute_src_dst_node_temporal_embeddings


def traced_embed(self, src_node_ids, dst_node_ids, node_interact_times):
    if S.get('pos_pending', True):
        S['pos_pending'] = False
        t = node_interact_times
        if hasattr(self.random_projections, 'advance_to'):
            self.random_projections.advance_to(t)
        sk = S['sketch']
        fut = sum(1 for e in sk if e[2] > t[0])
        same = sum(1 for e in sk if e[2] == t[0])
        rp = self.random_projections
        log(f'  {S["phase"]} b{S["batch"]:02d} predict {list(zip(src_node_ids.tolist(), dst_node_ids.tolist()))} at {fmt_t(t[0])}..{fmt_t(t[-1])} | '
            f'sketch: {len(sk)} edges, now {fmt_t(float(rp.scales[0].now_time))}, SAME-TIME {same}, FUTURE {fut}')
        if fut:
            violation(f'{S["phase"]} b{S["batch"]}: sketch holds {fut} edges later than the batch')
        pos = set(zip(src_node_ids.tolist(), dst_node_ids.tolist(), t.tolist()))
        if pos & set(sk):
            violation(f'{S["phase"]} b{S["batch"]}: the sketch already contains the edge being predicted')
    else:
        S['pos_pending'] = True
        pairs = list(zip(src_node_ids.tolist(), dst_node_ids.tolist()))
        false = sum((a, b, c) in S['true'] for (a, b), c in zip(pairs, node_interact_times.tolist()))
        loops = sum(a == b for a, b in pairs)
        S['neg_total'] = S.get('neg_total', 0) + len(pairs)
        S['neg_false'] = S.get('neg_false', 0) + false
        S['neg_loop'] = S.get('neg_loop', 0) + loops
        log(f'           negatives {pairs} | true edges {false}, self-loops {loops}')
    return orig_embed(self, src_node_ids, dst_node_ids, node_interact_times)


TP.TPNet.compute_src_dst_node_temporal_embeddings = traced_embed

import evaluate_models_utils as E  # noqa: E402

orig_eval = E.evaluate_model_link_prediction
PHASE = {0: 'val', 1: 'newnode-val', 2: 'test', 3: 'newnode-test'}


def traced_eval(*a, **k):
    S['phase'] = PHASE[k['evaluate_neg_edge_sampler'].seed]
    S['batch'] = 0
    log(f'--- {S["phase"]} starts: {k["evaluate_data"].num_interactions} edges, negative strategy {k["evaluate_neg_edge_sampler"].negative_sample_strategy}, '
        f'sampler {"full graph" if k["neighbor_sampler"] is not None else None}')
    res = orig_eval(*a, **k)
    log(f'--- {S["phase"]} ends: mean AP {np.mean([m["average_precision"] for m in res[1]]):.4f}, per batch {[round(m["average_precision"], 3) for m in res[1]]}')
    S['phase'] = 'train'
    return res


E.evaluate_model_link_prediction = traced_eval

from utils.EarlyStopping import EarlyStopping as ES  # noqa: E402

orig_step, orig_save, orig_load = ES.step, ES.save_checkpoint, ES.load_checkpoint


def traced_step(self, metrics, model, args):
    before = dict(self.best_metrics)
    r = orig_step(self, metrics, model, args)
    log(f'EARLY-STOP step: this epoch {[(m[0][:3], round(float(m[1]), 4)) for m in metrics]} best before {[(k[:3], round(float(v), 4)) for k, v in before.items()]} -> counter {self.counter}/{self.patience}')
    return r


def traced_save(self, model, args):
    S['ckpt'] = list(S['sketch'])
    rp = model[0].random_projections
    log(f'CHECKPOINT saved: sketch {len(S["sketch"])} edges, now {fmt_t(float(rp.scales[0].now_time))}')
    orig_save(self, model, args)


def traced_load(self, model, map_location=None):
    orig_load(self, model, map_location)
    rp = model[0].random_projections
    if S['ckpt'] is not None:
        S['sketch'] = list(S['ckpt'])
    log(f'CHECKPOINT loaded: now {fmt_t(float(rp.scales[0].now_time))}, P checksum {float(rp.scales[0].random_projections[0].sum()):.4f}, '
        f'layer-1 checksum {float(rp.scales[0].random_projections[1].sum()):.4f}')


ES.step, ES.save_checkpoint, ES.load_checkpoint = traced_step, traced_save, traced_load

S['phase'] = 'train'
if 'evaluate' in script:
    S['ckpt'] = None
    S['sketch'] = [('?', '?', float('-inf'))]
try:
    runpy.run_path(script, run_name='__main__')
except SystemExit:
    pass
log(f'negatives seen {S.get("neg_total", 0)}, true edges {S.get("neg_false", 0)}, self-loops {S.get("neg_loop", 0)}')
log(f'=== END: {S["vio"]} violations, {S["neigh_calls"]} neighbor calls, {S["reads"]} pairwise-feature reads')
