"""
Multi-scale temporal walk sketches for TPNet (MuST-Walk).

MultiScaleRandomProjectionModule keeps J copies of TPNet's RandomProjectionModule, one per
decay rate lambda_j, that all share ONE random projection P (layer 0). Each scale is exactly
TPNet's update rule with its own lambda, so TPNet's exactness theorem holds per scale.

It exposes the same interface the TPNet training/evaluation scripts use:
    pair_wise_feature_dim, update(), get_pair_wise_feature(), reset_random_projections(),
    backup_random_projections(), reload_random_projections()

Fusion modes for the per-scale pairwise features f_1..f_J (each of size D = (2k+2)^2):
    'concat' : [f_1, ..., f_J]                       (dimension J*D)   -- concatenation
    'avg'    : f_base + beta * mean_j f_j             (dimension D)     -- fixed-average ablation
    'gate'   : f_base + beta * sum_j alpha_j f_j      (dimension D)     -- gated residual
               alpha = softmax(MLP([f_1..f_J])), beta initialised to 0, so training starts at TPNet
               with the base scale (index rp_base_index, normally the dataset's tuned lambda).

Options, all off by default (off = the job-1 code path):
    lazy        : LazyRandomProjectionModule, same values, a batch costs O(batch) instead of O(nodes)
    cross       : with 'concat', add inner products between DIFFERENT scales, u at lambda_a with v at lambda_b.
                  They are meaningful only because P is shared. No extra sketch state.
    gate_norm   : LayerNorm on each scale's feature before 'avg' or 'gate'
    strict_time : the sketch holds only edges strictly older than the query time
    learn_decay : each scale's rate(s) learned from its given lambda with exact streaming gradients (models/LDTPNet.py); not with lazy
"""
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn

from models.TPNet import RandomProjectionModule
from utils.utils import NegativeEdgeSampler


class LazyRandomProjectionModule(RandomProjectionModule):
    """Same values as RandomProjectionModule, without writing to every node on every batch.

    True layer i = random_projections[i] * layer_scale[i], the forward-decay idea (Cormode et al., ICDE 2009).
    Moving time forward changes one number per layer, so a batch touches only the rows of its own nodes.
    """
    MIN_SCALE = 1e-12  # below this the layer is written out, which keeps stored values inside float32 range

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_scale = nn.Parameter(torch.ones(self.num_layer + 1, dtype=torch.float64), requires_grad=False)

    def update(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        src_node_ids = torch.from_numpy(src_node_ids).to(self.device)
        dst_node_ids = torch.from_numpy(dst_node_ids).to(self.device)
        next_time = node_interact_times[-1]
        node_interact_times = torch.from_numpy(node_interact_times).to(dtype=torch.float, device=self.device)
        time_weight = torch.exp(-self.time_decay_weight * (next_time - node_interact_times))[:, None]
        step = math.exp(-self.time_decay_weight * (float(next_time) - self.now_time.item()))

        scale = self.layer_scale.tolist()
        for i in range(1, self.num_layer + 1):
            scale[i] *= step ** i
            if scale[i] < self.MIN_SCALE:
                self.random_projections[i].data = self.random_projections[i].data * scale[i]
                scale[i] = 1.0

        for i in range(self.num_layer, 0, -1):
            ratio = scale[i - 1] / scale[i]
            src_update_messages = self.random_projections[i - 1][dst_node_ids] * time_weight * ratio
            dst_update_messages = self.random_projections[i - 1][src_node_ids] * time_weight * ratio
            self.random_projections[i].scatter_add_(dim=0, index=src_node_ids[:, None].expand(-1, self.dim),
                                                    src=src_update_messages)
            self.random_projections[i].scatter_add_(dim=0, index=dst_node_ids[:, None].expand(-1, self.dim),
                                                    src=dst_update_messages)
        self.layer_scale.data = torch.tensor(scale, dtype=torch.float64, device=self.device)
        self.now_time.data = torch.tensor(next_time, device=self.device)

    def get_random_projections(self, node_ids: np.ndarray):
        scale = self.layer_scale.float()
        return [self.random_projections[i][node_ids] * scale[i] for i in range(self.num_layer + 1)]

    def reset_random_projections(self):
        super().reset_random_projections()
        self.layer_scale.data.fill_(1.0)

    def backup_random_projections(self):
        return super().backup_random_projections(), self.layer_scale.clone()

    def reload_random_projections(self, random_projections):
        backup, layer_scale = random_projections
        super().reload_random_projections(backup)
        self.layer_scale.data = layer_scale.clone()


class MultiScaleRandomProjectionModule(nn.Module):
    def __init__(self, lambdas: List[float], fusion: str, base_index: int, node_num: int, edge_num: int,
                 dim_factor: int, num_layer: int, device: str, use_matrix: bool, beginning_time: np.float64,
                 not_scale: bool, enforce_dim: int, strict_time: bool = False, lazy: bool = False,
                 cross: bool = False, gate_norm: bool = False, learn_decay: str = 'none'):
        super().__init__()
        assert learn_decay == 'none' or not lazy, 'learned rates keep tangent states for every node, so lazy does not apply'
        self.strict_time = strict_time  # only edges strictly older than the query time may be in the sketch
        self.pending = []
        assert len(lambdas) >= 1, 'need at least one decay rate'
        assert fusion in ('concat', 'avg', 'gate'), f'unknown fusion {fusion}'
        assert 0 <= base_index < len(lambdas), 'rp_base_index out of range'
        assert not use_matrix, 'multi-scale module supports the random-projection mode only'
        assert not cross or (fusion == 'concat' and len(lambdas) > 1), 'cross needs concat and at least 2 scales'
        self.lambdas = [float(l) for l in lambdas]
        self.num_scales = len(self.lambdas)
        self.fusion = fusion
        self.base_index = base_index
        self.num_layer = num_layer
        self.not_scale = not_scale

        common = dict(node_num=node_num, edge_num=edge_num, dim_factor=dim_factor, num_layer=num_layer, device=device,
                      use_matrix=use_matrix, beginning_time=beginning_time, not_scale=not_scale, enforce_dim=enforce_dim)
        if learn_decay != 'none':
            from models.LDTPNet import LearnableDecayRandomProjectionModule
            self.scales = nn.ModuleList([LearnableDecayRandomProjectionModule(decay_mode=learn_decay, init_lambda=lam, **common)
                                         for lam in self.lambdas])
        else:
            module = LazyRandomProjectionModule if lazy else RandomProjectionModule
            self.scales = nn.ModuleList([module(time_decay_weight=lam, **common) for lam in self.lambdas])
        # share ONE projection P across all scales (layer 0); P is not trainable
        shared_p = self.scales[0].random_projections[0]
        for scale in self.scales[1:]:
            scale.random_projections[0] = shared_p
        self.dim = self.scales[0].dim

        per_scale_dim = self.scales[0].pair_wise_feature_dim
        self.per_scale_dim = per_scale_dim
        if fusion == 'concat':
            self.pair_wise_feature_dim = per_scale_dim * self.num_scales
        else:
            self.pair_wise_feature_dim = per_scale_dim
            # residual weight, initialised to 0 so the model starts exactly at the base-scale TPNet feature
            self.beta = nn.Parameter(torch.zeros(1))
            if fusion == 'gate':
                self.gate = nn.Sequential(nn.Linear(per_scale_dim * self.num_scales, per_scale_dim), nn.ReLU(),
                                          nn.Linear(per_scale_dim, self.num_scales))
        self.norms = nn.ModuleList([nn.LayerNorm(per_scale_dim) for _ in self.lambdas]) if gate_norm else None

        self.cross = cross
        if cross:
            layers = num_layer + 1
            block = torch.arange(self.num_scales * layers) // layers
            self.register_buffer('cross_mask', block[:, None] != block[None, :], persistent=False)
            cross_dim = int(self.cross_mask.sum())
            self.cross_mlp = nn.Sequential(nn.Linear(cross_dim, cross_dim * 4), nn.ReLU(), nn.Linear(cross_dim * 4, cross_dim))
            self.pair_wise_feature_dim += cross_dim

    def _apply_update(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        for scale in self.scales:
            scale.update(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids,
                         node_interact_times=node_interact_times)

    def update(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        if self.strict_time:
            self.pending.append((src_node_ids.copy(), dst_node_ids.copy(), node_interact_times.copy()))
        else:
            self._apply_update(src_node_ids, dst_node_ids, node_interact_times)

    def advance_to(self, node_interact_times: np.ndarray):
        """strict_time: before predicting at these times, apply the waiting edges that are strictly older than all of them."""
        if not self.pending:
            return
        src, dst, times = [np.concatenate(x) for x in zip(*self.pending)]
        older = times < np.min(node_interact_times)
        if older.any():
            self._apply_update(src[older], dst[older], times[older])
        self.pending = [(src[~older], dst[~older], times[~older])] if (~older).any() else []

    def flush(self):
        """strict_time: apply all waiting edges. Used at a split boundary, where later times are strictly greater."""
        if self.pending:
            self._apply_update(*[np.concatenate(x) for x in zip(*self.pending)])
            self.pending = []

    def cross_scale_inner_products(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray):
        """Inner products of u's layers at one scale with v's layers at a different scale, shape (batch, cross_dim)."""
        src = torch.cat([torch.stack(s.get_random_projections(src_node_ids), dim=1) for s in self.scales], dim=1)
        dst = torch.cat([torch.stack(s.get_random_projections(dst_node_ids), dim=1) for s in self.scales], dim=1)
        return torch.matmul(src, dst.transpose(1, 2))[:, self.cross_mask]

    def get_pair_wise_feature(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray):
        feats = [scale.get_pair_wise_feature(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids)
                 for scale in self.scales]
        if self.norms is not None:
            feats = [norm(f) for norm, f in zip(self.norms, feats)]
        if self.fusion == 'concat':
            if self.cross:
                x = self.cross_scale_inner_products(src_node_ids, dst_node_ids)
                if not self.not_scale:
                    x = torch.log(x.clamp(min=0) + 1.0)
                feats.append(self.cross_mlp(x))
            return torch.cat(feats, dim=-1)
        stacked = torch.stack(feats, dim=1)  # (batch, J, D)
        if self.fusion == 'avg':
            mixed = stacked.mean(dim=1)
        else:
            alpha = torch.softmax(self.gate(torch.cat(feats, dim=-1)), dim=-1)  # (batch, J)
            self.last_alpha = alpha.detach()
            mixed = (alpha.unsqueeze(-1) * stacked).sum(dim=1)
        return feats[self.base_index] + self.beta * mixed

    def reset_random_projections(self):
        self.pending = []
        # scale 0 re-draws the shared P exactly as TPNet does (same RNG use as single-scale TPNet when J = 1)
        self.scales[0].reset_random_projections()
        for scale in self.scales[1:]:
            for i in range(1, scale.num_layer + 1):
                nn.init.zeros_(scale.random_projections[i])
            scale.now_time.data = scale.begging_time.clone()
            if hasattr(scale, 'layer_scale'):
                scale.layer_scale.data.fill_(1.0)
            if hasattr(scale, 'zero_tangents'):
                scale.zero_tangents()
        for scale in self.scales[1:]:
            assert scale.random_projections[0] is self.scales[0].random_projections[0], 'P is no longer shared'

    def decay_summary(self) -> str:
        parts = [f'scale {j}: ' + (s.decay_summary().split(': ', 1)[1] if hasattr(s, 'decay_summary') else f'{s.time_decay_weight:.3e} (fixed)')
                 for j, s in enumerate(self.scales)]
        return 'decay rates per scale -> ' + ' ; '.join(parts)

    def backup_random_projections(self):
        self.flush()
        return [scale.backup_random_projections() for scale in self.scales]

    def reload_random_projections(self, random_projections):
        assert len(random_projections) == self.num_scales
        self.pending = []
        for scale, backup in zip(self.scales, random_projections):
            scale.reload_random_projections(backup)


def build_random_projection_module(args, node_num: int, edge_num: int, beginning_time):
    """Return TPNet's single-scale module, or the multi-scale module when --rp_lambdas is given."""
    common = dict(node_num=node_num, edge_num=edge_num, dim_factor=args.rp_dim_factor,
                  num_layer=args.rp_num_layer, device=args.device, use_matrix=args.rp_use_matrix,
                  beginning_time=beginning_time, not_scale=args.rp_not_scale, enforce_dim=args.enforce_dim)
    if getattr(args, 'rp_lambdas', None):
        return MultiScaleRandomProjectionModule(lambdas=args.rp_lambdas, fusion=args.rp_fusion,
                                                base_index=args.rp_base_index,
                                                strict_time=getattr(args, 'rp_strict_time', False),
                                                lazy=getattr(args, 'rp_lazy', False),
                                                cross=getattr(args, 'rp_cross', False),
                                                gate_norm=getattr(args, 'rp_gate_norm', False),
                                                learn_decay=getattr(args, 'rp_learn_decay', 'none'), **common)
    return RandomProjectionModule(time_decay_weight=args.rp_time_decay_weight, **common)


def replace_false_negatives(src_node_ids: np.ndarray, neg_dst_node_ids: np.ndarray, node_interact_times: np.ndarray,
                            data, tries: int = 20):
    """Redraw training negatives that are self-loops or true edges at that same time (either direction).

    On a near-complete yearly graph a clean negative may not exist; the last draw is then kept.
    """
    if not hasattr(data, 'edges_at_time'):
        src, dst, times = data.src_node_ids.tolist(), data.dst_node_ids.tolist(), data.node_interact_times.tolist()
        data.edges_at_time = set(zip(src, dst, times)) | set(zip(dst, src, times))
        data.dst_pool = np.unique(data.dst_node_ids)
    neg = neg_dst_node_ids.copy()
    for i, (u, t) in enumerate(zip(src_node_ids.tolist(), node_interact_times.tolist())):
        for _ in range(tries):
            v = int(neg[i])
            if v != u and (u, v, t) not in data.edges_at_time:
                break
            neg[i] = data.dst_pool[np.random.randint(len(data.dst_pool))]
    return neg


class HistoricalTrainNegatives:
    """Draw training negatives the way historical evaluation does, cheaply: pairs seen strictly before the batch that are
    not active in the batch's time interval. A batch costs O(batch) instead of a scan of all edges."""

    def __init__(self, data, tries: int = 10):
        self.tries = tries
        first = {}
        self.pairs_at_time = {}
        for u, v, t in zip(data.src_node_ids.tolist(), data.dst_node_ids.tolist(), data.node_interact_times.tolist()):
            first.setdefault((u, v), t)
            self.pairs_at_time.setdefault(t, set()).add((u, v))
        pairs = list(first.items())  # in order of first appearance, so first_time is sorted
        self.pair_src = np.array([p[0][0] for p in pairs], dtype=np.longlong)
        self.pair_dst = np.array([p[0][1] for p in pairs], dtype=np.longlong)
        self.first_time = np.array([p[1] for p in pairs], dtype=np.float64)

    def mix(self, batch_src: np.ndarray, batch_dst: np.ndarray, neg_dst: np.ndarray, batch_times: np.ndarray, fraction: float):
        """Replace about `fraction` of the batch's random negatives. Early batches with no history stay random."""
        neg_src, neg_dst = batch_src.copy(), neg_dst.copy()
        available = int(np.searchsorted(self.first_time, batch_times[0], side='left'))
        if available == 0:
            return neg_src, neg_dst
        active = set(zip(batch_src.tolist(), batch_dst.tolist())) | self.pairs_at_time.get(float(batch_times[0]), set()) \
            | self.pairs_at_time.get(float(batch_times[-1]), set())
        for i in np.flatnonzero(np.random.random(len(batch_src)) < fraction):
            for _ in range(self.tries):
                k = np.random.randint(available)
                if (int(self.pair_src[k]), int(self.pair_dst[k])) not in active:
                    neg_src[i], neg_dst[i] = self.pair_src[k], self.pair_dst[k]
                    break
        return neg_src, neg_dst


class CachedNegativeSampler(NegativeEdgeSampler):
    """A seeded evaluation sampler returns the same negatives on every pass, so compute them once.

    Historical sampling scans all edges per batch, which is too slow to repeat every epoch during validation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}

    def sample(self, size, batch_src_node_ids=None, batch_dst_node_ids=None, current_batch_start_time=0.0,
               current_batch_end_time=0.0):
        key = (size, float(current_batch_start_time), float(current_batch_end_time),
               None if batch_src_node_ids is None else batch_src_node_ids.tobytes())
        if key not in self.cache:
            self.cache[key] = super().sample(size, batch_src_node_ids, batch_dst_node_ids, current_batch_start_time,
                                             current_batch_end_time)
        return self.cache[key]
