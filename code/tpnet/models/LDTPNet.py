"""
Learnable decay rates for TPNet's temporal walk sketches, trained with exact streaming (forward-mode) gradients.

TPNet keeps H_0 = P and, for depth i = 1..k, H_i = A_i P where A_i weights every i-step temporal walk by
prod over its edges of exp(-lambda * (now - t_edge)); lambda is grid-searched. Here the rates are learned:
  * 'shared'    : one lambda for all depths (TPNet's model, lambda learned instead of grid-searched)
  * 'per_depth' : lambda_1..lambda_k, where lambda_h decays the h-th edge of a walk counted from its OLDEST edge
                  (the indexing under which the streaming state stays exact).
lambda_h = exp(phi_h); phi is the trainable parameter.

Exact gradients: we also maintain tangent states T_{i,m} = dH_i / d lambda_m (m <= i), updated with the product
rule in the same streaming pass (no backprop through time). When TPNet reads a node's sketch we return
    H_i[ids] + sum_m (lambda_m - lambda_m.detach()) * T_{i,m}[ids]
whose value is exactly H_i[ids] and whose gradient w.r.t. lambda_m is the exact sensitivity of the current state.
(As in all online/RTRL-style training, the state was built while lambda was changing during the epoch; TPNet
resets the state at the start of every epoch.)
"""
import math

import numpy as np
import torch
import torch.nn as nn

from models.TPNet import RandomProjectionModule

MIN_LAMBDA = 1e-12  # used when a requested initial rate is 0 (log(0) is undefined)


class LearnableDecayRandomProjectionModule(RandomProjectionModule):
    def __init__(self, decay_mode: str, init_lambda: float, node_num: int, edge_num: int, dim_factor: int,
                 num_layer: int, device: str, use_matrix: bool, beginning_time: np.float64, not_scale: bool,
                 enforce_dim: int):
        super().__init__(node_num=node_num, edge_num=edge_num, dim_factor=dim_factor, num_layer=num_layer,
                         time_decay_weight=float(init_lambda), device=device, use_matrix=use_matrix,
                         beginning_time=beginning_time, not_scale=not_scale, enforce_dim=enforce_dim)
        assert decay_mode in ('shared', 'per_depth'), f'unknown decay mode {decay_mode}'
        self.decay_mode = decay_mode
        n_params = 1 if decay_mode == 'shared' else num_layer
        self.log_lambda = nn.Parameter(torch.full((n_params,), math.log(max(float(init_lambda), MIN_LAMBDA))))
        shape = self.random_projections[0].shape
        # tangent states T_{i,m}, i = 1..k, m = 1..i, stored as buffers so checkpoints and .to(device) include them
        for i in range(1, num_layer + 1):
            for m in range(1, i + 1):
                self.register_buffer(f'tan_{i}_{m}', torch.zeros(shape))

    # ------------------------------------------------------------------ helpers
    def tangent(self, i: int, m: int) -> torch.Tensor:
        return getattr(self, f'tan_{i}_{m}')

    def lambdas(self) -> torch.Tensor:
        """decay rates lambda_1..lambda_k (with gradient)."""
        lam = torch.exp(self.log_lambda)
        return lam.expand(self.num_layer) if self.decay_mode == 'shared' else lam

    def zero_tangents(self):
        for i in range(1, self.num_layer + 1):
            for m in range(1, i + 1):
                self.tangent(i, m).zero_()

    def decay_summary(self) -> str:
        lam = self.lambdas().detach().double().cpu().numpy()
        return 'learned decay rates (oldest edge first): ' + ', '.join(f'{x:.3e}' for x in lam)

    # ------------------------------------------------------------------ streaming update
    @torch.no_grad()
    def update(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        device = self.random_projections[0].device
        dtype = self.random_projections[0].dtype
        k = self.num_layer
        lam = self.lambdas().detach().double()
        cum = torch.cumsum(lam, dim=0)                       # total rate of a depth-i walk: sum_{h<=i} lambda_h
        src = torch.from_numpy(np.asarray(src_node_ids)).to(device)
        dst = torch.from_numpy(np.asarray(dst_node_ids)).to(device)
        next_time = float(node_interact_times[-1])
        dt = next_time - float(self.now_time.item())
        ages = torch.from_numpy(next_time - np.asarray(node_interact_times, dtype=np.float64)).to(device)  # float64
        H = self.random_projections

        # 1) move the clock from now_time to next_time: H_i *= exp(-cum_i dt);  T_{i,m} = f (T_{i,m} - dt H_i)
        if dt != 0.0:
            for i in range(1, k + 1):
                f = math.exp(-float(cum[i - 1]) * dt)
                for m in range(1, i + 1):
                    T = self.tangent(i, m)
                    T.sub_(dt * H[i].data).mul_(f)
                H[i].data.mul_(f)

        # 2) add the batch's edges (depth i from depth i-1, highest depth first so edges in a batch do not chain)
        for i in range(k, 0, -1):
            w64 = torch.exp(-lam[i - 1] * ages)                # weight of the new (newest) edge, which is hop i
            w = w64.to(dtype)[:, None]
            dw = (-ages * w64).to(dtype)[:, None]             # d w / d lambda_i
            prev_dst, prev_src = H[i - 1][dst], H[i - 1][src]
            tan_msgs = []
            for m in range(1, i + 1):
                if m < i:
                    t_dst = self.tangent(i - 1, m)[dst] * w
                    t_src = self.tangent(i - 1, m)[src] * w
                else:  # m == i: only the new edge depends on lambda_i
                    t_dst = prev_dst * dw
                    t_src = prev_src * dw
                tan_msgs.append((m, t_dst, t_src))
            idx_src = src[:, None].expand(-1, self.dim)
            idx_dst = dst[:, None].expand(-1, self.dim)
            H[i].data.scatter_add_(0, idx_src, prev_dst * w)
            H[i].data.scatter_add_(0, idx_dst, prev_src * w)
            for m, t_dst, t_src in tan_msgs:
                self.tangent(i, m).scatter_add_(0, idx_src, t_dst)
                self.tangent(i, m).scatter_add_(0, idx_dst, t_src)

        self.now_time.data = torch.tensor(next_time, dtype=self.now_time.dtype, device=self.now_time.device)

    # ------------------------------------------------------------------ reading the sketch
    def get_random_projections(self, node_ids: np.ndarray):
        out = [self.random_projections[0][node_ids]]
        use_grad = torch.is_grad_enabled() and self.log_lambda.requires_grad
        lam = self.lambdas() if use_grad else None
        for i in range(1, self.num_layer + 1):
            h = self.random_projections[i][node_ids]
            if use_grad:
                for m in range(1, i + 1):
                    # value-preserving: (lam - lam.detach()) == 0, gradient = tangent
                    h = h + (lam[m - 1] - lam[m - 1].detach()) * self.tangent(i, m)[node_ids]
            out.append(h)
        return out

    # ------------------------------------------------------------------ state management (as TPNet)
    def reset_random_projections(self):
        super().reset_random_projections()
        self.zero_tangents()

    def backup_random_projections(self):
        now_time, layers = super().backup_random_projections()
        tans = {name: buf.clone() for name, buf in self.named_buffers() if name.startswith('tan_')}
        return now_time, layers, tans

    def reload_random_projections(self, random_projections):
        now_time, layers, tans = random_projections
        super().reload_random_projections((now_time, layers))
        for name, value in tans.items():
            getattr(self, name).copy_(value)


def add_decay_param_group(optimizer, model: nn.Module, args):
    """Rebuild the optimizer so learnable decay parameters get their own learning rate (no weight decay)."""
    decay, others, seen = [], [], set()
    for name, prm in model.named_parameters():
        if id(prm) in seen or not prm.requires_grad:
            continue
        seen.add(id(prm))
        (decay if name.endswith('log_lambda') else others).append(prm)
    if not decay:
        return optimizer
    groups = [{'params': others, 'lr': args.learning_rate, 'weight_decay': args.weight_decay},
              {'params': decay, 'lr': args.rp_decay_lr, 'weight_decay': 0.0}]
    return optimizer.__class__(groups)
