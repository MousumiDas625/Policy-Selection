"""
train_iql.py — Implicit Q-Learning for VLA policy selection.

Architecture:
  z_t = Qwen(image, instruction)           [precomputed, frozen]
  e_a = learned embedding per VLA           [trainable or frozen]
  Q(s,a) = MLP([z_t; e_a]) → scalar        [trainable, double-Q]
  V(s)   = MLP(z_t) → scalar               [trainable]

Three forward passes per state to get Q for all policies.
Same MLP weights, different policy embedding vectors.

IQL:
  L_V = E[L_τ(min(Q1,Q2) - V)]   expectile regression
  L_Q = E[(r + γ·V_target(s')·(1-d) - Q)²]   TD

Usage:
  CUDA_VISIBLE_DEVICES=0 python train_iql.py \
      --expt-dir /project2/.../iqlSpatial/iql_train/expt1 \
      --policies pi0,openvla,molmoact \
      --epochs 200 --lr 3e-4 --gamma 0.99 --tau 0.7 \
      --freeze-policy-emb        # optional: freeze policy embeddings
"""

import argparse
import json
import os
import pickle
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════════════
#  NETWORKS
# ═══════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class IQLModel(nn.Module):
    def __init__(self, state_dim, n_policies, policy_names,
                 policy_embed_dim=64, hidden=(256, 256), dropout=0.1,
                 freeze_policy_emb=False):
        super().__init__()
        self.policy_names = policy_names
        self.n_policies = n_policies
        self.name2idx = {n: i for i, n in enumerate(policy_names)}
        self.state_dim = state_dim
        self.policy_embed_dim = policy_embed_dim

        # Policy embeddings — randomly initialized
        self.policy_emb = nn.Embedding(n_policies, policy_embed_dim)
        nn.init.normal_(self.policy_emb.weight, std=0.02)

        if freeze_policy_emb:
            self.policy_emb.weight.requires_grad = False
            print("  Policy embeddings: FROZEN")
        else:
            print("  Policy embeddings: TRAINABLE")

        q_in = state_dim + policy_embed_dim
        self.q1 = MLP(q_in, list(hidden), 1, dropout)
        self.q2 = MLP(q_in, list(hidden), 1, dropout)
        self.v = MLP(state_dim, list(hidden), 1, dropout)

        # Target V
        self.v_tgt = MLP(state_dim, list(hidden), 1, dropout)
        self.v_tgt.load_state_dict(self.v.state_dict())
        for p in self.v_tgt.parameters():
            p.requires_grad = False

    def q_vals(self, s, a_idx):
        e = self.policy_emb(a_idx)
        x = torch.cat([s, e], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def v_val(self, s):
        return self.v(s).squeeze(-1)

    def v_tgt_val(self, s):
        with torch.no_grad():
            return self.v_tgt(s).squeeze(-1)

    def all_q(self, s):
        """Q-values for ALL policies → (batch, n_policies)."""
        qs = []
        for i in range(self.n_policies):
            idx = torch.full((s.shape[0],), i, dtype=torch.long, device=s.device)
            q1, q2 = self.q_vals(s, idx)
            qs.append(torch.min(q1, q2).unsqueeze(-1))
        return torch.cat(qs, -1)

    def select(self, s):
        """argmax Q → (indices, names)"""
        if s.dim() == 1:
            s = s.unsqueeze(0)
        q = self.all_q(s)
        idx = q.argmax(-1)
        return idx, [self.policy_names[i.item()] for i in idx]

    def update_target(self, polyak=0.005):
        for p, pt in zip(self.v.parameters(), self.v_tgt.parameters()):
            pt.data.mul_(1 - polyak).add_(polyak * p.data)


# ═══════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════

class ChunkDataset(Dataset):
    def __init__(self, chunks_path, emb_path, name2idx):
        chunks = pickle.load(open(chunks_path, "rb"))
        emb = torch.load(emb_path, map_location="cpu")

        self.obs = emb["obs_emb"]
        self.nobs = emb["next_obs_emb"]
        self.hdim = emb["hidden_dim"]
        assert len(chunks) == self.obs.shape[0]

        self.pol = torch.tensor([name2idx[c["policy"]] for c in chunks], dtype=torch.long)
        self.rew = torch.tensor([c["reward"] for c in chunks], dtype=torch.float32)
        self.done = torch.tensor([float(c["done"]) for c in chunks], dtype=torch.float32)
        self.tid = torch.tensor([c["task_id"] for c in chunks], dtype=torch.long)
        self._pol_names = [c["policy"] for c in chunks]

        pos = (self.rew > 0).sum().item()
        print(f"  {len(self)} chunks, dim={self.hdim}, reward>0: {pos}/{len(self)}")

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, i):
        return self.obs[i], self.nobs[i], self.pol[i], self.rew[i], self.done[i]


# ═══════════════════════════════════════════════════════════════════
#  LOSSES
# ═══════════════════════════════════════════════════════════════════

def expectile_loss(pred, target, tau):
    diff = target - pred
    w = torch.where(diff > 0, tau, 1 - tau)
    return (w * diff.pow(2)).mean()


def iql_step(model, batch, opt_q, opt_v, gamma, tau, device, reward_scale=1.0):
    s, s2, a, r, d = [x.to(device) for x in batch]

    # V update
    with torch.no_grad():
        q1, q2 = model.q_vals(s, a)
        q_min = torch.min(q1, q2)
    v = model.v_val(s)
    lv = expectile_loss(v, q_min, tau)
    opt_v.zero_grad()
    lv.backward()
    nn.utils.clip_grad_norm_(model.v.parameters(), 1.0)
    opt_v.step()

    # Q update
    with torch.no_grad():
        vn = model.v_tgt_val(s2)
        tgt = r * reward_scale + gamma * vn * (1.0 - d)
    q1, q2 = model.q_vals(s, a)
    lq = F.mse_loss(q1, tgt) + F.mse_loss(q2, tgt)

    opt_q.zero_grad()
    lq.backward()
    q_params = list(model.q1.parameters()) + list(model.q2.parameters())
    if model.policy_emb.weight.requires_grad:
        q_params += list(model.policy_emb.parameters())
    nn.utils.clip_grad_norm_(q_params, 1.0)
    opt_q.step()

    model.update_target()
    return {"v_loss": lv.item(), "q_loss": lq.item(),
            "q_mean": q1.mean().item(), "v_mean": v.mean().item()}


# ═══════════════════════════════════════════════════════════════════
#  OFFLINE EVAL
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_offline(model, ds, device, pol_names):
    model.eval()
    loader = DataLoader(ds, 512, shuffle=False)
    sel_all, data_a, rew_all, q_all_list = [], [], [], []
    for s, s2, a, r, d in loader:
        s = s.to(device)
        q = model.all_q(s)
        sel_all.append(q.argmax(-1).cpu())
        data_a.append(a)
        rew_all.append(r)
        q_all_list.append(q.cpu())

    sel_all = torch.cat(sel_all)
    data_a = torch.cat(data_a)
    rew_all = torch.cat(rew_all)
    q_all = torch.cat(q_all_list)

    agree = (sel_all == data_a).float().mean().item()
    sm = rew_all > 0
    s_agree = (sel_all[sm] == data_a[sm]).float().mean().item() if sm.any() else 0
    sel_dist = {pol_names[i]: (sel_all == i).sum().item() for i in range(len(pol_names))}
    mean_q = {pol_names[i]: q_all[:, i].mean().item() for i in range(len(pol_names))}
    best = max(mean_q, key=mean_q.get)

    model.train()
    return {"agreement": agree, "succ_agreement": s_agree,
            "sel_dist": sel_dist, "mean_q": mean_q, "preferred": best}


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expt-dir", required=True)
    parser.add_argument("--policies", default="pi0,openvla,molmoact")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", default="256,256")
    parser.add_argument("--policy-embed-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze-policy-emb", action="store_true",
                        help="Freeze policy embeddings (keep random init)")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reward-scale", type=float, default=1.0,
                        help="Multiply rewards by this before TD update (e.g. 10.0)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    pols = [p.strip() for p in args.policies.split(",")]
    n2i = {n: i for i, n in enumerate(pols)}
    hidden = tuple(int(h) for h in args.hidden.split(","))

    data_dir = os.path.join(args.expt_dir, "data")
    run_name = (f"iql_{'_'.join(pols)}_tau{args.tau}_g{args.gamma}"
                f"{'_frozen' if args.freeze_policy_emb else ''}"
                f"_{time.strftime('%Y%m%d_%H%M%S')}")
    run_dir = os.path.join(args.expt_dir, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Experiment: {args.expt_dir}")
    print(f"Run:        {run_dir}")
    print(f"Policies:   {pols}")
    print(f"Device:     {device}")

    # Load data
    print("\nTrain data:")
    train_ds = ChunkDataset(
        os.path.join(data_dir, "train_chunks.pkl"),
        os.path.join(data_dir, "train_embeddings.pt"), n2i)

    eval_ds = None
    ec = os.path.join(data_dir, "eval_chunks.pkl")
    ee = os.path.join(data_dir, "eval_embeddings.pt")
    if os.path.exists(ec) and os.path.exists(ee):
        print("Eval data:")
        eval_ds = ChunkDataset(ec, ee, n2i)

    sdim = train_ds.hdim

    # Model
    model = IQLModel(sdim, len(pols), pols, args.policy_embed_dim,
                     hidden, args.dropout, args.freeze_policy_emb).to(device)
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable params: {npar:,}, state_dim={sdim}")

    # Optimizers
    q_params = list(model.q1.parameters()) + list(model.q2.parameters())
    if not args.freeze_policy_emb:
        q_params += list(model.policy_emb.parameters())
    opt_q = torch.optim.Adam(q_params, lr=args.lr, weight_decay=1e-5)
    opt_v = torch.optim.Adam(model.v.parameters(), lr=args.lr, weight_decay=1e-5)

    from torch.optim.lr_scheduler import CosineAnnealingLR
    sched_q = CosineAnnealingLR(opt_q, T_max=args.epochs, eta_min=args.lr * 0.01)
    sched_v = CosineAnnealingLR(opt_v, T_max=args.epochs, eta_min=args.lr * 0.01)
    print(f"  LR schedule: cosine {args.lr:.0e} → {args.lr*0.01:.0e} over {args.epochs} epochs")

    loader = DataLoader(train_ds, args.batch_size, shuffle=True, drop_last=True)

    # Save config
    cfg = vars(args).copy()
    cfg["n_train"] = len(train_ds)
    cfg["n_eval"] = len(eval_ds) if eval_ds else 0
    cfg["state_dim"] = sdim
    cfg["trainable_params"] = npar
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Log
    log = open(os.path.join(run_dir, "log.csv"), "w")
    log.write("epoch,v_loss,q_loss,q_mean,v_mean,"
              "tr_agree,tr_succ_agree,tr_preferred,"
              "ev_agree,ev_succ_agree,ev_preferred\n")

    print(f"\nTraining {args.epochs} epochs, {len(loader)} batches/ep")
    print(f"{'─'*80}")

    best_metric = -1e9

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = defaultdict(list)
        for batch in loader:
            out = iql_step(model, batch, opt_q, opt_v, args.gamma, args.tau, device, args.reward_scale)
            for k, v in out.items():
                losses[k].append(v)

        avg = {k: np.mean(v) for k, v in losses.items()}

        do_eval = (ep % args.eval_every == 0) or ep == 1 or ep == args.epochs
        if do_eval:
            te = eval_offline(model, train_ds, device, pols)
            ee_res = eval_offline(model, eval_ds, device, pols) if eval_ds else {
                "agreement": 0, "succ_agreement": 0, "preferred": "", "mean_q": {}, "sel_dist": {}}

            log.write(f"{ep},{avg['v_loss']:.6f},{avg['q_loss']:.6f},"
                      f"{avg['q_mean']:.4f},{avg['v_mean']:.4f},"
                      f"{te['agreement']:.4f},{te['succ_agreement']:.4f},{te['preferred']},"
                      f"{ee_res['agreement']:.4f},{ee_res['succ_agreement']:.4f},{ee_res.get('preferred','')}\n")
            log.flush()

            print(f"Ep {ep:>4}/{args.epochs} | V={avg['v_loss']:.4f} Q={avg['q_loss']:.4f} "
                  f"Q̄={avg['q_mean']:.3f} V̄={avg['v_mean']:.3f}")
            print(f"  train: agree={te['agreement']:.3f} succ_agree={te['succ_agreement']:.3f} "
                  f"pref={te['preferred']}")
            print(f"  train Q: { {k: round(v,3) for k,v in te['mean_q'].items()} }")
            print(f"  train sel: {te['sel_dist']}")
            if eval_ds:
                print(f"  eval:  agree={ee_res['agreement']:.3f} pref={ee_res['preferred']}")
                print(f"  eval  Q: { {k: round(v,3) for k,v in ee_res['mean_q'].items()} }")
                print(f"  eval  sel: {ee_res['sel_dist']}")

                qv = list(ee_res["mean_q"].values())
                metric = max(qv) - min(qv) if qv else 0
                if metric > best_metric:
                    best_metric = metric
                    torch.save({
                        "state_dict": model.state_dict(),
                        "policies": pols, "state_dim": sdim,
                        "policy_embed_dim": args.policy_embed_dim,
                        "hidden": hidden, "dropout": args.dropout,
                        "freeze_policy_emb": args.freeze_policy_emb,
                        "epoch": ep, "eval": ee_res, "train": te,
                    }, os.path.join(run_dir, "best.pt"))
                    print(f"  → best saved (spread={metric:.4f})")
            print(f"{'─'*80}")

        sched_q.step()
        sched_v.step()

        if ep % 50 == 0:
            print(f"Ep {ep:>4}/{args.epochs} | V={avg['v_loss']:.4f} Q={avg['q_loss']:.4f}")

    # Final save
    torch.save({
        "state_dict": model.state_dict(),
        "policies": pols, "state_dim": sdim,
        "policy_embed_dim": args.policy_embed_dim,
        "hidden": hidden, "dropout": args.dropout,
        "freeze_policy_emb": args.freeze_policy_emb,
        "epoch": args.epochs,
    }, os.path.join(run_dir, "final.pt"))

    log.close()
    print(f"\nDone! {run_dir}")
    print(f"  best.pt / final.pt / config.json / log.csv")


if __name__ == "__main__":
    main()
