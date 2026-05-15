"""
build_chunks.py — Per-VLA chunk transitions with stratified split and optional class balancing.

What this script does:
  1. Loads pickles from <data_root>/<policy>/iql_data/*.pkl for each policy.
  2. Hard-filters out HELDOUT tasks (default 5, 9) — these are NEVER used here.
  3. Per (VLA, task): stratified split of episodes into train_pool / val,
     using --val-split (default 0.2, rounded per stratum).
  4. If --balance is passed: per (VLA, task) on the train side, downsamples
     successes to --balance-target (default 25) and replicates failures up
     to the same target. Val is left untouched.
  5. Sanity-checks the reward/done labelling on every loaded episode.
  6. Builds chunk-level transitions and writes:
       <expt_dir>/data/train_chunks.pkl
       <expt_dir>/data/eval_chunks.pkl   (val — name preserved so embed_qwen
                                           and train_iql don't need changes)
       <expt_dir>/audit.json             (full per-cell counts)
       <expt_dir>/expt_config.json       (frozen run config)

Reward / done convention (per spec):
  - reward = 1.0  only for the single chunk that contains the success step,
                   which exists only in successful episodes.
  - reward = 0.0  for every other chunk, INCLUDING the terminal chunk of a
                   failed episode.
  - done   = True for the chunk containing the success step, AND for the
                   final chunk of any episode (so the IQL TD target masks
                   the bootstrap correctly).

Usage examples:
  # Test run, no balancing (natural distribution):
  python build_chunks.py \
      --data-root /project2/jessetho_1732/mousumid/PolicySel/iqlSpatial \
      --policies pi0,pi05 \
      --train-tasks 0,1,2,3,4,6,7,8 \
      --heldout-tasks 5,9 \
      --val-split 0.2 \
      --seed 42 \
      --expt-dir expt_pipi_test

  # Balanced run (25 succ + 25 fail per (VLA, task) on train):
  python build_chunks.py ... --expt-dir expt_pipi_balanced --balance
"""

import argparse
import glob
import json
import os
import pickle
import random
from collections import defaultdict

from config import CHUNK_SIZES


# ─────────────────────────────────────────────────────────────────────
# 1.  LOADING
# ─────────────────────────────────────────────────────────────────────

def load_pkls(data_dir, train_task_ids, heldout_task_ids):
    """
    Load every pickle in `data_dir`. Drop heldout tasks. Drop anything not
    in train_task_ids (e.g. if you ever passed a subset).
    Returns (kept_episodes, heldout_filenames).
    """
    eps = []
    heldout = []
    other = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.pkl"))):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        tid = d["task_id"]
        if tid in heldout_task_ids:
            heldout.append(f)
            continue
        if tid not in train_task_ids:
            other.append(f)
            continue
        eps.append(d)
    return eps, heldout, other


# ─────────────────────────────────────────────────────────────────────
# 2.  SANITY CHECK  (per-step reward/done labelling)
# ─────────────────────────────────────────────────────────────────────

def label_sanity_check(episodes, tag):
    """
    Verify each episode obeys the spec:
      success episode  → dones[-1] == True, rewards[-1] == 1.0, all others 0/False
      failure episode  → all dones False, all rewards 0.0
    Returns list of issue strings.
    """
    issues = []
    for ep in episodes:
        rews = ep["rewards"]
        dones = ep["dones"]
        succ = ep["episode_success"]
        ti, ei = ep["task_id"], ep["episode_idx"]
        if succ:
            if not bool(dones[-1]):
                issues.append(f"[{tag}] t{ti} e{ei}: success but dones[-1] is not True")
            if float(rews[-1]) != 1.0:
                issues.append(f"[{tag}] t{ti} e{ei}: success but rewards[-1] = {rews[-1]}")
            if any(bool(d) for d in dones[:-1]):
                issues.append(f"[{tag}] t{ti} e{ei}: success but a True done appears before final step")
            if any(float(r) != 0.0 for r in rews[:-1]):
                issues.append(f"[{tag}] t{ti} e{ei}: success but non-zero reward before final step")
        else:
            if any(bool(d) for d in dones):
                issues.append(f"[{tag}] t{ti} e{ei}: failure but at least one True done")
            if any(float(r) > 0.0 for r in rews):
                issues.append(f"[{tag}] t{ti} e{ei}: failure but at least one positive reward")
    return issues


# ─────────────────────────────────────────────────────────────────────
# 3.  STRATIFIED SPLIT  (per task, success/fail strata)
# ─────────────────────────────────────────────────────────────────────

def stratified_split_per_task(episodes, val_ratio, seed):
    """
    Within each task, split successes 80/20 and failures 80/20 separately
    (rounded). Returns (train_pool, val_pool).
    """
    rng = random.Random(seed)
    by_task = defaultdict(list)
    for ep in episodes:
        by_task[ep["task_id"]].append(ep)

    train_pool, val_pool = [], []
    for tid in sorted(by_task):
        s = [e for e in by_task[tid] if e["episode_success"]]
        f = [e for e in by_task[tid] if not e["episode_success"]]
        rng.shuffle(s); rng.shuffle(f)

        n_val_s = round(len(s) * val_ratio) if s else 0
        # Failures: val gets >=1 only if task has >=2 failures;
        # ALL failures stay in train regardless (overlap allowed so
        # train_embeddings.pt stays valid — no Qwen recompute needed).
        n_val_f = (max(1, round(len(f) * val_ratio)) if len(f) >= 2 else 0) if f else 0

        val_pool.extend(s[:n_val_s])
        val_pool.extend(f[:n_val_f])
        train_pool.extend(s[n_val_s:])
        train_pool.extend(f)             # ALL failures stay in train too
    return train_pool, val_pool


# ─────────────────────────────────────────────────────────────────────
# 4.  BALANCING  (per task: cap successes, replicate failures)
# ─────────────────────────────────────────────────────────────────────

def balance_train(train_pool, target, seed):
    """
    For each task in train_pool:
      - keep up to `target` successes (downsample if more)
      - replicate failures up to `target` (round-robin), if any exist
    Episodes that are replicas get a __replica_idx field for traceability.
    Returns (balanced_list, audit_rows).
    """
    rng = random.Random(seed)
    by_task = defaultdict(lambda: {"succ": [], "fail": []})
    for ep in train_pool:
        key = "succ" if ep["episode_success"] else "fail"
        by_task[ep["task_id"]][key].append(ep)

    balanced = []
    audit_rows = []
    for tid in sorted(by_task):
        s = list(by_task[tid]["succ"])
        f = list(by_task[tid]["fail"])
        rng.shuffle(s); rng.shuffle(f)

        # Successes: keep up to target (downsample)
        kept_s = s[:target]
        for ep in kept_s:
            ep.setdefault("__replica_idx", 0)

        # Failures: replicate up to target
        if not f:
            kept_f = []
        else:
            kept_f = []
            for i in range(target):
                src = f[i % len(f)]
                rep = dict(src)              # shallow copy — observations etc are shared refs
                rep["__replica_idx"] = i // len(f)
                kept_f.append(rep)

        balanced.extend(kept_s)
        balanced.extend(kept_f)
        audit_rows.append({
            "task_id": tid,
            "succ_unique": len(s),
            "succ_kept": len(kept_s),
            "fail_unique": len(f),
            "fail_kept": len(kept_f),
            "fail_deficit": max(0, target - len(kept_f)),
            "total_kept": len(kept_s) + len(kept_f),
        })
    return balanced, audit_rows


# ─────────────────────────────────────────────────────────────────────
# 5.  CHUNKING  (per-VLA, fixed chunk size)
# ─────────────────────────────────────────────────────────────────────

def chunks_from_individual(episodes, policy_name):
    """
    Split each episode into fixed-size chunks of size CHUNK_SIZES[policy].

    Reward / done per spec:
      - reward = 1.0  iff this chunk contains the final success step
      - reward = 0.0  for everything else (incl. terminal chunk of a failure)
      - done   = True iff this chunk contains success OR is the final chunk
                  of the episode
    """
    cs = CHUNK_SIZES.get(policy_name, 1)
    transitions = []
    for ep in episodes:
        obs = ep["observations"]
        dones = ep["dones"]
        n = ep["total_steps"]
        if n == 0:
            continue
        step = 0
        while step < n:
            end = min(step + cs, n)
            chunk_has_success = any(bool(d) for d in dones[step:end])
            is_final_chunk = (end >= n)
            reward = 1.0 if chunk_has_success else 0.0
            done_flag = chunk_has_success or is_final_chunk

            if end < n and not chunk_has_success:
                nxt = obs[end]["agentview"]
            else:
                nxt = obs[min(end - 1, n - 1)]["agentview"]

            transitions.append({
                "obs_image": obs[step]["agentview"],
                "next_obs_image": nxt,
                "language_instruction": ep["language_instruction"],
                "policy": policy_name,
                "reward": reward,
                "done": done_flag,
                "task_id": ep["task_id"],
                "episode_idx": ep["episode_idx"],
                "episode_success": ep["episode_success"],
                "chunk_start": step,
                "chunk_end": end,
                "source": "individual",
                "replica_idx": ep.get("__replica_idx", 0),
            })

            # A success terminates the trajectory. A failure is treated as a
            # full-length rollout, so we just walk to the end.
            if chunk_has_success:
                break
            step = end
    return transitions


# ─────────────────────────────────────────────────────────────────────
# 6.  AUDIT PRINTING
# ─────────────────────────────────────────────────────────────────────

def print_train_audit(pol, rows):
    print(f"\n  ┌─ TRAIN  VLA={pol}  ─────────────────────────────────────────────────")
    print(f"  │ task | succ_uniq | succ_kept | fail_uniq | fail_kept | deficit | total")
    print(f"  │ ─────┼───────────┼───────────┼───────────┼───────────┼─────────┼──────")
    for r in rows:
        print(f"  │  {r['task_id']:>3} |  {r['succ_unique']:>6}   |  {r['succ_kept']:>6}   "
              f"|  {r['fail_unique']:>6}   |  {r['fail_kept']:>6}   |  {r['fail_deficit']:>5}  |  {r['total_kept']:>4}")
    totals = {
        "succ_unique": sum(r["succ_unique"] for r in rows),
        "succ_kept":   sum(r["succ_kept"]   for r in rows),
        "fail_unique": sum(r["fail_unique"] for r in rows),
        "fail_kept":   sum(r["fail_kept"]   for r in rows),
        "fail_deficit":sum(r["fail_deficit"] for r in rows),
        "total_kept":  sum(r["total_kept"]  for r in rows),
    }
    print(f"  │ ─────┼───────────┼───────────┼───────────┼───────────┼─────────┼──────")
    print(f"  │  ALL |  {totals['succ_unique']:>6}   |  {totals['succ_kept']:>6}   "
          f"|  {totals['fail_unique']:>6}   |  {totals['fail_kept']:>6}   "
          f"|  {totals['fail_deficit']:>5}  |  {totals['total_kept']:>4}")
    print(f"  └─────────────────────────────────────────────────────────────────────")


def print_val_audit(pol, rows):
    print(f"\n  ┌─ VAL    VLA={pol}  ─────────────────")
    print(f"  │ task |  succ |  fail | total")
    print(f"  │ ─────┼───────┼───────┼──────")
    s_t = f_t = 0
    for r in rows:
        print(f"  │  {r['task_id']:>3} |  {r['succ']:>4} |  {r['fail']:>4} |  {r['succ']+r['fail']:>4}")
        s_t += r["succ"]; f_t += r["fail"]
    print(f"  │ ─────┼───────┼───────┼──────")
    print(f"  │  ALL |  {s_t:>4} |  {f_t:>4} |  {s_t+f_t:>4}")
    print(f"  └────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True,
                        help="e.g. /project2/.../iqlSpatial — contains <policy>/iql_data/*.pkl")
    parser.add_argument("--policies", default="pi0,pi05")
    parser.add_argument("--train-tasks", default="0,1,2,3,4,6,7,8")
    parser.add_argument("--heldout-tasks", default="5,9")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--balance", action="store_true",
                        help="If set: cap successes at --balance-target and replicate failures up to it (train only)")
    parser.add_argument("--balance-target", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expt-dir", required=True,
                        help="Name under <data-root>/iql_train/ for outputs")
    parser.add_argument("--skip-sanity", action="store_true",
                        help="Skip the per-episode reward/done sanity check")
    args = parser.parse_args()

    policies = [p.strip() for p in args.policies.split(",")]
    train_tasks = set(int(t) for t in args.train_tasks.split(",") if t.strip())
    heldout_tasks = set(int(t) for t in args.heldout_tasks.split(",") if t.strip())

    overlap = train_tasks & heldout_tasks
    assert not overlap, f"Train and held-out task IDs overlap: {sorted(overlap)}"

    expt_root = os.path.join(args.data_root, "iql_train", args.expt_dir)
    data_dir = os.path.join(expt_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    print(f"╔══════════════════════════════════════════════════════════════════════╗")
    print(f"║  build_chunks.py")
    print(f"╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  expt root:   {expt_root}")
    print(f"║  policies:    {policies}")
    print(f"║  train tasks: {sorted(train_tasks)}")
    print(f"║  held-out:    {sorted(heldout_tasks)}  (excluded from train AND val)")
    print(f"║  val split:   {args.val_split}")
    print(f"║  balance:     {args.balance}  (target = {args.balance_target} per task per VLA)")
    print(f"║  seed:        {args.seed}")
    print(f"╚══════════════════════════════════════════════════════════════════════╝")

    all_train_chunks = []
    all_val_chunks = []

    audit = {
        "balance": args.balance,
        "balance_target": args.balance_target,
        "val_split": args.val_split,
        "train_tasks": sorted(train_tasks),
        "heldout_tasks": sorted(heldout_tasks),
        "policies": {},
    }

    for pol in policies:
        d = os.path.join(args.data_root, pol, "iql_data")
        if not os.path.isdir(d):
            print(f"\n  [{pol}] ERROR: {d} not found, skipping")
            continue

        eps, heldout_files, other_files = load_pkls(d, train_tasks, heldout_tasks)
        print(f"\n  [{pol}] loaded {len(eps)} eps "
              f"(filtered {len(heldout_files)} heldout, {len(other_files)} other)")

        # HARD ASSERTION — no heldout episodes should be in the kept set
        for ep in eps:
            assert ep["task_id"] not in heldout_tasks, \
                f"HELDOUT LEAK: {pol} ep task_id={ep['task_id']} in kept set"
            assert ep["task_id"] in train_tasks, \
                f"Unexpected task_id={ep['task_id']} in kept set for {pol}"

        # Sanity check the reward/done labels
        if not args.skip_sanity:
            issues = label_sanity_check(eps, pol)
            if issues:
                print(f"  [{pol}] LABEL ISSUES ({len(issues)}):")
                for line in issues[:8]:
                    print(f"     - {line}")
                if len(issues) > 8:
                    print(f"     - ... and {len(issues)-8} more")
                print(f"  [{pol}]   (set --skip-sanity to ignore, but you should fix the data)")
            else:
                print(f"  [{pol}] label sanity: OK")

        # Stratified per-task split
        tr_pool, va_pool = stratified_split_per_task(eps, args.val_split, args.seed)
        print(f"  [{pol}] after split: train_pool={len(tr_pool)}, val={len(va_pool)}")

        # Balance train side if asked
        if args.balance:
            tr_eps, tr_audit = balance_train(tr_pool, args.balance_target, args.seed)
        else:
            tr_eps = tr_pool
            tr_audit = []
            by_task = defaultdict(lambda: {"succ": 0, "fail": 0})
            for ep in tr_pool:
                by_task[ep["task_id"]]["succ" if ep["episode_success"] else "fail"] += 1
            for tid in sorted(by_task):
                tr_audit.append({
                    "task_id": tid,
                    "succ_unique": by_task[tid]["succ"],
                    "succ_kept":   by_task[tid]["succ"],
                    "fail_unique": by_task[tid]["fail"],
                    "fail_kept":   by_task[tid]["fail"],
                    "fail_deficit": 0,
                    "total_kept": by_task[tid]["succ"] + by_task[tid]["fail"],
                })

        # Val audit
        va_by = defaultdict(lambda: {"succ": 0, "fail": 0})
        for ep in va_pool:
            va_by[ep["task_id"]]["succ" if ep["episode_success"] else "fail"] += 1
        va_audit = [{"task_id": tid, "succ": va_by[tid]["succ"], "fail": va_by[tid]["fail"]}
                    for tid in sorted(va_by)]

        print_train_audit(pol, tr_audit)
        print_val_audit(pol, va_audit)

        # Chunks
        tr_chunks = chunks_from_individual(tr_eps, pol)
        va_chunks = chunks_from_individual(va_pool, pol)

        succ_chunks_tr = sum(1 for c in tr_chunks if c["episode_success"])
        fail_chunks_tr = len(tr_chunks) - succ_chunks_tr
        succ_chunks_va = sum(1 for c in va_chunks if c["episode_success"])
        fail_chunks_va = len(va_chunks) - succ_chunks_va
        print(f"\n  [{pol}] chunks  train: {len(tr_chunks)}  "
              f"(succ={succ_chunks_tr}, fail={fail_chunks_tr})")
        print(f"  [{pol}] chunks  val:   {len(va_chunks)}  "
              f"(succ={succ_chunks_va}, fail={fail_chunks_va})")

        all_train_chunks.extend(tr_chunks)
        all_val_chunks.extend(va_chunks)

        audit["policies"][pol] = {
            "loaded_eps": len(eps),
            "train_pool_eps": len(tr_pool),
            "val_eps": len(va_pool),
            "train_final_eps": len(tr_eps),
            "train_per_task": tr_audit,
            "val_per_task": va_audit,
            "train_chunks_total": len(tr_chunks),
            "val_chunks_total": len(va_chunks),
        }

    # Write outputs
    tr_out = os.path.join(data_dir, "train_chunks.pkl")
    va_out = os.path.join(data_dir, "eval_chunks.pkl")  # name retained for downstream tools
    with open(tr_out, "wb") as f:
        pickle.dump(all_train_chunks, f)
    with open(va_out, "wb") as f:
        pickle.dump(all_val_chunks, f)

    audit_path = os.path.join(expt_root, "audit.json")
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)

    cfg = {
        "policies": policies,
        "train_tasks": sorted(train_tasks),
        "heldout_tasks": sorted(heldout_tasks),
        "val_split": args.val_split,
        "balance": args.balance,
        "balance_target": args.balance_target,
        "chunk_sizes": {p: CHUNK_SIZES.get(p, 1) for p in policies},
        "seed": args.seed,
        "n_train_chunks": len(all_train_chunks),
        "n_val_chunks": len(all_val_chunks),
        "data_root": args.data_root,
    }
    with open(os.path.join(expt_root, "expt_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n╔══════════════════════════════════════════════════════════════════════╗")
    print(f"║  Done.")
    print(f"║   train chunks : {len(all_train_chunks):>7}  →  {tr_out}")
    print(f"║   val chunks   : {len(all_val_chunks):>7}  →  {va_out}")
    print(f"║   audit        : {audit_path}")
    print(f"║   config       : {os.path.join(expt_root, 'expt_config.json')}")
    print(f"╚══════════════════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
