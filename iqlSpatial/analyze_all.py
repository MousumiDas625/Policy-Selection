"""
analyze_all.py — Visualize and compare all rollout data

Analyzes:
  1. Individual VLA rollouts (pi0, openvla, molmoact, pi05)
  2. Random policy selection rollouts (allThreeRandom, allPiRandom)
  3. Step count comparison across all methods
  4. Chunk pattern verification for random selection

Usage:
  cd /project2/jessetho_1732/mousumid/PolicySel/iqlSpatial
  source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
  conda activate safe-molmoact
  export PYTHONNOUSERSITE=1

  python analyze_all.py                          # all tasks
  python analyze_all.py --tasks 0                # just task 0
  python analyze_all.py --tasks 0,1              # tasks 0 and 1
  python analyze_all.py --inspect allThreeRandom:0:0   # inspect one episode
"""

import argparse
import glob
import os
import pickle
from collections import defaultdict

import numpy as np

BASE = "/project2/jessetho_1732/mousumid/PolicySel/iqlSpatial"

INDIVIDUAL = ["pi0", "openvla", "molmoact", "pi05"]
RANDOM_DIRS = ["allThreeRandom", "allPiRandom"]


def load_pkls(data_dir, task_filter=None):
    eps = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.pkl"))):
        d = pickle.load(open(f, "rb"))
        if task_filter is not None and d.get("task_id") not in task_filter:
            continue
        d["_file"] = os.path.basename(f)
        eps.append(d)
    return eps


def sep(title, c="═", w=80):
    print(f"\n{c*w}\n  {title}\n{c*w}")


# ───────────────────────────────────────────────────────────────────────
#  1. INDIVIDUAL VLA STATS
# ───────────────────────────────────────────────────────────────────────

def show_individual(task_filter):
    sep("INDIVIDUAL VLA ROLLOUTS")
    all_data = {}
    for vla in INDIVIDUAL:
        d = os.path.join(BASE, vla, "iql_data")
        if not os.path.isdir(d):
            print(f"\n  {vla}: NOT FOUND")
            continue
        eps = load_pkls(d, task_filter)
        if not eps:
            print(f"\n  {vla}: no matching episodes")
            continue
        all_data[vla] = eps

        # First episode structure
        e0 = eps[0]
        print(f"\n{'─'*70}\n  {vla.upper()} — {len(eps)} episodes\n{'─'*70}")
        print(f"  Pickle keys: {sorted(e0.keys())}")
        print(f"  total_steps present: {'total_steps' in e0}")
        if e0.get("total_steps", 0) > 0:
            print(f"  obs shape: {e0['observations'][0]['agentview'].shape}")
            print(f"  action shape: {e0['actions'][0].shape}")

        # Per-task table
        ts = defaultdict(lambda: {"s": 0, "f": 0, "steps_s": [], "steps_f": [], "wt": []})
        for e in eps:
            tid = e["task_id"]
            if e["episode_success"]:
                ts[tid]["s"] += 1
                ts[tid]["steps_s"].append(e["total_steps"])
            else:
                ts[tid]["f"] += 1
                ts[tid]["steps_f"].append(e["total_steps"])
            ts[tid]["wt"].append(e.get("total_wall_time_s", 0))

        print(f"\n  {'Task':>4} | {'Succ':>4}/{' Tot':>4} | {'Rate':>6} | "
              f"{'Steps(succ) mean±std':>22} | {'Steps(fail) mean±std':>22} | {'WallTime(s)':>12}")
        print(f"  {'─'*90}")
        ts_total, tf_total = 0, 0
        for tid in sorted(ts):
            t = ts[tid]
            tot = t["s"] + t["f"]
            rate = 100 * t["s"] / tot
            ts_total += t["s"]; tf_total += t["f"]
            ss = np.array(t["steps_s"]) if t["steps_s"] else np.array([0])
            sf = np.array(t["steps_f"]) if t["steps_f"] else np.array([0])
            wt = np.array(t["wt"])
            s_str = f"{ss.mean():.1f}±{ss.std():.1f}" if t["steps_s"] else "N/A"
            f_str = f"{sf.mean():.1f}±{sf.std():.1f}" if t["steps_f"] else "N/A"
            print(f"  {tid:>4} | {t['s']:>4}/{tot:>4} | {rate:>5.1f}% | "
                  f"{s_str:>22} | {f_str:>22} | {wt.mean():>10.1f}s")
        tot_all = ts_total + tf_total
        if tot_all:
            print(f"  {'ALL':>4} | {ts_total:>4}/{tot_all:>4} | "
                  f"{100*ts_total/tot_all:>5.1f}% |")
    return all_data


# ───────────────────────────────────────────────────────────────────────
#  2. RANDOM POLICY SELECTION STATS
# ───────────────────────────────────────────────────────────────────────

def show_random(task_filter):
    sep("RANDOM POLICY SELECTION ROLLOUTS")
    all_rand = {}
    for rd in RANDOM_DIRS:
        d = os.path.join(BASE, rd, "iql_data")
        if not os.path.isdir(d) or not os.listdir(d):
            print(f"\n  {rd}: NOT FOUND or EMPTY")
            continue
        eps = load_pkls(d, task_filter)
        if not eps:
            print(f"\n  {rd}: no matching episodes")
            continue
        all_rand[rd] = eps

        e0 = eps[0]
        print(f"\n{'─'*70}\n  {rd.upper()} — {len(eps)} episodes\n{'─'*70}")
        print(f"  Pickle keys: {sorted(e0.keys())}")
        print(f"  selection_mode: {e0.get('selection_mode')}")
        print(f"  policies_used: {e0.get('policies_used')}")
        print(f"  policy_step_counts: {e0.get('policy_step_counts')}")

        # Chunk pattern
        pps = e0.get("policy_per_step", [])
        if pps:
            chunks = []
            cur, cnt = pps[0], 1
            for p in pps[1:]:
                if p == cur:
                    cnt += 1
                else:
                    chunks.append(f"{cur}×{cnt}")
                    cur, cnt = p, 1
            chunks.append(f"{cur}×{cnt}")
            print(f"  Ep0 chunk pattern (first 15): {chunks[:15]}")
            print(f"  Ep0 total decision points: {len(chunks)}")

        policies = e0.get("policies_used", [])

        # Per-task table
        ts = defaultdict(lambda: {"s": 0, "f": 0, "steps": [], "pol": defaultdict(list), "wt": []})
        for e in eps:
            tid = e["task_id"]
            if e["episode_success"]:
                ts[tid]["s"] += 1
            else:
                ts[tid]["f"] += 1
            ts[tid]["steps"].append(e["total_steps"])
            ts[tid]["wt"].append(e.get("total_wall_time_s", 0))
            psc = e.get("policy_step_counts", {})
            for p in policies:
                ts[tid]["pol"][p].append(psc.get(p, 0))

        # Header
        pol_hdr = " | ".join(f"{p:>10}" for p in policies)
        print(f"\n  {'Task':>4} | {'Succ':>4}/{' Tot':>4} | {'Rate':>6} | "
              f"{'Steps mean±std':>18} | {pol_hdr} | {'WallTime':>10}")
        print(f"  {'─'*(60 + 13*len(policies))}")

        ts_total, tf_total = 0, 0
        for tid in sorted(ts):
            t = ts[tid]
            tot = t["s"] + t["f"]
            rate = 100 * t["s"] / tot
            ts_total += t["s"]; tf_total += t["f"]
            steps = np.array(t["steps"])
            wt = np.array(t["wt"])
            pol_str = " | ".join(
                f"{np.mean(t['pol'][p]):>5.1f}±{np.std(t['pol'][p]):>3.1f}" for p in policies
            )
            print(f"  {tid:>4} | {t['s']:>4}/{tot:>4} | {rate:>5.1f}% | "
                  f"{steps.mean():>7.1f}±{steps.std():>6.1f} | {pol_str} | {wt.mean():>8.1f}s")
        tot_all = ts_total + tf_total
        if tot_all:
            print(f"  {'ALL':>4} | {ts_total:>4}/{tot_all:>4} | "
                  f"{100*ts_total/tot_all:>5.1f}% |")

        # Chunk size verification
        print(f"\n  Chunk size verification (across all episodes):")
        chunk_sizes = defaultdict(list)
        for e in eps:
            pps = e.get("policy_per_step", [])
            if not pps:
                continue
            cur, cnt = pps[0], 1
            for p in pps[1:]:
                if p == cur:
                    cnt += 1
                else:
                    chunk_sizes[cur].append(cnt)
                    cur, cnt = p, 1
            chunk_sizes[cur].append(cnt)
        for p in sorted(chunk_sizes):
            sizes = np.array(chunk_sizes[p])
            print(f"    {p}: mean={sizes.mean():.1f} median={np.median(sizes):.0f} "
                  f"mode={np.bincount(sizes).argmax()} min={sizes.min()} max={sizes.max()} "
                  f"(n={len(sizes)} chunks)")

    return all_rand


# ───────────────────────────────────────────────────────────────────────
#  3. COMPARISON: STEPS TO COMPLETE (successful only)
# ───────────────────────────────────────────────────────────────────────

def compare_steps(task_filter):
    sep("STEP COUNT COMPARISON — successful episodes only")

    all_methods = {}

    # Individual
    for vla in INDIVIDUAL:
        d = os.path.join(BASE, vla, "iql_data")
        if os.path.isdir(d):
            eps = load_pkls(d, task_filter)
            if eps:
                all_methods[vla] = eps

    # Random
    for rd in RANDOM_DIRS:
        d = os.path.join(BASE, rd, "iql_data")
        if os.path.isdir(d) and os.listdir(d):
            eps = load_pkls(d, task_filter)
            if eps:
                all_methods[f"rand:{rd}"] = eps

    if not all_methods:
        print("  No data.")
        return

    task_ids = set()
    for eps in all_methods.values():
        for e in eps:
            if task_filter is None or e["task_id"] in task_filter:
                task_ids.add(e["task_id"])

    for tid in sorted(task_ids):
        print(f"\n  Task {tid}:")
        print(f"  {'Method':<25} | {'N_succ':>6} | {'N_fail':>6} | "
              f"{'Steps mean':>10} | {'±std':>6} | {'Min':>4} | {'Max':>4} | {'WallTime':>10}")
        print(f"  {'─'*90}")

        for name in sorted(all_methods):
            succ_steps, fail_n, wts = [], 0, []
            for e in all_methods[name]:
                if e["task_id"] != tid:
                    continue
                if e["episode_success"]:
                    succ_steps.append(e["total_steps"])
                    wts.append(e.get("total_wall_time_s", 0))
                else:
                    fail_n += 1
            if succ_steps:
                a = np.array(succ_steps)
                wt = np.mean(wts)
                print(f"  {name:<25} | {len(succ_steps):>6} | {fail_n:>6} | "
                      f"{a.mean():>10.1f} | {a.std():>5.1f} | {a.min():>4} | {a.max():>4} | {wt:>8.1f}s")
            else:
                print(f"  {name:<25} | {0:>6} | {fail_n:>6} | "
                      f"{'N/A':>10} | {'':>6} | {'':>4} | {'':>4} |")


# ───────────────────────────────────────────────────────────────────────
#  4. INSPECT ONE EPISODE
# ───────────────────────────────────────────────────────────────────────

def inspect(spec):
    parts = spec.split(":")
    src = parts[0]
    tid = int(parts[1]) if len(parts) > 1 else 0
    eidx = int(parts[2]) if len(parts) > 2 else 0

    # Find the pkl
    for check in [
        os.path.join(BASE, src, "iql_data"),
        os.path.join(BASE, src),
    ]:
        pat = os.path.join(check, f"task{tid}_ep{eidx}_succ*.pkl")
        matches = glob.glob(pat)
        if matches:
            break
    else:
        print(f"Not found: {spec}")
        return

    f = matches[0]
    d = pickle.load(open(f, "rb"))

    sep(f"EPISODE DETAIL: {os.path.basename(f)}")
    print(f"  File: {f}")

    for k in sorted(d.keys()):
        v = d[k]
        if k in ("observations", "actions", "rewards", "dones", "wall_clock_per_step", "policy_per_step"):
            if isinstance(v, list):
                extra = ""
                if v and hasattr(v[0], "shape"):
                    extra = f" of shape {v[0].shape}"
                elif v and isinstance(v[0], dict):
                    extra = f" of dict keys={list(v[0].keys())}"
                print(f"  {k}: list[{len(v)}]{extra}")
            else:
                print(f"  {k}: {type(v).__name__}")
        elif k == "_file":
            continue
        else:
            val_str = str(v)
            if len(val_str) > 100:
                val_str = val_str[:100] + "..."
            print(f"  {k}: {val_str}")

    # Wall clock
    wc = d.get("wall_clock_per_step", [])
    if wc:
        wa = np.array(wc)
        print(f"\n  Wall clock: total={sum(wc):.1f}s  mean/step={wa.mean():.3f}s  "
              f"inference_steps(>0)={sum(1 for w in wc if w > 0)}")
        print(f"  First 10: {[round(w, 3) for w in wc[:10]]}")

    # Policy switching
    pps = d.get("policy_per_step")
    if pps:
        chunks = []
        cur, cnt = pps[0], 1
        for p in pps[1:]:
            if p == cur:
                cnt += 1
            else:
                chunks.append((cur, cnt))
                cur, cnt = p, 1
        chunks.append((cur, cnt))

        print(f"\n  Policy switching: {len(chunks)} decision points")
        print(f"  Full chunk sequence: {[f'{c[0]}×{c[1]}' for c in chunks]}")
        for pol in set(pps):
            sizes = [c[1] for c in chunks if c[0] == pol]
            print(f"    {pol}: {len(sizes)} selections, chunk sizes={sizes}")


# ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, default=None)
    parser.add_argument("--inspect", type=str, default=None,
                        help="e.g. 'pi0:0:0' or 'allThreeRandom:0:0'")
    args = parser.parse_args()

    tf = None
    if args.tasks:
        tf = set(int(t) for t in args.tasks.split(","))

    if args.inspect:
        inspect(args.inspect)
        return

    show_individual(tf)
    show_random(tf)
    compare_steps(tf)


if __name__ == "__main__":
    main()
