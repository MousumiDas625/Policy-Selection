#!/usr/bin/env python3
"""
eval_random_baseline.py
Random policy selection between pi0 and pi05 in LIBERO simulation.
Logic adapted from main_random_chunked.py.

Usage:
    python eval_random_baseline.py \
        --eval-tasks 0,1,2,3,4,5,6,7,8,9 \
        --num-trials 50 \
        --pi0-host 0.0.0.0  --pi0-port 8010 \
        --pi05-host 0.0.0.0 --pi05-port 8020 \
        --output-dir ./random_baseline_results
"""

import argparse
import collections
import csv
import logging
import math
import pathlib
import pickle
import random
import time

import imageio
import numpy as np
import tqdm

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _wsc

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger(__name__)

DUMMY_ACTION = [0.0] * 6 + [-1.0]
RESIZE       = 224
MAX_STEPS    = {"libero_spatial": 220, "libero_object": 280,
                "libero_goal": 300, "libero_10": 520, "libero_90": 400}


# ── Helpers (copied directly from main_random_chunked.py) ─────────────────

def _quat2axisangle(quat):
    if quat[3] > 1.0:  quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _prep_images(obs):
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), RESIZE, RESIZE))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), RESIZE, RESIZE))
    return img, wrist


def _get_libero_env(task, seed):
    task_bddl = (pathlib.Path(get_libero_path("bddl_files"))
                 / task.problem_folder / task.bddl_file)
    env = OffScreenRenderEnv(**{
        "bddl_file_name": str(task_bddl),
        "camera_heights": 256,
        "camera_widths":  256,
    })
    env.seed(seed)
    return env, task.language


def query_pi0(client, obs, task_desc, task_id, ep_idx, t, replan_steps=5):
    img, wrist = _prep_images(obs)
    result = client.infer({
        "observation/image":       img,
        "observation/wrist_image": wrist,
        "observation/state": np.concatenate((
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )),
        "prompt":            str(task_desc),
        "run/task_id":       task_id,
        "run/episode_idx":   ep_idx,
        "run/timestep":      t,
    })
    return list(result["actions"][:replan_steps])


def warmup(client, name):
    log.info(f"Warming up {name} (JIT compile may take 2-5 min)...")
    dummy_obs = {
        "agentview_image":          np.zeros((256, 256, 3), np.uint8),
        "robot0_eye_in_hand_image": np.zeros((256, 256, 3), np.uint8),
        "robot0_eef_pos":   np.zeros(3),
        "robot0_eef_quat":  np.array([0, 0, 0, 1], dtype=np.float64),
        "robot0_gripper_qpos": np.zeros(2),
    }
    try:
        r = client.infer({
            "observation/image":       np.zeros((RESIZE, RESIZE, 3), np.uint8),
            "observation/wrist_image": np.zeros((RESIZE, RESIZE, 3), np.uint8),
            "observation/state":       np.zeros(8),
            "prompt": "warmup",
        })
        log.info(f"  {name} ready — got {len(r.get('actions', []))} actions")
    except Exception as e:
        log.warning(f"  {name} warmup warning: {e}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-tasks",   default="5,9",
                        help="Comma-separated task IDs e.g. '0,1,2,5,9'")
    parser.add_argument("--num-trials",   type=int, default=50)
    parser.add_argument("--task-suite",   default="libero_spatial")
    parser.add_argument("--replan-steps", type=int, default=5,
                        help="Actions per chunk before re-selecting policy")
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--pi0-host",  default="0.0.0.0")
    parser.add_argument("--pi0-port",  type=int, default=8010)
    parser.add_argument("--pi05-host", default="0.0.0.0")
    parser.add_argument("--pi05-port", type=int, default=8020)
    parser.add_argument("--pi0-prob",  type=float, default=0.5,
                        help="Probability of selecting pi0 at each decision. "
                             "0.5=random, 1.0=always pi0, 0.0=always pi05")
    parser.add_argument("--output-dir", default="./random_baseline_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-video", action="store_true",
                        help="Save per-episode mp4 videos")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    task_ids  = [int(t.strip()) for t in args.eval_tasks.split(",")]
    max_steps = MAX_STEPS[args.task_suite]

    out = pathlib.Path(args.output_dir)
    (out / "iql_data").mkdir(parents=True, exist_ok=True)
    if args.save_video:
        (out / "videos").mkdir(exist_ok=True)

    log.info(f"Tasks: {task_ids}  Trials: {args.num_trials}  "
             f"Chunk: {args.replan_steps}  pi0_prob: {args.pi0_prob}")

    # ── Connect + warmup ──────────────────────────────────────────
    log.info(f"Connecting to pi0  {args.pi0_host}:{args.pi0_port} ...")
    pi0_client  = _wsc.WebsocketClientPolicy(args.pi0_host,  args.pi0_port)
    log.info(f"Connecting to pi05 {args.pi05_host}:{args.pi05_port} ...")
    pi05_client = _wsc.WebsocketClientPolicy(args.pi05_host, args.pi05_port)
    warmup(pi0_client,  "pi0")
    warmup(pi05_client, "pi05")

    # ── CSV logs ──────────────────────────────────────────────────
    ep_csv   = open(out / "episode_log.csv",  "w", newline="")
    step_csv = open(out / "step_log.csv",     "w", newline="")
    ep_w   = csv.DictWriter(ep_csv,   fieldnames=[
        "task_id","episode_idx","success","total_steps",
        "pi0_steps","pi05_steps","n_decisions","language_instruction"])
    step_w = csv.DictWriter(step_csv, fieldnames=[
        "task_id","episode_idx","timestep","policy"])
    ep_w.writeheader(); step_w.writeheader()

    # ── Benchmark ─────────────────────────────────────────────────
    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    total_succ, total_ep = 0, 0

    for task_id in task_ids:
        task   = suite.get_task(task_id)
        inits  = suite.get_task_init_states(task_id)
        env, task_desc = _get_libero_env(task, args.seed)
        log.info(f"\nTask {task_id}: {task_desc}")

        for ep_idx in tqdm.tqdm(range(args.num_trials), desc=f"Task {task_id}"):

            # ── Init episode ──────────────────────────────────────
            env.reset()
            obs  = env.set_init_state(inits[ep_idx % len(inits)])
            done = False
            t    = 0

            iql_obs, iql_actions, iql_rewards = [], [], []
            iql_dones, iql_wall, iql_pol      = [], [], []
            policy_steps  = {"pi0": 0, "pi05": 0}
            n_decisions   = 0
            action_buf    = collections.deque()
            cur_pol       = None
            replay_frames = []
            ep_start      = time.time()

            # ── Episode loop (from main_random_chunked.py) ────────
            while t < max_steps + args.num_steps_wait:

                # Wait steps — just step with dummy, no recording
                if t < args.num_steps_wait:
                    obs, _, done, _ = env.step(DUMMY_ACTION)
                    t += 1
                    continue

                img, wrist = _prep_images(obs)
                if args.save_video:
                    replay_frames.append(img)

                # Decision point: pick policy, get action chunk
                if not action_buf:
                    cur_pol = "pi0" if random.random() < args.pi0_prob else "pi05"
                    n_decisions += 1
                    client = pi0_client if cur_pol == "pi0" else pi05_client
                    t_infer = time.time()
                    try:
                        chunk = query_pi0(client, obs, task_desc,
                                          task_id, ep_idx, t,
                                          args.replan_steps)
                        action_buf.extend(chunk)
                    except Exception as e:
                        log.warning(f"  Query failed ({cur_pol}): {e} — dummy")
                        action_buf.append(np.array(DUMMY_ACTION))
                    infer_t = time.time() - t_infer
                else:
                    infer_t = 0.0

                action = action_buf.popleft()
                if not isinstance(action, np.ndarray):
                    action = np.array(action, dtype=np.float32)

                # Record BEFORE stepping (same as main_random_chunked.py)
                iql_obs.append({
                    "agentview": obs["agentview_image"][::-1, ::-1].copy(),
                    "wrist":     obs["robot0_eye_in_hand_image"][::-1, ::-1].copy(),
                })
                iql_actions.append(action.copy())
                iql_wall.append(infer_t)
                iql_pol.append(cur_pol)
                policy_steps[cur_pol] += 1

                step_w.writerow({"task_id": task_id, "episode_idx": ep_idx,
                                  "timestep": t, "policy": cur_pol})
                step_csv.flush()

                # Step env
                obs, _, done, _ = env.step(action.tolist())

                # Record reward/done AFTER stepping
                iql_rewards.append(1.0 if done else 0.0)
                iql_dones.append(bool(done))

                if done:
                    total_succ += 1
                    break
                t += 1

            # ── Save episode ──────────────────────────────────────
            total_ep += 1
            ep_t  = time.time() - ep_start
            n_s   = len(iql_actions)
            stem  = f"task{task_id}_ep{ep_idx}_succ{int(done)}"

            if args.save_video and replay_frames:
                imageio.mimwrite(str(out / "videos" / f"{stem}.mp4"),
                                 replay_frames, fps=30)

            with open(out / "iql_data" / f"{stem}.pkl", "wb") as f:
                pickle.dump({
                    "task_suite_name":    args.task_suite,
                    "task_id":            task_id,
                    "episode_idx":        ep_idx,
                    "language_instruction": task_desc,
                    "episode_success":    bool(done),
                    "selection_mode":     "random_50_50",
                    "policies_used":      ["pi0", "pi05"],
                    "pi0_selection_prob": args.pi0_prob,
                    "seed":               args.seed,
                    "observations":       iql_obs,
                    "actions":            iql_actions,
                    "rewards":            iql_rewards,
                    "dones":              iql_dones,
                    "wall_clock_per_step": iql_wall,
                    "policy_per_step":    iql_pol,
                    "total_steps":        n_s,
                    "total_wall_time_s":  ep_t,
                    "num_steps_wait":     args.num_steps_wait,
                    "max_steps":          max_steps,
                    "policy_step_counts": policy_steps,
                    "n_decisions":        n_decisions,
                }, f)

            ep_w.writerow({
                "task_id": task_id, "episode_idx": ep_idx,
                "success": int(done), "total_steps": n_s,
                "pi0_steps": policy_steps["pi0"],
                "pi05_steps": policy_steps["pi05"],
                "n_decisions": n_decisions,
                "language_instruction": task_desc,
            })
            ep_csv.flush()

            log.info(f"[t{task_id} e{ep_idx}] succ={done}  "
                     f"pi0={policy_steps['pi0']}  pi05={policy_steps['pi05']}  "
                     f"dec={n_decisions}  | {total_succ}/{total_ep}")

        env.close()

    ep_csv.close(); step_csv.close()

    # ── Final summary ─────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  RANDOM BASELINE DONE")
    print(f"  pi0_prob={args.pi0_prob}  chunk={args.replan_steps}")
    print(f"  Total: {total_succ}/{total_ep} = "
          f"{100*total_succ/max(total_ep,1):.1f}%")
    print(f"  Results: {out}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
