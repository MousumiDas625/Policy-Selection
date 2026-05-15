"""
eval_online.py — Online evaluation: IQL model selects VLAs in live LIBERO env.

Flow per decision point:
  1. Get current obs from env (agentview 256x256)
  2. Encode with Qwen: z = Qwen(obs, instruction)
  3. Q(s,pi0), Q(s,openvla), Q(s,molmoact) → pick argmax
  4. Call that VLA server, get action chunk
  5. Execute full chunk in env
  6. Repeat from 1

Needs: all VLA servers running + 1 GPU for Qwen + IQL

Usage:
  CUDA_VISIBLE_DEVICES=0 python eval_online.py \
      --checkpoint /path/to/best.pt \
      --policies pi0,openvla,molmoact \
      --task-suite-name libero_spatial \
      --eval-tasks 5 \
      --num-trials 50 \
      --pi0-host 0.0.0.0 --pi0-port 8010 \
      --openvla-host 0.0.0.0 --openvla-port 8001 \
      --molmoact-host 0.0.0.0 --molmoact-port 8002 \
      --output-dir /path/to/expt1/online_eval
"""

import argparse
import collections
import csv
import gc
import logging
import math
import os
import pathlib
import pickle
import time

import imageio
import numpy as np
import requests
import torch
import tqdm
from PIL import Image

from config import CHUNK_SIZES, MAX_STEPS, MOLMO_UNNORM

# ═══════════════════════════════════════════════════════════════════
#  IMPORTS — conditional on whether inside apptainer or not
# ═══════════════════════════════════════════════════════════════════

try:
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy as _wsc
    HAS_OPENPI = True
except ImportError:
    HAS_OPENPI = False
    logging.warning("openpi_client not available — pi0/pi05 queries will fail")

try:
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    HAS_LIBERO = True
except ImportError:
    HAS_LIBERO = False
    logging.warning("LIBERO not available — cannot run online eval")


DUMMY_ACTION = [0.0] * 6 + [-1.0]
ENV_RES = 256


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════

def _get_env(task, res, seed):
    desc = task.language
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
    env.seed(seed)
    return env, desc


def _quat2aa(q):
    if q[3] > 1: q[3] = 1
    elif q[3] < -1: q[3] = -1
    den = np.sqrt(1 - q[3]**2)
    return np.zeros(3) if math.isclose(den, 0) else (q[:3] * 2 * math.acos(q[3])) / den


def _prep(obs, sz=224):
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), sz, sz))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), sz, sz))
    return img, wrist


def norm_grip(a):
    a = np.array(a, dtype=np.float32)
    a[..., -1] = np.sign(2 * a[..., -1] - 1)
    return a


def inv_grip(a):
    a = np.array(a, dtype=np.float32)
    a[..., -1] *= -1
    return a


# ═══════════════════════════════════════════════════════════════════
#  VLA QUERY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def query_pi0(client, img, wrist, obs, desc, task_id, ep, t, replan=5):
    chunk = client.infer({
        "observation/image": img, "observation/wrist_image": wrist,
        "observation/state": np.concatenate((
            obs["robot0_eef_pos"], _quat2aa(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"])),
        "prompt": str(desc), "run/run_note": "iql_eval",
        "run/task_id": task_id, "run/episode_idx": ep, "run/timestep": t,
    })["actions"]
    return list(chunk[:replan])


def query_openvla(url, img, desc, unnorm_key):
    resp = requests.post(f"{url}/act", json={
        "image": img, "instruction": str(desc), "unnorm_key": unnorm_key,
    }, timeout=120)
    resp.raise_for_status()
    a = np.array(resp.json(), dtype=np.float32)
    return [inv_grip(norm_grip(a))]


def query_molmoact(url, img, wrist, desc, unnorm_key):
    resp = requests.post(f"{url}/act", json={
        "image": img.tolist(), "wrist_image": wrist.tolist(),
        "instruction": str(desc), "unnorm_key": unnorm_key, "save_trace": False,
    }, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    actions = []
    for raw in data["actions"]:
        a = np.array(raw, dtype=np.float32)
        actions.append(inv_grip(norm_grip(a)))
    return actions


# ═══════════════════════════════════════════════════════════════════
#  QWEN ENCODER (for online inference)
# ═══════════════════════════════════════════════════════════════════

class QwenEncoder:
    def __init__(self, model_name, device, cache_dir):
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(
                model_name, cache_dir=cache_dir, trust_remote_code=True)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, cache_dir=cache_dir,
                torch_dtype=torch.bfloat16, trust_remote_code=True,
            ).to(device).eval()
            self.hdim = getattr(self.model.config, "hidden_size", None) or self.model.config.text_config.hidden_size
            self.mode = "vl"
            print(f"  Qwen VL loaded, dim={self.hdim}")
        except Exception as e:
            print(f"  VL failed ({e}), using text-only")
            fallback = model_name.replace("VL-", "")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.processor = AutoTokenizer.from_pretrained(
                fallback, cache_dir=cache_dir, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                fallback, cache_dir=cache_dir,
                torch_dtype=torch.bfloat16, trust_remote_code=True,
            ).to(device).eval()
            self.hdim = getattr(self.model.config, "hidden_size", None) or self.model.config.text_config.hidden_size
            self.mode = "text"

        self.device = device

    @torch.no_grad()
    def encode(self, img_np, instruction):
        """Encode one (image, instruction) → (1, hidden_dim) tensor."""
        if self.mode == "vl":
            from qwen_vl_utils import process_vision_info
            pil = Image.fromarray(img_np.astype(np.uint8))
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": f"Robot task: {instruction}. Describe the workspace."},
            ]}]
            text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            im_in, vid_in = process_vision_info(msgs)
            inputs = self.processor(text=[text], images=im_in, videos=vid_in,
                                    padding=True, return_tensors="pt").to(self.device)
            out = self.model(**inputs, output_hidden_states=True)
            emb = out.hidden_states[-1][:, -1, :].float()
            del inputs, out
            torch.cuda.empty_cache()
            return emb  # (1, D) on device
        else:
            mean_rgb = img_np.mean(axis=(0, 1))
            text = (f"Robot task: {instruction}. "
                    f"Image RGB=({mean_rgb[0]:.0f},{mean_rgb[1]:.0f},{mean_rgb[2]:.0f})")
            inputs = self.processor(text, return_tensors="pt", truncation=True, max_length=256).to(self.device)
            out = self.model(**inputs, output_hidden_states=True)
            emb = out.hidden_states[-1][:, -1, :].float()
            del inputs, out
            return emb


# ═══════════════════════════════════════════════════════════════════
#  LOAD IQL MODEL
# ═══════════════════════════════════════════════════════════════════

def load_iql(ckpt_path, device):
    from train_iql import IQLModel
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = IQLModel(
        state_dim=ckpt["state_dim"],
        n_policies=len(ckpt["policies"]),
        policy_names=ckpt["policies"],
        policy_embed_dim=ckpt["policy_embed_dim"],
        hidden=ckpt["hidden"],
        dropout=ckpt.get("dropout", 0.1),
        freeze_policy_emb=ckpt.get("freeze_policy_emb", False),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    print(f"  IQL loaded: {ckpt['policies']}, epoch={ckpt.get('epoch','?')}")
    return model


# ═══════════════════════════════════════════════════════════════════
#  ONLINE EVAL
# ═══════════════════════════════════════════════════════════════════

def run_online(args):
    assert HAS_LIBERO, "LIBERO not available"
    assert HAS_OPENPI, "openpi_client not available"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load IQL + Qwen
    print("Loading IQL model...")
    iql = load_iql(args.checkpoint, device)
    policies = iql.policy_names

    print("Loading Qwen encoder...")
    qwen = QwenEncoder(args.qwen_model, device, args.cache_dir)
    assert qwen.hdim == iql.state_dim, f"Dim mismatch: Qwen={qwen.hdim} vs IQL={iql.state_dim}"

    # VLA clients
    clients = {}
    if "pi0" in policies:
        clients["pi0"] = _wsc.WebsocketClientPolicy(args.pi0_host, args.pi0_port)
    if "pi05" in policies:
        clients["pi05"] = _wsc.WebsocketClientPolicy(args.pi05_host, args.pi05_port)
    if "openvla" in policies:
        clients["openvla"] = f"http://{args.openvla_host}:{args.openvla_port}"
    if "molmoact" in policies:
        clients["molmoact"] = f"http://{args.molmoact_host}:{args.molmoact_port}"

    # Output
    os.makedirs(args.output_dir, exist_ok=True)
    iql_dir = os.path.join(args.output_dir, "iql_data")
    vid_dir = os.path.join(args.output_dir, "videos")
    os.makedirs(iql_dir, exist_ok=True)
    os.makedirs(vid_dir, exist_ok=True)

    ep_csv = open(os.path.join(args.output_dir, "episode_log.csv"), "w", newline="")
    ep_w = csv.DictWriter(ep_csv, fieldnames=[
        "task_id", "ep", "success", "steps", "wall_s",
        "decisions", "preferred"] + [f"{p}_steps" for p in policies])
    ep_w.writeheader()

    # LIBERO
    bm = benchmark.get_benchmark_dict()
    suite = bm[args.task_suite_name]()
    max_steps = MAX_STEPS[args.task_suite_name]
    wait = args.num_steps_wait
    ovla_unnorm = args.task_suite_name
    molmo_unnorm = MOLMO_UNNORM.get(args.task_suite_name, "")
    eval_tasks = [int(t) for t in args.eval_tasks.split(",")]

    total_ep, total_succ = 0, 0

    for tid in eval_tasks:
        task = suite.get_task(tid)
        inits = suite.get_task_init_states(tid)
        env, desc = _get_env(task, ENV_RES, args.seed)
        logging.info(f"Task {tid}: {desc}")

        for ep_idx in tqdm.tqdm(range(args.num_trials), desc=f"Task {tid}"):
            env.reset()
            obs = env.set_init_state(inits[ep_idx % len(inits)])
            done = False; t = 0

            iql_obs, iql_act, iql_rew, iql_done = [], [], [], []
            iql_wt, iql_pol = [], []
            replay, pol_counts = [], {p: 0 for p in policies}
            buf = collections.deque()
            cur_pol = None
            n_decisions = 0
            ep_start = time.time()

            while t < max_steps + wait:
                if t < wait:
                    obs, _, done, _ = env.step(DUMMY_ACTION)
                    t += 1; continue

                img, wrist = _prep(obs)
                replay.append(img)

                if not buf:
                    # ── IQL decision point ──
                    raw_obs = obs["agentview_image"][::-1, ::-1]
                    z = qwen.encode(raw_obs, desc)  # (1, D) on device

                    with torch.no_grad():
                        q_all = iql.all_q(z)  # (1, n_pol)
                    best_idx = q_all.argmax(-1).item()
                    cur_pol = policies[best_idx]
                    n_decisions += 1

                    q_str = {policies[i]: round(q_all[0, i].item(), 3) for i in range(len(policies))}
                    logging.info(f"  t={t} Q={q_str} → {cur_pol}")

                    infer_start = time.time()
                    try:
                        if cur_pol in ("pi0", "pi05"):
                            cl = clients[cur_pol]
                            rs = CHUNK_SIZES.get(cur_pol, 5)
                            chunk = query_pi0(cl, img, wrist, obs, desc, tid, ep_idx, t, rs)
                        elif cur_pol == "openvla":
                            chunk = query_openvla(clients["openvla"], img, desc, ovla_unnorm)
                        elif cur_pol == "molmoact":
                            chunk = query_molmoact(clients["molmoact"], img, wrist, desc, molmo_unnorm)
                        else:
                            raise ValueError(f"Unknown policy: {cur_pol}")
                    except Exception as e:
                        logging.error(f"Query {cur_pol} failed: {e}")
                        break
                    infer_time = time.time() - infer_start
                    buf.extend(chunk)
                else:
                    infer_time = 0.0

                action = buf.popleft()
                if not isinstance(action, np.ndarray):
                    action = np.array(action, dtype=np.float32)

                # Record
                iql_obs.append({
                    "agentview": obs["agentview_image"][::-1, ::-1].copy(),
                    "wrist": obs["robot0_eye_in_hand_image"][::-1, ::-1].copy(),
                })
                iql_act.append(action.copy())
                iql_wt.append(infer_time)
                iql_pol.append(cur_pol)
                pol_counts[cur_pol] += 1

                obs, _, done, _ = env.step(action.tolist())
                iql_rew.append(1.0 if done else 0.0)
                iql_done.append(bool(done))

                if done:
                    total_succ += 1; break
                t += 1

            total_ep += 1
            wall = time.time() - ep_start
            n_steps = len(iql_act)

            # Video
            stem = f"task{tid}_ep{ep_idx}_succ{int(done)}"
            if replay:
                imageio.mimwrite(os.path.join(vid_dir, f"{stem}.mp4"),
                                 [np.asarray(x) for x in replay], fps=30)

            # Pickle
            pickle.dump({
                "task_suite_name": args.task_suite_name,
                "task_id": tid, "episode_idx": ep_idx,
                "language_instruction": desc,
                "episode_success": bool(done),
                "selection_mode": "iql",
                "policies_used": policies,
                "seed": args.seed,
                "observations": iql_obs, "actions": iql_act,
                "rewards": iql_rew, "dones": iql_done,
                "wall_clock_per_step": iql_wt,
                "policy_per_step": iql_pol,
                "total_steps": n_steps,
                "total_wall_time_s": wall,
                "num_steps_wait": wait,
                "max_steps": max_steps,
                "policy_step_counts": pol_counts,
                "n_decisions": n_decisions,
                "checkpoint": args.checkpoint,
            }, open(os.path.join(iql_dir, f"{stem}.pkl"), "wb"))

            # CSV
            row = {"task_id": tid, "ep": ep_idx, "success": int(done),
                   "steps": n_steps, "wall_s": f"{wall:.1f}",
                   "decisions": n_decisions, "preferred": ""}
            for p in policies:
                row[f"{p}_steps"] = pol_counts[p]
            ep_w.writerow(row); ep_csv.flush()

            counts_s = " ".join(f"{p}={pol_counts[p]}" for p in policies)
            logging.info(f"[t{tid} e{ep_idx}] succ={done} {counts_s} "
                         f"dec={n_decisions} | {total_succ}/{total_ep}")

        env.close()

    ep_csv.close()
    print(f"\nDone: {total_succ}/{total_ep} ({100*total_succ/max(total_ep,1):.1f}%)")
    print(f"Output: {args.output_dir}")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument("--policies", default="pi0,openvla,molmoact")
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument("--eval-tasks", default="5", help="Comma-separated task IDs")
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)

    # VLA servers
    parser.add_argument("--pi0-host", default="0.0.0.0")
    parser.add_argument("--pi0-port", type=int, default=8010)
    parser.add_argument("--pi05-host", default="0.0.0.0")
    parser.add_argument("--pi05-port", type=int, default=8020)
    parser.add_argument("--openvla-host", default="0.0.0.0")
    parser.add_argument("--openvla-port", type=int, default=8001)
    parser.add_argument("--molmoact-host", default="0.0.0.0")
    parser.add_argument("--molmoact-port", type=int, default=8002)

    # Qwen
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--cache-dir",
                        default="/project2/jessetho_1732/mousumid/PolicySel/hf_cache")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_online(args)


if __name__ == "__main__":
    main()
