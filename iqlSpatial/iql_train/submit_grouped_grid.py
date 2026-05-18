import itertools
import subprocess
import os

# ── Original parameters (unchanged) ───────────────────────────
gammas = [0.85, 0.90, 0.95, 0.99]
taus   = [0.2, 0.3, 0.4, 0.5, 0.7]

architectures = {
    "hidden":           ["512,512", "1024,1024", "2048,2048", "4096,4096"],
    "policy_embed_dim": [32, 64, 128, 256, 512, 1024, 2048],
    "batch_size":       [16, 32, 64, 128, 256, 512, 1024, 2048],
}

BASE_DIR      = "/project2/jessetho_1732/mousumid/PolicySel/iqlSpatial/iql_train"
EPOCHS        = 1000
EVAL_EVERY    = 10

# ── FIX 1: correct data path ───────────────────────────────────
REAL_DATA_DIR = os.path.join(BASE_DIR, "expt_pipi_envfix", "data")

# ── Sanity check ───────────────────────────────────────────────
for f in ["train_chunks.pkl", "eval_chunks.pkl",
          "train_embeddings.pt", "eval_embeddings.pt"]:
    assert os.path.exists(os.path.join(REAL_DATA_DIR, f)), \
        f"Missing: {REAL_DATA_DIR}/{f}"
print(f"Data verified at {REAL_DATA_DIR}")

os.makedirs(f"{BASE_DIR}/slurm_logs",   exist_ok=True)
os.makedirs(f"{BASE_DIR}/grouped_grid", exist_ok=True)

# ── Generate combinations ──────────────────────────────────────
keys, values = zip(*architectures.items())
arch_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
total_runs = len(arch_combinations) * len(gammas) * len(taus)
print(f"SLURM jobs : {len(arch_combinations)}")
print(f"Runs/job   : {len(gammas) * len(taus)}")
print(f"Total runs : {total_runs}")

# ── Submit ─────────────────────────────────────────────────────
for i, arch in enumerate(arch_combinations):
    h_str    = arch["hidden"].split(",")[0]
    job_name = f"iql_h{h_str}_p{arch['policy_embed_dim']}_b{arch['batch_size']}"

    command_block = ""
    for g in gammas:
        for t in taus:
            expt_name = (f"grid_g{g}_t{t}_h{h_str}"
                         f"_p{arch['policy_embed_dim']}"
                         f"_b{arch['batch_size']}")
            expt_dir = os.path.join(BASE_DIR, "grouped_grid", expt_name)
            os.makedirs(expt_dir, exist_ok=True)

            # FIX 2: use lexists to handle broken symlinks
            symlink = os.path.join(expt_dir, "data")
            if os.path.lexists(symlink):
                os.remove(symlink)
            os.symlink(REAL_DATA_DIR, symlink)
            if False:
                os.symlink(REAL_DATA_DIR, symlink)

            command_block += f"""
echo "====== gamma={g} tau={t} h={arch['hidden']} p={arch['policy_embed_dim']} b={arch['batch_size']} ======"
python $BASE_DIR/train_iql.py \\
    --expt-dir {expt_dir} \\
    --policies pi0,pi05 \\
    --epochs {EPOCHS} \\
    --lr 1e-4 \\
    --gamma {g} \\
    --tau {t} \\
    --hidden {arch['hidden']} \\
    --policy-embed-dim {arch['policy_embed_dim']} \\
    --batch-size {arch['batch_size']} \\
    --eval-every {EVAL_EVERY} \\
    --reward-scale 10.0 \\
    --freeze-policy-emb
echo "Exit: $?"
"""

    # FIX 3: conda env activated inside SLURM script
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={BASE_DIR}/slurm_logs/%x_%j.out
#SBATCH --error={BASE_DIR}/slurm_logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=nlp
#SBATCH --account=jessetho_1732
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu
#SBATCH --mem=120G

source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate qwen-eval
export PYTHONNOUSERSITE=1
CONDA_SITE=/scratch1/mousumid/miniconda3/envs/qwen-eval/lib/python3.11/site-packages
export PYTHONPATH=/project2/jessetho_1732/mousumid/PolicySel/LIBERO:$CONDA_SITE
BASE_DIR={BASE_DIR}

echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Job:  {job_name}"

{command_block}

echo "All runs done for {job_name}"
"""

    proc = subprocess.run(["sbatch"], input=slurm_script, text=True, capture_output=True)
    if proc.returncode == 0:
        print(f"[{i+1:>3}/{len(arch_combinations)}] Submitted {job_name}  →  {proc.stdout.strip()}")
    else:
        print(f"[{i+1:>3}/{len(arch_combinations)}] FAILED   {job_name}")
        print(f"  {proc.stderr.strip()}")

print(f"\nDone. {len(arch_combinations)} jobs submitted, {total_runs} total runs.")
print(f"Monitor: squeue -u $USER")
print(f"Logs:    {BASE_DIR}/slurm_logs/")
