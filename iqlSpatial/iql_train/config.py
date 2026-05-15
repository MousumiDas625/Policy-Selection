"""
config.py — Central configuration for IQL policy selection experiments.
"""

# Chunk sizes: how many env steps each VLA executes per selection
CHUNK_SIZES = {
    "pi0": 5,
    "pi05": 5,
    "openvla": 1,
    "molmoact": 8,
    "pi0fast": 5,
}

# LIBERO task suite max steps
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

# MolmoAct unnorm key per suite
MOLMO_UNNORM = {
    "libero_spatial": "libero_spatial_no_noops_modified",
    "libero_object": "libero_object_no_noops_modified",
    "libero_goal": "libero_goal_no_noops_modified",
    "libero_10": "libero_10_no_noops_modified",
}

TASK_DESCRIPTIONS = {
    "libero_spatial": {
        0: "pick up the black bowl between the plate and the ramekin and place it on the plate",
        1: "pick up the black bowl next to the ramekin and place it on the plate",
        2: "pick up the black bowl from table center and place it on the plate",
        3: "pick up the black bowl on the cookie box and place it on the plate",
        4: "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
        5: "pick up the black bowl on the ramekin and place it on the plate",
        6: "pick up the black bowl next to the cookie box and place it on the plate",
        7: "pick up the black bowl on the stove and place it on the plate",
        8: "pick up the black bowl next to the plate and place it on the plate",
        9: "pick up the black bowl on the wooden cabinet and place it on the plate",
    },
}

# Tasks reserved for ONLINE evaluation only. No episodes from these tasks
# are ever loaded into offline train or val. build_chunks asserts this.
HELDOUT_TASKS = {
    "libero_spatial": [5, 9],
}

# Default tasks to use for offline IQL training (everything except held-out).
DEFAULT_TRAIN_TASKS = {
    "libero_spatial": [0, 1, 2, 3, 4, 6, 7, 8],
}

# Default paths
DEFAULT_DATA_ROOT = "/project2/jessetho_1732/mousumid/PolicySel/iqlSpatial"
DEFAULT_HF_CACHE = "/project2/jessetho_1732/mousumid/PolicySel/hf_cache"
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
