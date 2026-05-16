"""
embed_qwen.py — Precompute Qwen2.5-VL-3B embeddings for chunk observations.

Encodes (agentview_image, language_instruction) → fixed embedding vector.
Qwen is frozen — used only as feature extractor.

Usage:
  CUDA_VISIBLE_DEVICES=0 python embed_qwen.py \
      --expt-dir /project2/jessetho_1732/mousumid/PolicySel/iqlSpatial/iql_train/expt1 \
      --model Qwen/Qwen2.5-VL-3B-Instruct \
      --batch-size 4
"""

import argparse
import gc
import os
import pickle

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def load_model(model_name, device, cache_dir):
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir

    # Try VL model first
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        print(f"Loading {model_name}...")
        processor = AutoProcessor.from_pretrained(
            model_name, cache_dir=cache_dir, trust_remote_code=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, cache_dir=cache_dir,
            torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(device).eval()
        # FIX: Safely extract hidden_size based on HF config structure
        if hasattr(model.config, "text_config"):
            hdim = model.config.text_config.hidden_size
        else:
            hdim = getattr(model.config, "hidden_size", 2048)
            
        print(f"  VL model loaded, hidden_dim={hdim}")

#        hdim = model.config.hidden_size
 #       print(f"  VL model loaded, hidden_dim={hdim}")
        return model, processor, hdim, "vl"
    except Exception as e:
        print(f"  VL failed ({e}), trying text-only...")

    # Fallback text-only
    fallback = model_name.replace("VL-", "").replace("VL_", "")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(fallback, cache_dir=cache_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        fallback, cache_dir=cache_dir,
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    hdim = model.config.hidden_size
    print(f"  Text model loaded, hidden_dim={hdim}")
    return model, tok, hdim, "text"


@torch.no_grad()
def encode_one_vl(model, processor, img_np, instruction, device):
    """Encode single (image, instruction) with Qwen2.5-VL."""
    from qwen_vl_utils import process_vision_info
    pil = Image.fromarray(img_np.astype(np.uint8))
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": pil},
        {"type": "text", "text": f"Robot task: {instruction}. Describe the workspace."},
    ]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(msgs)
    inputs = processor(text=[text], images=img_in, videos=vid_in,
                       padding=True, return_tensors="pt").to(device)
    out = model(**inputs, output_hidden_states=True)
    emb = out.hidden_states[-1][:, -1, :].float().cpu()  # (1, D)
    del inputs, out
    torch.cuda.empty_cache()
    return emb


@torch.no_grad()
def encode_one_text(model, tokenizer, img_np, instruction, device):
    """Fallback: text-only encoding."""
    mean_rgb = img_np.mean(axis=(0, 1))
    text = (f"Robot task: {instruction}. "
            f"Image mean RGB=({mean_rgb[0]:.0f},{mean_rgb[1]:.0f},{mean_rgb[2]:.0f})")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    out = model(**inputs, output_hidden_states=True)
    emb = out.hidden_states[-1][:, -1, :].float().cpu()
    del inputs, out
    return emb


def embed_all(transitions, model, proc, hdim, mtype, device, batch_size):
    n = len(transitions)
    enc = encode_one_vl if mtype == "vl" else encode_one_text

    obs_list, nobs_list = [], []

    print(f"  Encoding {n} obs...")
    for i in tqdm(range(n)):
        t = transitions[i]
        obs_list.append(enc(model, proc, t["obs_image"], t["language_instruction"], device))
        if (i + 1) % 100 == 0:
            gc.collect(); torch.cuda.empty_cache()

    print(f"  Encoding {n} next_obs...")
    for i in tqdm(range(n)):
        t = transitions[i]
        nobs_list.append(enc(model, proc, t["next_obs_image"], t["language_instruction"], device))
        if (i + 1) % 100 == 0:
            gc.collect(); torch.cuda.empty_cache()

    return torch.cat(obs_list, 0), torch.cat(nobs_list, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expt-dir", required=True,
                        help="Experiment dir (contains data/ with *_chunks.pkl)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true",
                        help="Recompute embeddings even if output file already exists")
    parser.add_argument("--cache-dir",
                        default="/project2/jessetho_1732/mousumid/PolicySel/hf_cache")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, proc, hdim, mtype = load_model(args.model, device, args.cache_dir)

    data_dir = os.path.join(args.expt_dir, "data")

    for split in ["train", "eval"]:
        cp = os.path.join(data_dir, f"{split}_chunks.pkl")
        if not os.path.exists(cp):
            print(f"  Skip {split}: not found")
            continue
        out_check = os.path.join(data_dir, f"{split}_embeddings.pt")
        if os.path.exists(out_check) and not args.force:
            print(f"  Skip {split}: embeddings already exist at {out_check}")
            print(f"  Run with --force to recompute")
            continue
        print(f"\n{'='*60}\n  {split}: {cp}\n{'='*60}")
        trans = pickle.load(open(cp, "rb"))
        print(f"  {len(trans)} transitions")

        obs_e, nobs_e = embed_all(trans, model, proc, hdim, mtype, device, args.batch_size)

        out = os.path.join(data_dir, f"{split}_embeddings.pt")
        torch.save({
            "obs_emb": obs_e, "next_obs_emb": nobs_e,
            "hidden_dim": hdim, "model_name": args.model,
            "model_type": mtype, "n": len(trans),
        }, out)
        print(f"  Saved {out}  shape={obs_e.shape}")

    print("\nDone!")


if __name__ == "__main__":
    main()
