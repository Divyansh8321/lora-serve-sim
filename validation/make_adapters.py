"""Write N random rank-32 LoRA adapters for a base model, in PEFT format.

Adapter quality is irrelevant to this experiment; only size and count matter.
Rank 32 on all 7 linear layers of Llama-3.1-8B is ~168 MB in fp16, matching
core.ADAPTER_MB (180, see the note there) and ELORA's ranks (32/64).

Needs only numpy + safetensors and the base model's config.json (no weights,
no GPU). Shapes are read from the config, so other Llama-style models work.

  python validation/make_adapters.py --config /path/to/config.json \
      --base-model meta-llama/Llama-3.1-8B-Instruct --out /workspace/adapters --n 20
  # or let it fetch config.json from the Hub:
  python validation/make_adapters.py --base-model meta-llama/Llama-3.1-8B-Instruct ...

Prints the LORA_MODULES / ADAPTER_NAMES lines to export.
"""

import argparse
import json
import os

import numpy as np
from safetensors.numpy import save_file

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def shapes(cfg):
    h = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    heads = cfg["num_attention_heads"]
    kv = cfg.get("num_key_value_heads", heads)
    hd = cfg.get("head_dim", h // heads)
    return {  # (in_features, out_features)
        "q_proj": (h, heads * hd), "k_proj": (h, kv * hd), "v_proj": (h, kv * hd),
        "o_proj": (heads * hd, h), "gate_proj": (h, inter), "up_proj": (h, inter),
        "down_proj": (inter, h),
    }


def load_config(args):
    if args.config:
        return json.load(open(args.config))
    from huggingface_hub import hf_hub_download
    return json.load(open(hf_hub_download(args.base_model, "config.json")))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--config", default=None, help="local config.json (else fetched)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--rank", type=int, default=32)
    args = ap.parse_args()

    cfg = load_config(args)
    shp = shapes(cfg)
    layers = cfg["num_hidden_layers"]
    names = []
    for i in range(args.n):
        rng = np.random.default_rng(i)
        tensors = {}
        for L in range(layers):
            for t in TARGETS:
                mod = "self_attn" if t in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
                fin, fout = shp[t]
                key = f"base_model.model.model.layers.{L}.{mod}.{t}"
                # small values: outputs stay finite; content is irrelevant
                tensors[f"{key}.lora_A.weight"] = (rng.standard_normal((args.rank, fin)) * 0.01).astype(np.float16)
                tensors[f"{key}.lora_B.weight"] = (rng.standard_normal((fout, args.rank)) * 0.01).astype(np.float16)
        d = os.path.join(args.out, f"lora{i}")
        os.makedirs(d, exist_ok=True)
        save_file(tensors, os.path.join(d, "adapter_model.safetensors"))
        json.dump({
            "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": args.rank,
            "lora_alpha": 2 * args.rank, "lora_dropout": 0.0, "bias": "none",
            "target_modules": TARGETS, "base_model_name_or_path": args.base_model,
            "fan_in_fan_out": False, "inference_mode": True,
        }, open(os.path.join(d, "adapter_config.json"), "w"), indent=1)
        mb = os.path.getsize(os.path.join(d, "adapter_model.safetensors")) / 1e6
        names.append(f"lora{i}")
        print(f"wrote {d} ({mb:.0f} MB)", flush=True)

    print("\nexport LORA_MODULES=\"" +
          " ".join(f"{n}={os.path.abspath(os.path.join(args.out, n))}" for n in names) + "\"")
    print("export ADAPTER_NAMES=\"" + " ".join(names) + "\"")


if __name__ == "__main__":
    main()
