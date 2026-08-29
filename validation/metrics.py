"""Scrape vLLM's Prometheus /metrics endpoint and diff two snapshots.

The two numbers that matter for the validation (VALIDATION_PLAN.md section 7):

  * preemptions  -- vllm:num_preemptions_total. These are KV evictions: a
    running sequence kicked out of the GPU KV pool. This is the real-hardware
    analogue of the simulator's kv_recomputes / kv_from_cpu counters.

  * adapter activity -- vllm:lora_requests_info gauge. NOTE the known gap
    (vLLM issue #45325): idle adapters vanish from this gauge, so you often
    cannot read reload COUNTS off it directly. Fall back to inferring reloads
    from TTFT spikes on adapter switches in the workload.py output.

Usage:
  python validation/metrics.py snapshot --out before.json
  # ... run validation/workload.py ...
  python validation/metrics.py snapshot --out after.json
  python validation/metrics.py diff before.json after.json
"""

import argparse
import json
import sys
import time
import urllib.request

WATCH_COUNTERS = [
    "vllm:num_preemptions_total",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_swapped",
    "vllm:gpu_cache_usage_perc",
    "vllm:cpu_cache_usage_perc",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
]


def fetch(url):
    with urllib.request.urlopen(url, timeout=10) as resp:
        return resp.read().decode()


def parse_prom(text):
    """Very small Prometheus text parser: returns {metric: [(labels, value), ...]}."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            name_labels, value = line.rsplit(" ", 1)
        except ValueError:
            continue
        if "{" in name_labels:
            name, labels = name_labels.split("{", 1)
            labels = labels.rstrip("}")
        else:
            name, labels = name_labels, ""
        try:
            val = float(value)
        except ValueError:
            continue
        out.setdefault(name.strip(), []).append((labels, val))
    return out


def summarize_snapshot(parsed):
    snap = {"_t": time.time()}
    for name in WATCH_COUNTERS:
        series = parsed.get(name, [])
        if not series:
            continue
        # sum across label sets (e.g. per-model-name counters)
        snap[name] = sum(v for _, v in series)
    # keep raw lora gauge for eyeballing which adapters were resident
    if "vllm:lora_requests_info" in parsed:
        snap["vllm:lora_requests_info"] = parsed["vllm:lora_requests_info"]
    return snap


def cmd_snapshot(args):
    text = fetch(args.url)
    snap = summarize_snapshot(parse_prom(text))
    with open(args.out, "w") as f:
        json.dump(snap, f, indent=2)
    print(f"[metrics] snapshot -> {args.out}")
    for k, v in snap.items():
        if k.startswith("_") or k == "vllm:lora_requests_info":
            continue
        print(f"  {k:40s} {v}")


def cmd_diff(args):
    before = json.load(open(args.before))
    after = json.load(open(args.after))
    dt = after.get("_t", 0) - before.get("_t", 0)
    print(f"[metrics] window: {dt:.1f}s\n")
    print(f"{'counter':42s} {'before':>14s} {'after':>14s} {'delta':>14s}")
    print("-" * 88)
    keys = [k for k in WATCH_COUNTERS if k in before or k in after]
    for k in keys:
        b = before.get(k, 0.0)
        a = after.get(k, 0.0)
        print(f"{k:42s} {b:14.3f} {a:14.3f} {a - b:+14.3f}")
    print("\nKV evictions during the run  ~=  delta(vllm:num_preemptions_total)")
    print("Adapter reloads: not reliably in /metrics (issue #45325) -- infer from")
    print("TTFT spikes on adapter switches in the workload.py jsonl.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("snapshot")
    s.add_argument("--url", default="http://localhost:8000/metrics")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_snapshot)

    d = sub.add_parser("diff")
    d.add_argument("before")
    d.add_argument("after")
    d.set_defaults(func=cmd_diff)

    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    try:
        args.func(args)
    except urllib.error.URLError as e:
        sys.exit(f"could not reach vLLM /metrics: {e}")
