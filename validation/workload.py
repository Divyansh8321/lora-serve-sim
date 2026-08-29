"""Drive a real vLLM server with the SAME workload the simulator uses.

This imports `make_multi_turn_workload` from the simulator's `core.py` so the
conversation count, turn count, per-turn token counts, adapter assignment
(Zipf), and arrival schedule are byte-for-byte identical to what phases 2-4
simulate. The only thing that differs is that here a real model actually runs.

What it records, per request (one turn):
  - arrival_time   : the simulator's scheduled arrival (ms, from t=0)
  - send_time      : wall-clock we actually dispatched it
  - ttft_ms        : time to first token
  - e2e_ms         : arrival -> final token, matching core.Request.latency()
  - prompt_tokens / completion_tokens : as reported by the server
  - adapter        : which LoRA served it

Output: newline-delimited JSON to --out (default validation/results/run.jsonl).

Usage:
  python validation/workload.py \
      --base-url http://localhost:8000/v1 \
      --adapters lora0 lora1 lora2 lora3 lora4 lora5 \
                 lora6 lora7 lora8 lora9 lora10 lora11 \
      --max-loras 5 \
      --seed 0

Run one invocation per (seed, max-loras) cell of the sweep. `--max-loras` here
is only recorded into the output for bookkeeping; the actual slab size is set
when you launch the server (validation/serve.sh).
"""

import argparse
import asyncio
import json
import os
import sys
import time

# Import the simulator's own workload generator, unmodified.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
from core import make_multi_turn_workload  # noqa: E402

try:
    from openai import AsyncOpenAI
except ImportError:
    sys.exit("pip install -r validation/requirements.txt  (need the openai package)")


# The simulator's default multi-turn parameters (phases 2-4). Keep in lockstep
# with the trial() calls in phase2/3/4 -- if you change one, change both.
SIM_DEFAULTS = dict(
    n_conversations=40,
    n_adapters=12,
    turns_per_convo=5,
    skew=1.0,
    rate=0.0006,
    mean_tokens=40,
    turn_gap=6000.0,
)


def build_turns(seed, overrides):
    """Return (conversations, requests) from the simulator, plus a cid->adapter map."""
    params = dict(SIM_DEFAULTS)
    params.update(overrides)
    convos, requests = make_multi_turn_workload(seed=seed, **params)
    cid_to_adapter = {c.conversation_id: c.adapter_id for c in convos}
    # requests are already emitted grouped by conversation; sort by arrival for
    # dispatch, but keep per-conversation ordering stable for history assembly.
    requests.sort(key=lambda r: (r.conversation_id, r.arrival_time))
    return convos, requests, cid_to_adapter


def group_by_conversation(requests):
    convos = {}
    for r in requests:
        convos.setdefault(r.conversation_id, []).append(r)
    for turns in convos.values():
        turns.sort(key=lambda r: r.arrival_time)
    return convos


async def run_conversation(client, model_for_cid, cid, turns, history_chars,
                           time_scale, t0_wall, results, sem):
    """Issue one conversation's turns in order, sending full prior history each turn.

    history_chars: approximate characters of synthetic history to prepend so the
    server's KV cache grows the way the simulator assumes (~mean_tokens per turn
    accumulated). We can't control the model's real tokenization exactly, but we
    can make the prompt grow monotonically with turn index, which is what drives
    the stale-KV pathology.
    """
    model = model_for_cid[cid]
    messages = [{"role": "system", "content": "You are a terse assistant. Answer in one sentence."}]

    for turn_idx, r in enumerate(turns):
        # Wait until this turn's simulated arrival time (scaled).
        target_wall = t0_wall + (r.arrival_time * time_scale) / 1000.0
        now = time.monotonic()
        if target_wall > now:
            await asyncio.sleep(target_wall - now)

        # Grow the prompt with turn index: prior turns' text is the "history".
        user_msg = ("Continue the discussion. " * (1 + turn_idx)
                    + f"This is turn {turn_idx + 1} of conversation {cid}. "
                    + "Give me one more short thought.")
        messages.append({"role": "user", "content": user_msg})

        arrival_wall = time.monotonic()
        first_token_wall = None
        text_parts = []
        err = None
        async with sem:
            try:
                stream = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=r.output_tokens,
                    temperature=0.0,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                usage = None
                async for chunk in stream:
                    if chunk.usage is not None:
                        usage = chunk.usage
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content
                    if delta:
                        if first_token_wall is None:
                            first_token_wall = time.monotonic()
                        text_parts.append(delta)
            except Exception as e:  # noqa: BLE001 - record and continue the sweep
                err = repr(e)
                usage = None

        done_wall = time.monotonic()
        reply = "".join(text_parts)
        messages.append({"role": "assistant", "content": reply or "(no reply)"})

        rec = {
            "conversation_id": cid,
            "request_id": r.request_id,
            "turn_index": turn_idx,
            "adapter": model,
            "adapter_id": model_for_cid.get("_ids", {}).get(cid),
            "arrival_time_ms": r.arrival_time,          # simulator clock
            "target_output_tokens": r.output_tokens,
            "send_wall_s": arrival_wall - t0_wall,
            "ttft_ms": ((first_token_wall - arrival_wall) * 1000.0
                        if first_token_wall else None),
            # e2e measured from the SCALED simulated arrival, to match
            # core.Request.latency() == end_time - arrival_time.
            "e2e_ms": (done_wall - target_wall) * 1000.0,
            "server_e2e_ms": (done_wall - arrival_wall) * 1000.0,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "error": err,
        }
        results.append(rec)


async def main_async(args):
    convos, requests, cid_to_adapter = build_turns(args.seed, _parse_overrides(args))
    by_convo = group_by_conversation(requests)

    n_adapters_needed = max(cid_to_adapter.values()) + 1
    if len(args.adapters) < n_adapters_needed:
        sys.exit(f"workload uses {n_adapters_needed} distinct adapters but only "
                 f"{len(args.adapters)} names were passed via --adapters")

    # Map each conversation's integer adapter_id -> a served adapter name.
    model_for_cid = {cid: args.adapters[aid] for cid, aid in cid_to_adapter.items()}
    model_for_cid["_ids"] = dict(cid_to_adapter)

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
    sem = asyncio.Semaphore(args.max_inflight)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    results = []
    t0_wall = time.monotonic()

    total_span_ms = max(r.arrival_time for r in requests)
    print(f"[workload] seed={args.seed} conversations={len(by_convo)} "
          f"turns={len(requests)} adapters={n_adapters_needed} "
          f"sim_span={total_span_ms/1000:.1f}s scaled_span={total_span_ms*args.time_scale/1000:.1f}s")

    tasks = [
        run_conversation(client, model_for_cid, cid, turns,
                         history_chars=0, time_scale=args.time_scale,
                         t0_wall=t0_wall, results=results, sem=sem)
        for cid, turns in by_convo.items()
    ]
    await asyncio.gather(*tasks)
    await client.close()

    results.sort(key=lambda x: (x["conversation_id"], x["turn_index"]))
    with open(args.out, "w") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")

    ok = [r for r in results if r["error"] is None]
    errs = len(results) - len(ok)
    lats = sorted(r["e2e_ms"] for r in ok)
    if lats:
        p50 = lats[len(lats) // 2]
        p95 = lats[min(len(lats) - 1, int(len(lats) * 0.95))]
        print(f"[workload] wrote {len(results)} records ({errs} errors) -> {args.out}")
        print(f"[workload] measured e2e  p50={p50:.0f}ms  p95={p95:.0f}ms")
    else:
        print(f"[workload] wrote {len(results)} records, ALL errored -> {args.out}")


def _parse_overrides(args):
    ov = {}
    if args.n_conversations is not None:
        ov["n_conversations"] = args.n_conversations
    if args.turns_per_convo is not None:
        ov["turns_per_convo"] = args.turns_per_convo
    if args.rate is not None:
        ov["rate"] = args.rate
    if args.mean_tokens is not None:
        ov["mean_tokens"] = args.mean_tokens
    if args.turn_gap is not None:
        ov["turn_gap"] = args.turn_gap
    return ov


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--adapters", nargs="+", required=True,
                   help="Served LoRA adapter names, one per simulator adapter_id "
                        "(need at least n_adapters of them).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-loras", type=int, default=None,
                   help="Recorded for bookkeeping only; set the real slab size in serve.sh.")
    p.add_argument("--out", default=None)
    p.add_argument("--time-scale", type=float, default=1.0,
                   help="Multiply simulated inter-arrival gaps by this. The sim's "
                        "turn_gap is 6000ms; keep 1.0 for a faithful comparison, "
                        "raise it only if the box can't keep up, and note it.")
    p.add_argument("--max-inflight", type=int, default=64,
                   help="Client-side concurrency cap; keep well above the number "
                        "of conversations so the client is never the bottleneck.")
    p.add_argument("--timeout", type=float, default=120.0)
    # optional workload overrides (default: match the simulator exactly)
    p.add_argument("--n-conversations", type=int, default=None)
    p.add_argument("--turns-per-convo", type=int, default=None)
    p.add_argument("--rate", type=float, default=None)
    p.add_argument("--mean-tokens", type=int, default=None)
    p.add_argument("--turn-gap", type=float, default=None)
    args = p.parse_args(argv)
    if args.out is None:
        tag = f"seed{args.seed}"
        if args.max_loras is not None:
            tag = f"maxloras{args.max_loras}_" + tag
        args.out = os.path.join(_REPO_ROOT, "validation", "results", f"run_{tag}.jsonl")
    return args


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
