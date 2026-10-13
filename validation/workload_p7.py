"""Drive a real vLLM server with phase 7's chat workload, token for token.

The conversations, adapters, arrival times, shared system prompts and every
turn's prompt / reply length come from `phase7_elora.chat_workload` (the same
call the simulator makes), so the GPU run and `phase7_elora.py` see identical
traffic.

To make KV sizes exact rather than "about right", prompts are sent as TOKEN
IDS to /v1/completions (no chat template, no tokenizer guesswork):
  * shared system prompt: `prefix_tokens` ids fixed per (adapter, group), so
    conversations in the same group really share a cached prefix
  * each turn adds `input_tokens` new ids
  * the reply is forced to exactly `output_tokens` tokens (ignore_eos) and its
    token ids are read back (return_tokens_as_token_ids) and appended to the
    history, so the next turn's prompt = previous prompt + previous reply,
    exactly what prefix caching / KV offload can reuse.
Token ids are random; the text is meaningless and that does not matter here.

A turn is sent at its scheduled arrival time, or as soon as the previous turn
of the same conversation finishes if that is later (the history is needed).

Per turn it records:
  ttft_ms   send -> first token
  tpot_ms   (last token - first token) / (tokens - 1)
  e2e_ms    send -> last token
  late_ms   how long after its scheduled arrival the turn could be sent

Usage:
  python validation/workload_p7.py --adapters lora0 ... lora19 \
      --seed 0 --burst 1 --tag ml6_off --out validation/results/p7_ml6_off_b1_s0.jsonl
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
from phase7_elora import chat_workload  # noqa: E402

try:
    from openai import AsyncOpenAI
except ImportError:
    sys.exit("pip install -r validation/requirements.txt  (need the openai package)")

# Llama-3 vocab is 128,256 with special tokens at >= 128,000. Stay well inside
# the ordinary range so random ids are always valid.
TOKEN_LO, TOKEN_HI = 1000, 100000


def ids(rng, n):
    return [rng.randrange(TOKEN_LO, TOKEN_HI) for _ in range(n)]


def build(seed, burst, n_adapters):
    convos, reqs = chat_workload(seed, burst=burst, n_adapters=n_adapters)
    # id streams are seeded by (seed, burst) so different runs on one server
    # never share a prefix by accident
    base = seed * 1000 + int(burst * 10)
    prefixes = {}
    for c in convos:
        key = getattr(c, "prefix_key", None)
        if key is not None and key not in prefixes:
            prefixes[key] = ids(random.Random(f"prefix-{base}-{key}"), c.prefix_tokens)
    turns = {}
    for r in reqs:
        turns.setdefault(r.conversation_id, []).append(r)
    for t in turns.values():
        t.sort(key=lambda r: r.arrival_time)
    return convos, reqs, prefixes, turns, base


def parse_token_ids(logprobs):
    """With return_tokens_as_token_ids, vLLM returns tokens as 'token_id:123'."""
    out = []
    for tok in (logprobs.tokens or []):
        if isinstance(tok, str) and tok.startswith("token_id:"):
            out.append(int(tok.split(":", 1)[1]))
    return out


async def run_conversation(client, convo, turns, prefix, adapter, base, t0,
                           results, args):
    rng = random.Random(f"turns-{base}-{convo.conversation_id}")
    history = list(prefix)
    for i, r in enumerate(turns):
        target = t0 + r.arrival_time * args.time_scale / 1000.0
        now = time.monotonic()
        if target > now:
            await asyncio.sleep(target - now)
        history += ids(rng, r.input_tokens)
        send = time.monotonic()
        first = last = None
        n_tok = 0
        new_ids = []
        err = None
        try:
            stream = await client.completions.create(
                model=adapter, prompt=history, max_tokens=r.output_tokens,
                temperature=0.0, stream=True, logprobs=1,
                extra_body={"ignore_eos": True, "min_tokens": r.output_tokens,
                            "return_tokens_as_token_ids": True})
            async for chunk in stream:
                if not chunk.choices:
                    continue
                ch = chunk.choices[0]
                got = parse_token_ids(ch.logprobs) if ch.logprobs else []
                if got or ch.text:
                    t = time.monotonic()
                    if first is None:
                        first = t
                    last = t
                    n_tok += max(1, len(got))
                    new_ids += got
        except Exception as e:  # noqa: BLE001 -- record and keep the run going
            err = repr(e)
        done = time.monotonic()
        if err is None and len(new_ids) != r.output_tokens:
            err = f"got {len(new_ids)} token ids, expected {r.output_tokens}"
        # keep the history exact even on a short reply
        history += new_ids + ids(rng, max(0, r.output_tokens - len(new_ids)))
        results.append({
            "tag": args.tag, "seed": args.seed, "burst": args.burst,
            "conversation_id": convo.conversation_id, "turn_index": i,
            "adapter": adapter, "scheduled_s": r.arrival_time / 1000.0,
            "late_ms": (send - target) * 1000.0,
            "prompt_tokens": len(history) - r.output_tokens,
            "output_tokens": r.output_tokens,
            "ttft_ms": (first - send) * 1000.0 if first else None,
            "tpot_ms": ((last - first) * 1000.0 / (n_tok - 1)) if first and n_tok > 1 else None,
            "e2e_ms": (done - send) * 1000.0,
            "error": err,
        })


async def main_async(args):
    convos, reqs, prefixes, turns, base = build(args.seed, args.burst, args.n_adapters)
    if len(args.adapters) < args.n_adapters:
        sys.exit(f"need {args.n_adapters} adapter names, got {len(args.adapters)}")
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
    span = max(r.arrival_time for r in reqs) / 1000.0
    print(f"[p7] seed={args.seed} burst={args.burst} tag={args.tag}: "
          f"{len(convos)} conversations, {len(reqs)} turns, schedule {span:.0f} s", flush=True)
    results = []
    t0 = time.monotonic() + 1.0
    await asyncio.gather(*[
        run_conversation(client, c, turns[c.conversation_id],
                         prefixes.get(getattr(c, "prefix_key", None), []),
                         args.adapters[c.adapter_id], base, t0, results, args)
        for c in convos])
    await client.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    results.sort(key=lambda x: (x["conversation_id"], x["turn_index"]))
    with open(args.out, "w") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")
    ok = [r for r in results if r["error"] is None and r["ttft_ms"] is not None]
    errs = len(results) - len(ok)
    if ok:
        mean = lambda k: sum(r[k] for r in ok if r[k] is not None) / len(ok)
        print(f"[p7] {len(ok)} ok, {errs} errors -> {args.out}")
        print(f"[p7] mean TTFT {mean('ttft_ms'):.0f} ms  TPOT {mean('tpot_ms'):.1f} ms  "
              f"e2e {mean('e2e_ms'):.0f} ms  late {mean('late_ms'):.0f} ms", flush=True)
    else:
        print(f"[p7] ALL {len(results)} turns failed; first error: "
              f"{results[0]['error'] if results else '-'}", flush=True)
    if errs:
        print(f"[p7] WARNING {errs} turns errored; see the 'error' field", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--adapters", nargs="+", required=True)
    p.add_argument("--n-adapters", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--burst", type=float, default=1.0)
    p.add_argument("--tag", default="run")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--time-scale", type=float, default=1.0,
                   help="multiply the schedule by this; keep 1 for the real run "
                        "(smaller values only for testing the harness)")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    if args.out is None:
        args.out = os.path.join(_REPO_ROOT, "validation", "results",
                                f"p7_{args.tag}_b{args.burst:g}_s{args.seed}.jsonl")
    return args


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
