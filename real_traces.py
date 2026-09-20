"""Load real production traces and convert them into the same (conversations,
requests) shape core.py's synthetic generators produce -- so every phase can
run, unmodified, on real arrival timing and real prefix-sharing structure
instead of our invented formulas.

Three traces, each fixing one specific gap named in the project's honesty list:

  BurstGPT (github.com/HPMLL/BurstGPT, CC-BY-4.0)
    10M+ real ChatGPT/GPT-4 request logs from Azure OpenAI, arrival timestamp
    per row. Replaces make_bursty_workload's invented "5-10x rate spike for
    25% of a 40s window" with REAL inter-arrival gaps.

  Mooncake FAST'25 trace (github.com/kvcache-ai/Mooncake, traces/*.jsonl)
    Real anonymized request trace built specifically to study KV-cache
    sharing: each request lists `hash_ids`, the 512-token prefix blocks its
    prompt matches. Matching hash_ids across requests = shareable KV cache.
    Replaces make_prefix_sharing_workload's invented "3 fixed-size groups"
    with REAL, Zipf-shaped, multi-turn sharing structure. No LoRA-adapter
    field -- adapter identity is inferred from the depth-2 hash node (see
    load_mooncake_conversations for the exact mapping and its limitation).

  Google Taskmaster TM-1 (github.com/google-research-datasets/Taskmaster)
    ONE OF ELORA'S OWN THREE EVALUATION DATASETS ("Personal Agents"). Real
    task-oriented dialogs with a genuine task-type field (`instruction_id`),
    used directly as adapter identity -- no inference needed, unlike
    Mooncake. See load_taskmaster_conversations for detail, including how we
    handle its missing timestamps the same way ELORA's own paper says they
    did (overlay Azure-trace-style arrival timing -- here, BurstGPT).

HONEST LIMITATION: Mooncake's real prefix chains are variable-depth and grow
turn-by-turn (turn 1 shares 4 blocks, turn 2 shares 5, ...). Our simulator's
prefix-tree model (phase 6/7) uses ONE prefix_tokens length per conversation.
We collapse the real chain to a single number: 512 * (blocks shared with at
least one other conversation), taken from the conversation's LAST turn (its
deepest, most-shared prefix). This under-counts a conversation's sharing in
its early turns and is a real simplification -- flagged in README/ROADMAP.

Run standalone to print summary stats:  python real_traces.py
"""

import csv
import json
import os
from collections import defaultdict

from core import Conversation, Request

TRACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces")
MIN_ADAPTER_GROUP = 2      # depth-2 nodes with fewer requesters -> overflow adapter
BLOCK_TOKENS = 512         # Mooncake's fixed prefix-block size


# ==========================================================================
# BurstGPT -- real arrival timing
# ==========================================================================

def load_burstgpt_arrivals(path=None, model="ChatGPT", limit=None, start_at=0):
    """Return a sorted list of real arrival times (ms, rebased to start at 0)
    for one model's traffic. `start_at` skips into the file (row index) so
    different seeds/windows can sample different real time ranges."""
    path = path or os.path.join(TRACES_DIR, "BurstGPT_without_fails_1.csv")
    times = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if i < start_at:
                continue
            if row["Model"] != model:
                continue
            times.append(float(row["Timestamp"]) * 1000.0)   # seconds -> ms
            if limit and len(times) >= limit:
                break
    times.sort()
    if not times:
        raise ValueError(f"no rows for model={model} in {path}")
    t0 = times[0]
    return [t - t0 for t in times]


def load_burstgpt_tokens(path=None, model="ChatGPT", limit=None, start_at=0):
    """Return list of (request_tokens, response_tokens) aligned with
    load_burstgpt_arrivals for the same window."""
    path = path or os.path.join(TRACES_DIR, "BurstGPT_without_fails_1.csv")
    out = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if i < start_at:
                continue
            if row["Model"] != model:
                continue
            out.append((int(row["Request tokens"]), int(row["Response tokens"])))
            if limit and len(out) >= limit:
                break
    return out


# ==========================================================================
# Mooncake -- real prefix-sharing + multi-turn structure
# ==========================================================================

def _load_mooncake_raw(name="conversation_trace"):
    path = os.path.join(TRACES_DIR, f"mooncake_{name}.jsonl")
    return [json.loads(l) for l in open(path) if l.strip()]


def _group_into_conversations(records):
    """Mooncake's file is a flat list of requests, not pre-grouped into
    conversations. We reconstruct conversations by depth-3 node (hash_ids[2]):
    empirically this is the level where a real multi-turn thread's identity
    lives (root=global, depth-2=adapter/system-prompt, depth-3=this thread),
    and turns within a thread arrive with growing hash_ids and timestamps."""
    threads = defaultdict(list)
    for i, r in enumerate(records):
        h = r["hash_ids"]
        key = tuple(h[:3]) if len(h) >= 3 else tuple(h)
        threads[key].append(i)
    return threads


def load_mooncake_conversations(name="conversation_trace", seed=0,
                                min_adapter_group=MIN_ADAPTER_GROUP):
    """Convert the Mooncake trace into (conversations, requests) in core.py's
    shape. Adapter identity = depth-2 hash node, remapped to small contiguous
    ints. A conversation whose depth-2 node has fewer than
    `min_adapter_group` requesters gets its OWN unique adapter id -- it
    genuinely does not share a system prompt with anyone, so treating it as a
    private one-off adapter is the honest mapping. (An earlier version pooled
    all such conversations into one shared "overflow" bucket, which fabricated
    a single fake mega-adapter used by ~60% of conversations -- a bug, not a
    finding. Fixed here.)"""
    records = _load_mooncake_raw(name)
    threads = _group_into_conversations(records)

    depth2_count = defaultdict(int)
    for r in records:
        if len(r["hash_ids"]) > 1:
            depth2_count[r["hash_ids"][1]] += 1
    adapter_remap = {}
    next_id = 0
    for node, count in depth2_count.items():
        if count >= min_adapter_group:
            adapter_remap[node] = next_id
            next_id += 1
    # ids for private, unshared conversations are handed out below, one per
    # conversation, continuing from next_id.

    conversations = []
    requests = []
    rid = 0
    private_id = next_id
    for cid, (key, idxs) in enumerate(sorted(threads.items(),
                                             key=lambda kv: records[kv[1][0]]["timestamp"])):
        idxs = sorted(idxs, key=lambda i: records[i]["timestamp"])
        first = records[idxs[0]]
        h = first["hash_ids"]
        node = h[1] if len(h) > 1 else None
        if node is not None and node in adapter_remap:
            adapter = adapter_remap[node]
        else:
            adapter = private_id      # unique, unshared adapter for this conversation
            private_id += 1
        convo = Conversation(adapter, cid)

        # deepest shared prefix across this thread's turns: blocks in the
        # LAST turn's hash chain that also appear as a depth-2+ shared node
        # elsewhere (i.e. more than just this thread uses it). Approximated
        # here as: total blocks in the last turn's chain that are shared
        # (count >= 2 across the whole trace) times 512 tokens.
        last = records[idxs[-1]]
        shared_blocks = sum(1 for h in last["hash_ids"] if depth2_count.get(h, 0) >= 2
                            or h == first["hash_ids"][0])
        convo.prefix_group = adapter          # one group per adapter here
        convo.prefix_tokens = shared_blocks * BLOCK_TOKENS
        convo.prefix_key = (adapter, convo.prefix_group) if convo.prefix_tokens > 0 else None

        for turn_idx, i in enumerate(idxs):
            rec = records[i]
            r = Request(rid, cid, rec["output_length"], float(rec["timestamp"]))
            r.is_first_turn = (turn_idx == 0)
            r.input_length = rec["input_length"]     # extra: real prompt size, unused by core Request
            requests.append(r)
            rid += 1
        conversations.append(convo)

    return conversations, requests


# ==========================================================================
# Taskmaster -- real task-type adapter identity (ELORA's own third dataset)
# ==========================================================================

# rough words -> tokens ratio for a quick, defensible proxy; real tokenizers
# vary, but this is in the right ballpark for English chat text.
WORDS_TO_TOKENS = 1.3


def load_taskmaster_conversations(path=None):
    """Convert Google Taskmaster (TM-1, self-dialogs) into core.py's
    (conversations, requests) shape.

    This is one of ELORA's OWN three evaluation datasets ("Personal Agents").
    Unlike Mooncake, Taskmaster ships a genuine task-type field --
    `instruction_id` (e.g. "restaurant-table-2", "pizza-ordering-1") -- which
    is a much more natural adapter-identity proxy than anything we inferred
    from Mooncake's hash structure: it is literally "which specialized task
    this conversation needs," directly analogous to "which specialized LoRA
    this customer's request needs." 15 distinct task types across 7,708 real
    dialogs, with a real, substantial popularity skew (top type used by 1,211
    conversations, versus Mooncake's most-popular-adapter count of just 7) --
    a genuinely different, more ELORA-shaped regime than the Mooncake trace.

    Taskmaster has NO arrival timestamps. Neither did ELORA's own copy of it:
    their paper states plainly they "adopt query arrival patterns from the
    Microsoft Azure function trace" for exactly this dataset. We do the same
    thing they did -- overlay real BurstGPT arrival timing (see
    build_real_workload in run_on_real_traces.py) -- rather than inventing a
    new arrival model, so this reconstruction follows ELORA's OWN documented
    method for handling this dataset's missing field, not a method we made up.

    Adds to each Conversation:
      .prefix_group / .prefix_tokens / .prefix_key : NOT populated (no
        cross-conversation prefix-sharing signal exists in Taskmaster the way
        it does in Mooncake's hash_ids). Set to 0/None. Phase 6/7 prefix-tree
        behavior on this trace reduces to "no sharing" -- an honest gap, not
        a fabricated one.
    Adds to each Request:
      .is_first_turn : bool
    """
    path = path or os.path.join(TRACES_DIR, "taskmaster_tm1.json")
    dialogs = json.load(open(path))

    conversations = []
    requests = []
    rid = 0
    adapter_remap = {}
    next_id = 0
    for cid, dlg in enumerate(dialogs):
        task = dlg["instruction_id"]
        if task not in adapter_remap:
            adapter_remap[task] = next_id
            next_id += 1
        adapter = adapter_remap[task]
        convo = Conversation(adapter, cid)
        convo.prefix_group = 0
        convo.prefix_tokens = 0
        convo.prefix_key = None

        turn_idx = 0
        pending_output = None
        for u in dlg["utterances"]:
            if u["speaker"] == "USER":
                # a USER utterance opens a turn; its output is whatever the
                # ASSISTANT says next (or a small default if the dialog ends
                # on a user turn).
                pending_output = None
            elif u["speaker"] == "ASSISTANT" and pending_output is None:
                out_tokens = max(1, round(len(u["text"].split()) * WORDS_TO_TOKENS))
                # placeholder arrival_time=turn_idx; real timing is overlaid
                # by build_real_workload() using BurstGPT gaps, matching
                # ELORA's own documented method for this dataset.
                r = Request(rid, cid, out_tokens, float(turn_idx))
                r.is_first_turn = (turn_idx == 0)
                requests.append(r)
                rid += 1
                turn_idx += 1
                pending_output = True
        if turn_idx == 0:
            # a dialog with no assistant replies at all (rare edge case):
            # give it one placeholder turn so it isn't silently dropped
            r = Request(rid, cid, 10, 0.0)
            r.is_first_turn = True
            requests.append(r)
            rid += 1
        conversations.append(convo)

    return conversations, requests


# ==========================================================================
# Summary / sanity check when run standalone
# ==========================================================================

def main():
    print("=== BurstGPT arrivals (ChatGPT rows) ===")
    times = load_burstgpt_arrivals(limit=50000)
    gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    import statistics
    print(f"  {len(times)} requests, span {times[-1]/1000/3600:.1f}h")
    print(f"  median gap {statistics.median(gaps):.0f}ms, "
          f"mean {statistics.mean(gaps):.0f}ms, "
          f"burstiness (stdev/mean) {statistics.stdev(gaps)/statistics.mean(gaps):.2f}")

    print("\n=== Mooncake conversation_trace ===")
    convos, reqs = load_mooncake_conversations()
    print(f"  {len(convos)} conversations, {len(reqs)} requests")
    n_adapters = len(set(c.adapter_id for c in convos))
    print(f"  {n_adapters} distinct adapters (depth-2 nodes + overflow)")
    turns = [0] * len(convos)
    for r in reqs:
        turns[r.conversation_id] += 1
    print(f"  turns per conversation: min {min(turns)} max {max(turns)} "
          f"mean {sum(turns)/len(turns):.1f}")
    prefixed = [c for c in convos if c.prefix_tokens > 0]
    print(f"  conversations with a shared prefix: {len(prefixed)}/{len(convos)}, "
          f"mean shared tokens {sum(c.prefix_tokens for c in prefixed)/max(1,len(prefixed)):.0f}")

    print("\n=== Taskmaster TM-1 (self-dialogs) ===")
    tconvos, treqs = load_taskmaster_conversations()
    print(f"  {len(tconvos)} conversations, {len(treqs)} requests")
    tn_adapters = len(set(c.adapter_id for c in tconvos))
    from collections import Counter
    tpop = Counter(c.adapter_id for c in tconvos)
    print(f"  {tn_adapters} distinct adapters (real task types)")
    print(f"  top 5 adapter popularity: {tpop.most_common(5)}")
    tturns = [0] * len(tconvos)
    for r in treqs:
        tturns[r.conversation_id] += 1
    print(f"  turns per conversation: min {min(tturns)} max {max(tturns)} "
          f"mean {sum(tturns)/len(tturns):.1f}")


if __name__ == "__main__":
    main()
