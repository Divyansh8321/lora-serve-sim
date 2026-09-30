"""Load real production traces and convert them into the same (conversations,
requests) shape core.py's synthetic generators produce -- so every phase can
run, unmodified, on real arrival timing and real prefix-sharing structure
instead of our invented formulas.

Four traces, each fixing one specific gap named in the project's honesty list:

  LMSYS Chatbot Arena (huggingface.co/datasets/lmsys/chatbot_arena_conversations)
    ELORA'S OWN "CHATBOT" EVALUATION DATASET, named directly in their paper
    as "LMSYS-33k". 33,000 real arena battles, REAL timestamps, 20 real
    named models ("20 SOTA models such as GPT-4, Claude, and LLaMA-based
    Vicuna" -- ELORA's own words) used directly as adapter identity. No
    inference, no missing-field workaround -- the closest thing we have to
    literally ELORA's setup. Gated on HuggingFace (free login + accept
    terms); see REAL_DATA_FINDINGS.md for how to obtain it.

  BurstGPT (github.com/HPMLL/BurstGPT, CC-BY-4.0)
    10M+ real ChatGPT/GPT-4 request logs from Azure OpenAI, arrival timestamp
    per row. Replaces make_bursty_workload's invented "5-10x rate spike for
    25% of a 40s window" with REAL inter-arrival gaps.

  Mooncake FAST'25 trace (github.com/kvcache-ai/Mooncake, traces/*.jsonl)
    Real anonymized request trace built specifically to study KV-cache
    sharing: each request lists `hash_ids`, the 512-token prefix blocks its
    prompt matches. Matching hash_ids across requests = shareable KV cache.
    Replaces make_prefix_sharing_workload's invented "3 fixed-size groups"
    with a DERIVED prefix length -- see the WARNING in
    load_mooncake_conversations: it collapses to a constant in practice,
    and staleness results are highly sensitive to it. No LoRA-adapter
    field -- adapter identity is inferred from the depth-2 hash node (see
    load_mooncake_conversations for the exact mapping and its limitation).

  Google Taskmaster TM-1 (github.com/google-research-datasets/Taskmaster)
    ONE OF ELORA'S OWN THREE EVALUATION DATASETS ("Personal Agents"). Real
    task-oriented dialogs with a genuine task-type field (`instruction_id`),
    used directly as adapter identity -- no inference needed, unlike
    Mooncake. See load_taskmaster_conversations for detail, including how we
    handle its missing timestamps the same way ELORA's own paper says they
    did (overlay Azure-trace-style arrival timing -- here, BurstGPT).

CRITICAL LIMITATION -- READ BEFORE TRUSTING ANY STALENESS NUMBER:

Mooncake's real prefix chains ARE variable-depth and DO grow turn-by-turn,
but our derivation of a single prefix_tokens per conversation does NOT
preserve that. Measured on the actual shipped data (960-conversation slice,
min_adapter_reuse=2): prefix_tokens is a CONSTANT 1024 for every single
conversation -- min == median == max. The filter
`depth2_count[h] >= 2 or h == root` matches only 1-2 blocks per chain in
practice (the universal root hash that literally every request shares,
plus at most one genuinely-shared depth-2 block), and after the reuse
filter every survivor lands on exactly 2 blocks.

Worse, the simulator's staleness output is HIGHLY SENSITIVE to this
derived constant. Same slice, same pool, same policy, only prefix_tokens
changed:
    prefix_tokens=1024 -> 42.5% stale      (the value we ship)
    prefix_tokens= 512 -> 53.7% stale
    prefix_tokens=   0 ->  0.0% stale

So Mooncake staleness figures are a function of OUR DERIVATION, not a
measurement of Mooncake's real sharing structure. The fact that 1024
happens to yield ~42.4% -- numerically close to ELORA's reported 42.4% --
is a COINCIDENCE of the constant we chose, and must not be presented as
independent corroboration. See REAL_DATA_FINDINGS.md.

Run standalone to print summary stats:  python real_traces.py
"""

import csv
import json
import os
from collections import defaultdict

from core import Conversation, Request

try:
    import pyarrow.parquet as _pq
except ImportError:
    _pq = None

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
    finding. Fixed here.)

    WARNING -- prefix_tokens derived below is effectively a CONSTANT (1024)
    for every conversation after filtering, NOT the variable per-conversation
    sharing depth the module docstring's earlier drafts claimed. Staleness
    results are highly sensitive to this constant (1024->42.5%, 512->53.7%,
    0->0.0%). Any staleness number from this loader reflects our derivation,
    not a measurement of Mooncake's real prefix structure. See the module
    docstring's CRITICAL LIMITATION section."""
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
# LMSYS Chatbot Arena -- ELORA's own "Chatbot" dataset, real timestamps
# ==========================================================================

def load_lmsys_conversations(path=None, limit=None):
    """Convert LMSYS Chatbot Arena conversations
    (huggingface.co/datasets/lmsys/chatbot_arena_conversations) into
    core.py's (conversations, requests) shape.

    THIS IS ELORA'S OWN "CHATBOT" EVALUATION DATASET -- the paper names
    "LMSYS-33k" directly. Of the three real datasets in this project, this
    is the closest we get to literally what ELORA evaluated on.

    Format: each row is one ARENA BATTLE -- two models (model_a, model_b)
    each answer the SAME prompt sequence, judged head-to-head. We treat each
    side of the battle as an independent real conversation, using the real
    model name as adapter identity: "20 SOTA models such as GPT-4, Claude,
    and LLaMA-based Vicuna" (ELORA's own description) is a real, named,
    ground-truth adapter identity -- no inference needed, unlike Mooncake,
    and no missing-field workaround needed, unlike Taskmaster.

    Unlike Taskmaster, this dataset carries REAL TIMESTAMPS (`tstamp`, a
    Unix time per battle) -- no BurstGPT overlay needed. Real arrival timing
    AND real adapter identity, both directly from the source, on the one
    dataset that's actually ELORA's.

    Requires `pyarrow` (pip install pyarrow) to read the .parquet file, and
    the file itself, which is gated -- see README/REAL_DATA_FINDINGS.md for
    how to obtain it (free HuggingFace login + accepting the dataset's terms).

    Adds to each Conversation:
      .prefix_group / .prefix_tokens / .prefix_key : NOT populated. Arena
        battles share a prompt PREFIX ACROSS conversations that use
        DIFFERENT adapters (model_a and model_b see the same first message)
        -- the opposite of our prefix-tree model's assumption (same adapter,
        shared prefix). Left at 0/None; an honest gap, not fabricated.
    Adds to each Request:
      .is_first_turn : bool
    """
    if _pq is None:
        raise ImportError("load_lmsys_conversations needs pyarrow: pip install pyarrow")
    path = path or os.path.join(TRACES_DIR, "lmsys_arena.parquet")
    cols = ["model_a", "model_b", "conversation_a", "conversation_b", "tstamp"]
    table = _pq.read_table(path, columns=cols)
    rows = table.to_pylist()
    if limit:
        rows = rows[:limit]

    adapter_remap = {}
    next_adapter_id = 0

    def adapter_for(model_name):
        nonlocal next_adapter_id
        if model_name not in adapter_remap:
            adapter_remap[model_name] = next_adapter_id
            next_adapter_id += 1
        return adapter_remap[model_name]

    conversations = []
    requests = []
    rid = 0
    cid = 0
    for row in rows:
        t0_ms = row["tstamp"] * 1000.0
        for side, model_key, convo_key in [("a", "model_a", "conversation_a"),
                                           ("b", "model_b", "conversation_b")]:
            adapter = adapter_for(row[model_key])
            convo = Conversation(adapter, cid)
            convo.prefix_group = 0
            convo.prefix_tokens = 0
            convo.prefix_key = None

            turn_idx = 0
            for msg in row[convo_key]:
                if msg.get("role") != "assistant":
                    continue
                out_tokens = max(1, round(len(msg["content"].split()) * WORDS_TO_TOKENS))
                # all turns of one battle land at (approximately) the same
                # real arrival time -- the arena logs one tstamp per battle,
                # not per message. Nudge subsequent turns forward by 1ms each
                # so ordering within a conversation stays well-defined.
                r = Request(rid, cid, out_tokens, t0_ms + turn_idx)
                r.is_first_turn = (turn_idx == 0)
                requests.append(r)
                rid += 1
                turn_idx += 1
            if turn_idx == 0:
                r = Request(rid, cid, 10, t0_ms)
                r.is_first_turn = True
                requests.append(r)
                rid += 1
            conversations.append(convo)
            cid += 1

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

    if _pq is not None and os.path.exists(os.path.join(TRACES_DIR, "lmsys_arena.parquet")):
        print("\n=== LMSYS Chatbot Arena (ELORA's own 'Chatbot' dataset) ===")
        lconvos, lreqs = load_lmsys_conversations()
        print(f"  {len(lconvos)} conversations, {len(lreqs)} requests")
        ln_adapters = len(set(c.adapter_id for c in lconvos))
        from collections import Counter
        lpop = Counter(c.adapter_id for c in lconvos)
        print(f"  {ln_adapters} distinct adapters (real model names)")
        print(f"  top 5 adapter popularity: {lpop.most_common(5)}")
        lturns = [0] * len(lconvos)
        for r in lreqs:
            lturns[r.conversation_id] += 1
        print(f"  turns per conversation: min {min(lturns)} max {max(lturns)} "
              f"mean {sum(lturns)/len(lturns):.1f}")
        ts = sorted(r.arrival_time for r in lreqs)
        print(f"  real arrival span: {(ts[-1]-ts[0])/1000/3600/24:.1f} days")
    else:
        print("\n=== LMSYS Chatbot Arena: skipped (no traces/lmsys_arena.parquet, "
              "or pyarrow not installed) ===")


if __name__ == "__main__":
    main()
