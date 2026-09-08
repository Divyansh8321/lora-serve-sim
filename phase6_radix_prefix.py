"""PHASE 6 -- RadixAttention prefix tree. Does dependency-awareness finally pay?

WORLD: unified pool + continuous batching (phase 5's substrate), plus a
       RADIX PREFIX TREE for KV. SGLang's RadixAttention and ELORA's
       "dependency-aware cache manager" both store KV in a tree keyed by token
       prefix: a new request walks from the root matching its prompt tokens,
       and every node on the matched path is KV it reuses FOR FREE.

       ELORA's structure exactly: the tree's top layer is LoRAs; KV prefix
       nodes hang inside a LoRA's subtree. Two conversations share a prefix
       only if they use the same adapter AND the same system prompt -- because
       a LoRA rewrites the KV. make_prefix_sharing_workload models this with
       `shared_prefix_groups` groups; conversations in the same (adapter,
       group) pair share their first `shared_prefix_tokens` tokens.

WHY THIS PHASE EXISTS:
       Phase 4 found: once the pool is unified, crude "evict KV before
       adapters" ORDERING performs identically to full dependency-aware
       scoring. Phase 5 (squeezed) found ordering is even BETTER than the
       dependency-aware policy. This contradicts ELORA's ELORA-WOM ablation
       (dependency manager removed -> 1.51x worse TTFT).

       The suspected reconciliation: ELORA's dependency tree earns its keep
       through PREFIX SHARING, which our workload has never had. Evicting the
       wrong LoRA now strands a prefix that MANY sequences wanted -- and
       "evict the biggest single thing" does not see that. This phase adds
       prefix sharing and re-runs the policy comparison.

QUESTION: as `shared_prefix_tokens` grows 0 -> 1600, does a dependency-aware
       eviction policy open a gap over plain ordering?

POLICIES (all on the unified pool, continuous batching):
  lru-leaf      : evict least-recently-used leaf, ignore which LoRA it is under
  ordering      : phase-4 "kv-first" -- evict the biggest KV subtree before
                  touching any adapter; within KV, biggest-then-LRU
  dep-aware     : ELORA-style -- never evict a LoRA whose subtree still holds
                  recently-used KV; rank KV leaves by (is-stale, -shared_by,
                  size, LRU) so a widely-shared prefix is evicted last

FINDING (mean over 5 seeds, adapter 180MB, 3 prefix groups):

  GENEROUS pool (3000MB), varying shared prefix length:
    prefix tok   policy       p50   TTFT p50   stale KV   dep vs ordering
         0       (all three)  689      23        0.0%        0.0%
       400       ordering     732      29        0.0%
       400       dep-aware    715      32        0.0%       +2.2%
       800       ordering     842     102        0.0%
       800       dep-aware    785      38        0.0%       +6.7%
      1600       ordering    1214     277        0.4%
      1600       dep-aware   1160     220        0.1%       +4.5%
      3200       ordering    5583    2061        4.0%
      3200       dep-aware   5327    1837        3.4%       +4.6%
      6400       ordering   39747   36007        4.0%
      6400       dep-aware  39737   36007        3.9%       +0.0%

  The dependency-aware advantage is a MID-RANGE effect. It needs the shared
  prefix big enough to matter (>400 tok) but small enough that keeping it
  resident is FEASIBLE (roughly <10% of pool). At 6400 tok / 800MB on a
  3000MB pool, neither policy can hold the prefix -- the system thrashes
  (p50 ~40s, 36k recomputes) and dep-aware == ordering because there is no
  good decision left to make. Outside the band -- too small to matter, or
  too big to save -- the sophisticated policy earns nothing.

  TWO results, and the smaller one is ELORA's:

  1. PREFIX-STRUCTURE AWARENESS MATTERS A LOT. `lru-leaf` -- blind LRU over
     leaves -- collapses as the shared prefix grows: p50 884 at 400 tokens,
     28553 at 1600. It evicts shared prefix nodes that many sequences need,
     forcing everyone to re-prefill. Any policy that does NOT blindly
     LRU-evict a shared node avoids this -- including `ordering`, which
     evicts biggest-single-node-first (a shared prefix under an active
     adapter is neither the biggest node nor unpinned).

  2. THE DEPENDENCY SCORING ELORA LAYERS ON TOP IS WORTH ~2-7%. dep-aware
     beats ordering by +2% to +7% p50 (and much better TTFT: 38 vs 102 at
     800 tokens) by protecting an adapter whose subtree still holds live
     shared KV and evicting widely-shared prefixes last. Real, consistent,
     grows with sharing -- but nowhere near ELORA's claimed 1.51x for their
     full dependency manager vs ELORA-WOM.

  RECONCILIATION with ELORA-WOM's 1.51x: ELORA-WOM removes the dependency
  manager ENTIRELY -- it is closer to our `lru-leaf` (which DOES collapse,
  ~40x) than to `ordering`. ELORA's paper never isolates "prefix-aware
  ordering, no dependency scoring", which is the cheap policy that captures
  most of the win here. Honest read: the pathology is real, prefix structure
  must be respected, but simple ordering discipline -- not elaborate
  dependency scoring -- does the heavy lifting.

  SQUEEZED pool (1800MB, the default): same ranking, gap compressed to
  +0.6-2.7% because eviction is so frequent all policies thrash. lru-leaf
  still collapses.

HARDWARE PROFILE (--hw): phase 6 has no CPU/PCIe KV path, so --hw only moves
  decode/prefill/swap-cold/adapter. At --hw elora-aggressive the dep-aware vs
  ordering gap SHRINKS (800 tok: +6.7% -> +2.2%): faster compute makes the
  recompute penalty that dep-aware avoids cheaper, so protecting shared
  prefixes matters less. lru-leaf still degrades under sharing but less
  catastrophically (315 -> 271 at 800 tok, vs 884 -> 785 at --hw ours).
  Reading: dependency-scoring's value is INVERSELY related to hardware speed
  -- another reason ELORA's 1.51x (measured on H800) does not transfer to a
  slower single GPU as a policy claim.

Run:  python phase6_radix_prefix.py
      python phase6_radix_prefix.py --hw elora-aggressive --prefix-tokens 0 400 800
      python phase6_radix_prefix.py --prefix-tokens 0 400 800 1600 --pool-mb 3000
"""

import argparse
from collections import deque

from dataclasses import replace as _replace
from core import (MB_PER_TOKEN, PREFILL_MS_PER_TOKEN, DECODE_MS_PER_TOKEN,
                  SWAP_COLD_MS, ADAPTER_MB, HardwareProfile, OURS, get_profile,
                  make_prefix_sharing_workload,
                  percentile)

STEP_TOKEN_BUDGET = 512
FIXED_STEP_MS = 4.0
PER_SEQ_STEP_MS = 0.05
MAX_BATCH_SEQS = 192
WAIT_THRESHOLD_MS = 30000


# ==========================================================================
# Radix prefix tree
# ==========================================================================

class RadixNode:
    """A run of tokens shared by everything in its subtree.

    The tree has one root per (adapter_id, prefix_group). Under that root:
      depth 1 : the shared system-prompt node (prefix_tokens long)
      deeper  : per-conversation continuations (history + generated tokens)

    We do not model token identity -- a conversation's continuation is keyed by
    its cid, and the shared prefix is keyed by the (adapter, group) root. That
    is enough to get the reuse accounting right.
    """
    __slots__ = ("key", "tokens", "adapter_id", "parent", "children",
                 "last_used", "refs", "is_prefix")

    def __init__(self, key, tokens, adapter_id, parent, is_prefix):
        self.key = key
        self.tokens = tokens              # token count this node covers
        self.adapter_id = adapter_id
        self.parent = parent
        self.children = {}
        self.last_used = 0.0
        self.refs = 0                     # active sequences holding this path
        self.is_prefix = is_prefix        # True = shared system prompt node

    def mb(self):
        return self.tokens * MB_PER_TOKEN

    def is_leaf(self):
        return not self.children


class RadixTree:
    def __init__(self):
        self.roots = {}                   # (adapter_id, group) -> prefix RadixNode
        self.convo_leaf = {}              # cid -> RadixNode (its continuation)
        self.resident = set()             # id(node) for nodes whose KV is on GPU
        self._nodes = []                  # every node ever created (small: <100)
        self._resident_mb = 0.0           # kept incrementally

    # ---- structure ----
    def get_prefix_node(self, adapter_id, group, prefix_tokens):
        k = (adapter_id, group)
        n = self.roots.get(k)
        if n is None:
            n = RadixNode(k, prefix_tokens, adapter_id, None, is_prefix=True)
            self.roots[k] = n
            self._nodes.append(n)
        return n

    def get_convo_leaf(self, cid, adapter_id, group, prefix_tokens):
        n = self.convo_leaf.get(cid)
        if n is None:
            parent = (self.get_prefix_node(adapter_id, group, prefix_tokens)
                      if prefix_tokens > 0 else None)
            n = RadixNode(("c", cid), 0, adapter_id, parent, is_prefix=False)
            if parent is not None:
                parent.children[cid] = n
            self.convo_leaf[cid] = n
            self._nodes.append(n)
        return n

    # ---- residency (mb kept incrementally; call resize() before changing tokens) ----
    def is_resident(self, node):
        return id(node) in self.resident

    def make_resident(self, node):
        if id(node) not in self.resident:
            self.resident.add(id(node))
            self._resident_mb += node.mb()

    def evict(self, node):
        if id(node) in self.resident:
            self.resident.discard(id(node))
            self._resident_mb -= node.mb()

    def resize(self, node, new_tokens):
        """Change a resident node's token count, keeping _resident_mb correct."""
        if id(node) in self.resident:
            self._resident_mb += (new_tokens - node.tokens) * MB_PER_TOKEN
        node.tokens = new_tokens

    def resident_mb(self):
        return self._resident_mb

    def _all_nodes(self):
        return self._nodes

    # ---- the match: how much of this request's prompt is already on GPU ----
    def matched_mb(self, cid, adapter_id, group, prefix_tokens, history_tokens):
        """Tokens the request can reuse for free = resident prefix + resident
        own-continuation. Returns (reused_mb, needed_mb, [nodes_on_path])."""
        path = []
        reused = 0.0
        needed = 0.0
        if prefix_tokens > 0:
            p = self.get_prefix_node(adapter_id, group, prefix_tokens)
            path.append(p)
            if self.is_resident(p):
                reused += p.mb()
            else:
                needed += p.mb()
        leaf = self.get_convo_leaf(cid, adapter_id, group, prefix_tokens)
        self.resize(leaf, history_tokens)
        path.append(leaf)
        if self.is_resident(leaf):
            reused += leaf.mb()
        else:
            needed += leaf.mb()
        return reused, needed, path

    # ---- shared_by: how many conversations depend on this prefix node ----
    def shared_by(self, node):
        if not node.is_prefix:
            return 1
        return max(1, len(node.children))


# ==========================================================================
# Sequence state
# ==========================================================================

class SeqState:
    __slots__ = ("req", "cid", "adapter_id", "prefill_left", "decode_left",
                 "phase", "leaf")

    def __init__(self, req, cid, adapter_id, to_prefill, leaf):
        self.req = req
        self.cid = cid
        self.adapter_id = adapter_id
        self.prefill_left = to_prefill
        self.decode_left = req.output_tokens
        self.phase = "prefill" if to_prefill > 0 else "decode"
        self.leaf = leaf

    def done(self):
        return self.prefill_left <= 0 and self.decode_left <= 0


# ==========================================================================
# Policies -- all differ ONLY in choose_evictions()
# ==========================================================================

class Policy:
    name = "base"

    def __init__(self):
        self.a_last = {}     # adapter_id -> last used
        self.n_last = {}     # id(node) -> last used

    def touch_adapter(self, a, now):
        self.a_last[a] = now

    def touch_node(self, node, now):
        node.last_used = now
        self.n_last[id(node)] = now

    # returns a list of ('adapter', id) / ('kv', node) to free, in order
    def choose_evictions(self, tree, adapters_mb, need_mb, pinned_adapters,
                         pinned_nodes):
        raise NotImplementedError


class LRULeaf(Policy):
    """Evict the LRU resident leaf, regardless of which LoRA it belongs to.
    If still short, evict LRU adapters. The naive radix baseline."""
    name = "lru-leaf"

    def choose_evictions(self, tree, adapters_mb, need_mb, pinned_adapters,
                         pinned_nodes):
        ev, freed = [], 0.0
        leaves = [n for n in tree._all_nodes()
                  if tree.is_resident(n) and n.is_leaf()
                  and id(n) not in pinned_nodes]
        for n in sorted(leaves, key=lambda x: x.last_used):
            if freed >= need_mb:
                return ev
            ev.append(("kv", n)); freed += n.mb()
        for a in sorted([a for a in adapters_mb if a not in pinned_adapters],
                        key=lambda a: self.a_last.get(a, -1)):
            if freed >= need_mb:
                return ev
            ev.append(("adapter", a)); freed += adapters_mb[a]
        return ev


class OrderingOnly(Policy):
    """Phase-4 'kv-first': evict the biggest resident KV node before touching
    any adapter. Within KV, biggest first then LRU. No notion of sharing."""
    name = "ordering"

    def choose_evictions(self, tree, adapters_mb, need_mb, pinned_adapters,
                         pinned_nodes):
        ev, freed = [], 0.0
        kv = [n for n in tree._all_nodes()
              if tree.is_resident(n) and id(n) not in pinned_nodes]
        for n in sorted(kv, key=lambda x: (-x.mb(), x.last_used)):
            if freed >= need_mb:
                return ev
            ev.append(("kv", n)); freed += n.mb()
        for a in sorted([a for a in adapters_mb if a not in pinned_adapters],
                        key=lambda a: self.a_last.get(a, -1)):
            if freed >= need_mb:
                return ev
            ev.append(("adapter", a)); freed += adapters_mb[a]
        return ev


class DepAware(Policy):
    """ELORA-style. Two ideas:
      1. never evict a LoRA whose subtree still holds recently-used resident KV
         (an adapter with live dependents is protected);
      2. rank KV leaves so a WIDELY-SHARED prefix is evicted last:
         key = (is-stale, -shared_by, size_is_small, LRU)
         -- stale KV first, then keep high-fan-out prefixes, then small, then LRU.
    """
    name = "dep-aware"

    def _subtree_recent_kv(self, tree, adapter_id, now, horizon=20000):
        for n in tree._all_nodes():
            if n.adapter_id != adapter_id:
                continue
            if tree.is_resident(n) and (now - n.last_used) < horizon:
                return True
        return False

    def choose_evictions(self, tree, adapters_mb, need_mb, pinned_adapters,
                         pinned_nodes, now=0.0):
        ev, freed = [], 0.0
        kv = [n for n in tree._all_nodes()
              if tree.is_resident(n) and id(n) not in pinned_nodes]

        def rank(n):
            stale = self._adapter_resident(tree, adapters_mb, n.adapter_id)
            return (
                0 if not stale else 1,          # stale (adapter gone) first
                tree.shared_by(n),              # low fan-out evicted before high
                n.mb(),                         # small before big  (opposite of ordering!)
                n.last_used,
            )

        for n in sorted(kv, key=rank):
            if freed >= need_mb:
                return ev
            ev.append(("kv", n)); freed += n.mb()

        # adapters last, and only those without recent resident dependents
        prot = set(pinned_adapters)
        for a in list(adapters_mb):
            if a not in prot and self._subtree_recent_kv(tree, a, now):
                prot.add(a)
        for a in sorted([a for a in adapters_mb if a not in prot],
                        key=lambda a: self.a_last.get(a, -1)):
            if freed >= need_mb:
                return ev
            ev.append(("adapter", a)); freed += adapters_mb[a]
        # if STILL short, fall back to evicting protected adapters LRU-first
        for a in sorted([a for a in adapters_mb
                         if a not in pinned_adapters and a not in
                         {x for k, x in ev if k == "adapter"}],
                        key=lambda a: self.a_last.get(a, -1)):
            if freed >= need_mb:
                return ev
            ev.append(("adapter", a)); freed += adapters_mb[a]
        return ev

    @staticmethod
    def _adapter_resident(tree, adapters_mb, adapter_id):
        return adapter_id in adapters_mb


POLICIES = {p.name: p for p in [LRULeaf, OrderingOnly, DepAware]}


# ==========================================================================
# Simulator: unified pool + continuous batching + radix tree
# ==========================================================================

class RadixSim:
    def __init__(self, requests, conversations, pool_mb, policy, max_loras,
                 prefix_tokens, hw=OURS):
        hw = get_profile(hw)
        self.hw = hw
        self.adapter_mb = hw.adapter_mb
        self.prefill_ms_per_token = hw.prefill_ms_per_token
        self.decode_ms_per_token = hw.decode_ms_per_token
        self.swap_cold_ms = hw.swap_cold_ms
        self.pool_mb = pool_mb
        self.policy = policy
        self.max_loras = max_loras
        self.prefix_tokens = prefix_tokens
        self.tree = RadixTree()
        self.adapters_mb = {}            # adapter_id -> ADAPTER_MB  (resident set)
        self.incoming = deque(sorted(requests, key=lambda r: r.arrival_time))
        self.convos = {c.conversation_id: c for c in conversations}

        self.now = 0.0
        self.waiting = deque()
        self.batch = []
        self.finished = []

        self.adapter_loads = 0
        self.recompute_tokens = 0
        self.reused_tokens = 0
        self.prefill_tokens = 0
        self.steps = 0
        self.stale_samples = []
        self.ttft = {}
        self.tpot_num = 0.0
        self.tpot_den = 0

    # ---------- helpers ----------
    def admit_arrivals(self):
        while self.incoming and self.incoming[0].arrival_time <= self.now:
            self.waiting.append(self.incoming.popleft())

    def used_mb(self):
        return sum(self.adapters_mb.values()) + self.tree.resident_mb()

    def free_mb(self):
        return self.pool_mb - self.used_mb()

    def pinned_adapters(self):
        return {s.adapter_id for s in self.batch}

    def pinned_nodes(self):
        p = set()
        for s in self.batch:
            n = s.leaf
            while n is not None:
                p.add(id(n))
                n = n.parent
        return p

    def stale_pct(self):
        total = self.tree.resident_mb()
        if total == 0:
            return 0.0
        stale = 0.0
        for n in self.tree._all_nodes():
            if self.tree.is_resident(n) and n.adapter_id not in self.adapters_mb:
                stale += n.mb()
        return 100.0 * stale / total

    # ---------- eviction ----------
    def make_room(self, need_mb):
        if self.free_mb() >= need_mb - 1e-9:
            return
        deficit = need_mb - self.free_mb()
        pol = self.policy
        kw = {}
        if isinstance(pol, DepAware):
            kw["now"] = self.now
        plan = pol.choose_evictions(self.tree, self.adapters_mb, deficit,
                                    self.pinned_adapters(), self.pinned_nodes(),
                                    **kw)
        for kind, obj in plan:
            if self.free_mb() >= need_mb - 1e-9:
                break
            if kind == "adapter":
                self.adapters_mb.pop(obj, None)
            else:
                self.tree.evict(obj)

    # ---------- ensure resident ----------
    def ensure_adapter(self, a):
        if a in self.adapters_mb:
            return 0.0
        self.make_room(self.adapter_mb)
        if self.free_mb() < self.adapter_mb - 1e-9:
            return 0.0
        self.adapters_mb[a] = self.adapter_mb
        self.adapter_loads += 1
        return self.swap_cold_ms

    # ---------- scheduling ----------
    def schedule(self):
        if not self.waiting:
            return 0.0
        self.waiting = deque(sorted(self.waiting, key=lambda r: r.arrival_time))
        deferred = deque()
        swap_ms = 0.0

        while self.waiting:
            if len(self.batch) >= MAX_BATCH_SEQS:
                break
            r = self.waiting[0]
            starving = (self.now - r.arrival_time) > WAIT_THRESHOLD_MS
            convo = self.convos[r.conversation_id]
            a = convo.adapter_id
            group = getattr(convo, "prefix_group", 0)
            ptoks = getattr(convo, "prefix_tokens", 0)

            batch_adapters = {s.adapter_id for s in self.batch}
            if a not in batch_adapters and len(batch_adapters) >= self.max_loras:
                if not starving:
                    deferred.append(self.waiting.popleft())
                    continue

            self.waiting.popleft()
            swap_ms += self.ensure_adapter(a)
            if a not in self.adapters_mb:
                deferred.append(r)
                continue

            history = convo.kv_cache_size          # tokens already generated
            reused, needed, path = self.tree.matched_mb(
                r.conversation_id, a, group, ptoks, history)

            # bring the unmatched part on-GPU (prefix + own continuation)
            self.make_room(needed)
            to_prefill = 0
            for n in path:
                if not self.tree.is_resident(n):
                    if self.free_mb() >= n.mb() - 1e-9:
                        self.tree.make_resident(n)
                        to_prefill += n.tokens
                    else:
                        # cannot cache this node: still must prefill its tokens
                        to_prefill += n.tokens
                self.policy.touch_node(n, self.now)

            reused_toks = int(round(reused / MB_PER_TOKEN))
            self.reused_tokens += reused_toks
            self.recompute_tokens += to_prefill

            leaf = path[-1]
            seq = SeqState(r, r.conversation_id, a, to_prefill, leaf)
            for n in path:
                n.refs += 1
            self.batch.append(seq)
            self.policy.touch_adapter(a, self.now)

        self.waiting.extendleft(reversed(deferred))
        return swap_ms

    def _force_admit_oldest(self):
        """Batch is empty and no arrivals remain but waiting is non-empty:
        admit the oldest waiter unconditionally (evict freely, ignore
        max_loras since the batch is empty so nothing is pinned)."""
        if not self.waiting:
            return 0.0
        r = self.waiting.popleft()
        convo = self.convos[r.conversation_id]
        a = convo.adapter_id
        group = getattr(convo, "prefix_group", 0)
        ptoks = getattr(convo, "prefix_tokens", 0)
        swap = self.ensure_adapter(a)
        if a not in self.adapters_mb:
            self.waiting.appendleft(r)
            return swap
        history = convo.kv_cache_size
        reused, needed, path = self.tree.matched_mb(
            r.conversation_id, a, group, ptoks, history)
        self.make_room(needed)
        to_prefill = 0
        for n in path:
            if not self.tree.is_resident(n):
                if self.free_mb() >= n.mb() - 1e-9:
                    self.tree.make_resident(n)
                to_prefill += n.tokens
            self.policy.touch_node(n, self.now)
        self.recompute_tokens += to_prefill
        seq = SeqState(r, r.conversation_id, a, to_prefill, path[-1])
        for n in path:
            n.refs += 1
        self.batch.append(seq)
        self.policy.touch_adapter(a, self.now)
        return swap

    # ---------- one step ----------
    def step(self):
        swap_ms = self.schedule()
        if not self.batch:
            if self.incoming:
                self.now = max(self.now, self.incoming[0].arrival_time)
                return
            if self.waiting:
                # nothing admitted and no more arrivals: the pool cannot fit the
                # oldest waiter even alone. Force it -- evict everything not
                # pinned by an (empty) batch -- and count the stall in its
                # latency by advancing a step's worth of time.
                swap_ms += self._force_admit_oldest()
                self.now += FIXED_STEP_MS + self.decode_ms_per_token + swap_ms
                if not self.batch:
                    # still impossible (pool < one adapter): drop it, unserved
                    r = self.waiting.popleft()
                    r.finish(self.now - r.arrival_time)
                    self.finished.append(r)
                return
            return

        n_decode = sum(1 for s in self.batch if s.phase == "decode")
        prefill_room = STEP_TOKEN_BUDGET - n_decode
        prefill_this_step = 0
        for s in self.batch:
            if s.phase == "prefill" and prefill_room > 0:
                chunk = min(s.prefill_left, prefill_room)
                s.prefill_left -= chunk
                prefill_this_step += chunk
                prefill_room -= chunk
                if s.prefill_left <= 0:
                    s.phase = "decode"

        step_ms = (FIXED_STEP_MS + PER_SEQ_STEP_MS * len(self.batch)
                   + self.decode_ms_per_token
                   + prefill_this_step * self.prefill_ms_per_token
                   + swap_ms)
        self.now += step_ms
        self.steps += 1
        self.prefill_tokens += prefill_this_step

        still = []
        for s in self.batch:
            if s.phase == "decode":
                if s.req.request_id not in self.ttft:
                    self.ttft[s.req.request_id] = self.now - s.req.arrival_time
                s.decode_left -= 1
                self.tpot_num += step_ms
                self.tpot_den += 1
                convo = self.convos[s.cid]
                convo.grow_cache(1)
                self.tree.resize(s.leaf, convo.kv_cache_size)
                if self.tree.is_resident(s.leaf):
                    self.policy.touch_node(s.leaf, self.now)
            if s.done():
                s.req.finish(self.now - s.req.arrival_time)
                self.finished.append(s.req)
                n = s.leaf
                while n is not None:
                    n.refs = max(0, n.refs - 1)
                    n = n.parent
            else:
                still.append(s)
        self.batch = still

    def run(self):
        guard = 0
        while self.incoming or self.waiting or self.batch:
            self.admit_arrivals()
            if not self.waiting and not self.batch:
                if not self.incoming:
                    break
                self.now = self.incoming[0].arrival_time
                continue
            self.stale_samples.append(self.stale_pct())
            self.step()
            guard += 1
            if guard > 5_000_000:
                raise RuntimeError("step loop did not terminate")
        return self.finished


# ==========================================================================
# Harness
# ==========================================================================

SEEDS = 5


def trial(prefix_tokens, policy_name, seed, pool_mb, max_loras=5,
          groups=3, hw="ours"):
    convos, reqs = make_prefix_sharing_workload(
        n_conversations=40, n_adapters=12, turns_per_convo=5, mean_tokens=40,
        rate=0.0006, turn_gap=6000, shared_prefix_tokens=prefix_tokens,
        shared_prefix_groups=groups, seed=seed)
    sim = RadixSim(reqs, convos, pool_mb, POLICIES[policy_name](),
                   max_loras, prefix_tokens, hw=hw)
    sim.run()
    lats = [r.latency() for r in sim.finished]
    ttfts = list(sim.ttft.values())
    stale = (sum(sim.stale_samples) / len(sim.stale_samples)
             if sim.stale_samples else 0.0)
    tpot = sim.tpot_num / sim.tpot_den if sim.tpot_den else float("nan")
    reuse_frac = (100.0 * sim.reused_tokens
                  / max(1, sim.reused_tokens + sim.recompute_tokens))
    return {
        "p50": percentile(lats, 50),
        "p95": percentile(lats, 95),
        "ttft_p50": percentile(ttfts, 50),
        "tpot": tpot,
        "stale": stale,
        "adapter_loads": sim.adapter_loads,
        "recompute_tokens": sim.recompute_tokens,
        "reuse_frac": reuse_frac,
    }


def avg(**kw):
    rows = [trial(seed=s, **kw) for s in range(SEEDS)]
    return {k: sum(r[k] for r in rows) / SEEDS for k in rows[0]}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix-tokens", type=int, nargs="+",
                    default=[0, 200, 400, 800, 1600])
    ap.add_argument("--pool-mb", type=float, default=1800.0,
                    help="unified pool MB. Default 1800 = squeezed (phase 5's "
                         "'interesting case'), where ordering beat dep-aware.")
    ap.add_argument("--max-loras", type=int, default=5)
    ap.add_argument("--groups", type=int, default=3)
    ap.add_argument("--hw", default="ours",
                    choices=["ours", "elora", "elora-aggressive", "elora-conservative"],
                    help="hardware profile (see core.py). Phase 6 has no CPU/PCIe "
                         "KV path, so --hw moves decode/prefill/swap-cold/adapter.")
    ap.add_argument("--decode-ms-per-token", type=float, default=None)
    ap.add_argument("--prefill-ms-per-token", type=float, default=None)
    ap.add_argument("--swap-cold-ms", type=float, default=None)
    args = ap.parse_args()

    hw = get_profile(args.hw)
    _ov = {}
    if args.decode_ms_per_token is not None: _ov["decode_ms_per_token"] = args.decode_ms_per_token
    if args.prefill_ms_per_token is not None: _ov["prefill_ms_per_token"] = args.prefill_ms_per_token
    if args.swap_cold_ms is not None: _ov["swap_cold_ms"] = args.swap_cold_ms
    if _ov:
        hw = _replace(hw, **_ov)

    print(__doc__)
    print(f"unified pool {args.pool_mb:.0f}MB | max_loras {args.max_loras} | "
          f"{args.groups} prefix groups | hw={hw.name} "
          f"decode={hw.decode_ms_per_token:.2f} | adapter {hw.adapter_mb:.0f}MB | "
          f"mean of {SEEDS} seeds\n")
    print(f"{'prefix tok':>10} {'policy':>10} {'p50':>8} {'p95':>9} {'TTFT p50':>9}"
          f" {'TPOT':>7} {'stale KV':>9} {'reuse%':>8} {'ldrs':>6}")
    print("-" * 84)
    for pt in args.prefix_tokens:
        base = None
        for pol in ["lru-leaf", "ordering", "dep-aware"]:
            m = avg(prefix_tokens=pt, policy_name=pol, pool_mb=args.pool_mb,
                    max_loras=args.max_loras, groups=args.groups, hw=hw)
            if pol == "ordering":
                base = m["p50"]
            tag = ""
            if base and pol == "dep-aware":
                d = 100 * (base - m["p50"]) / base
                tag = f"  dep vs ord {d:+.1f}%"
            print(f"{pt:>10} {pol:>10} {m['p50']:>8.0f} {m['p95']:>9.0f}"
                  f" {m['ttft_p50']:>9.0f} {m['tpot']:>7.2f} {m['stale']:>8.1f}%"
                  f" {m['reuse_frac']:>7.1f}% {m['adapter_loads']:>6.0f}{tag}")
        print()

    print("Hypothesis: as prefix-tokens grows, dep-aware should open a gap over")
    print("ordering -- evicting the wrong LoRA now strands a SHARED prefix that")
    print("'evict the biggest single node' does not protect. If no gap opens,")
    print("ELORA's dependency tree is over-built for the pure LoRA/KV case.")


if __name__ == "__main__":
    main()
