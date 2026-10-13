"""PHASE 7 -- ELORA as the paper specifies it, against vLLM and S-LoRA.

Replaces the original phase 7 (archive/phase7_heuristic_swapper.py), which
was our own approximation: KV was deleted on eviction, every adapter miss was
a 300 ms disk load, adapters lived outside the tree, the scoring formula was
our invention, and prefetch read future arrivals. Every item below follows
ELORA (HPCA 2026, sections V-VII); ELORA_ALIGNMENT.md has the paper's text and
the choices the paper leaves open.

MEMORY MODEL (all designs):
  * GPU memory is managed in 16 MB blocks (ELORA: "232 bytes per 16 MB
    block"). KV: 128 tokens per block (0.125 MB/token). A LoRA is one node of
    ADAPTER_MB; a shared system prompt is one node.
  * Host memory (default 3.2x GPU, ELORA's 256 GB : 80 GB). Every adapter is
    loaded from disk once (swap_cold_ms) and stays cached in host RAM after
    that; reloads are PCIe copies (hw.pcie_ms_per_mb per MB).
  * The tree: LoRA -> shared prefix -> each conversation's chain of KV blocks.
    A chain's state is (blocks on GPU, blocks on host, blocks gone): the GPU
    part is always the front of the chain, then the host part, then the gone
    tail, which must be recomputed (re-prefilled) before use.

DESIGNS (--designs):
  vllm         vLLM with prefix caching: fixed adapter slab of max_loras slots
               (also the cap on distinct adapters per batch), KV region = the
               rest. LRU everywhere. Evicted KV is deleted. Adapter loads
               block the step. ELORA's baseline uses a 0.2 LoRA memory ratio,
               which is max_loras = 0.2 * GPU / ADAPTER_MB (default here).
  vllm-offload vLLM plus CPU KV offload (what LMCache or vLLM's offloading
               connector add): evicted KV goes to host memory and is copied
               back asynchronously. Still a fixed slab, LRU, no tree, no
               swapper. At the 0.2 slab ratio this IS ELORA's own vLLM
               baseline (FastLibra sec. 6.1: "We adapt a swap-out policy
               based on LRU for vLLM to offload its history KV caches and
               LoRAs in the main memory"). Tuned, it tests what is left of
               ELORA's win once vLLM's slab size is chosen well.
  slora        S-LoRA: one unified pool, adapters LRU, NO history-KV reuse
               (a conversation's KV is freed when its turn ends, so every
               turn re-prefills the whole history). ELORA's description.
  elora        the full design: unified pool, dependency tree (a LoRA cannot
               leave the GPU while any of its KV is there; KV is swapped out
               from leaves and swapped in from roots), evicted KV goes to host
               memory, transfers are asynchronous (only the request waiting on
               a transfer waits), and the 100 ms swapper below.
  elora-wos    ELORA without the swapper: same tree and host tier, plain LRU,
               no prefetch (the paper's WOS ablation).
  elora-wom    ELORA without the dependency tree: LoRAs and KV are evicted
               independently, so KV can outlive its LoRA (the WOM ablation).
  elora-nohost ELORA with evicted KV deleted instead of swapped to host. Not
               in the paper; shows what the host tier contributes.

THE SWAPPER (ELORA Eq. 3-6; runs every 100 ms in elora / elora-wom):
  prob_i        decayed request share of LoRA i (half-life --visit-halflife)
  BS            mean batch size over the last 5 s
  Low_lora    = sum_i 1 - (1 - prob_i)^BS              expected LoRAs per batch
  Now_i       = LoRAs on GPU after swapping node i out
  LoRA_Eval_i = max(1, Low_lora / Now_i)                 --eq4 fastlibra (default)
              = min(1, Now_i / Low_lora)                 --eq4 hpca
  Retain_i    = cost_i * visit_i * (1 - sigmoid(t_i))
                cost_i  = PCIe transfer ms for the node
                visit_i = the node's decayed share of visits (among its type)
                t_i     = time since last use, in --t-unit (default s)
  Eval_i      = LoRA_Eval_i * Retain_i
  Swap-out: lowest Eval GPU leaf first. --trigger hpca (default): only when
            an allocation does not fit. --trigger fastlibra: also proactively
            above 95% full, down to 70%.
  Swap-in : while GPU < 70% full, host-memory roots (a node whose parent is
            on GPU) by highest Eval, asynchronously. Causal: uses only past
            visits.

QUESTION: with ELORA's real mechanisms in place -- host tier, async
transfers, the dependency tree, Eq. 3-6 -- does ELORA beat vLLM and S-LoRA,
and which mechanism carries the difference?

FINDING (2026-10-11, chat workload: 20 adapters, ~200-token prompts and
replies, 400-token shared system prompt, 15 s think time, GPU 6000 MB, host
19200 MB; mean of 5 seeds). "Oracle vLLM" = the best of max_loras
{6 (ELORA's 0.2 ratio), 10, 14, 20}, as in ELORA's own oracle comparison.

  change in mean TTFT vs oracle vLLM      --hw ours          --hw elora-aggressive
                                          steady  burst x5   steady  burst x5
    elora                                  -62%    -87%       -21%    -36%
    elora-wos (tree + LRU, no swapper)     -61%    -87%       -21%    -36%
    elora-wom (no dependency tree)         -62%    -87%       -21%    -36%
    elora-nohost (KV deleted, not swapped)  -1%     -7%        -7%    -22%
    vllm-offload (vLLM + CPU KV offload)   -61%    -85%       -21%    -36%
    slora                                 +106%    +42%       +23%    +15%
  TPOT, elora vs oracle vLLM: -21% / -36% (ours), -5% / -10% (elora-
  aggressive). ELORA reports -37.8%.

  READING:
  * The direction and rough size of ELORA's headline reproduce once the
    simulator has ELORA's cost structure: -21% to -87% TTFT vs a tuned
    vLLM (paper: -38.7% vs oracle vLLM, -45.7% vs default vLLM).
  * ALL of the win comes from one mechanism: evicted KV goes to host memory
    and comes back over PCIe instead of being recomputed. Take that away
    (elora-nohost) and ELORA is roughly a tuned vLLM (0 to -22%: the
    rest is the unified pool letting KV use the adapter slab's memory). Give vLLM only that
    (vllm-offload, tuned) and it matches ELORA within ~1%, except at the
    tight, bursty point (6000 MB, burst x5, --hw ours: 920 vs 794 ms TTFT,
    where the unified pool's flexibility is worth ~15%).
  * The dependency tree and the Eq. 3-6 swapper change nothing measurable:
    elora, elora-wos and elora-wom are within 1% in every cell, under every
    reading of the paper's ambiguities (--eq4 hpca/fastlibra, --trigger
    hpca/fastlibra, --t-unit 1 ms / 1 s / 60 s, GPU 6000 / 12000 MB). Stale
    KV stays ~0% even without the tree, because a 180 MB LoRA is costly to
    move and is almost never the eviction victim. Prefetch rarely fires:
    with history kept, the GPU is almost always above 70% full.
  * ELORA's ablations say the tree (WOM: 1.51x TTFT) and the swapper
    (WOS: 1.42x) each matter a lot. This model does not reproduce either.
    Either their workloads stress something ours does not, or the ablations
    change more than the paper describes. Measuring it needs their code or
    a GPU run (VALIDATION_PLAN.md).
  * PEAK LOAD (--peak, --hw elora-aggressive, mean TTFT < 500 ms): elora
    5.99 req/s, vllm-offload 5.99, oracle vllm 5.05 (+19% for ELORA), vllm
    at the 0.2 ratio 3.04 (+97%; paper: +78.9% vs vLLM), slora 5.05,
    elora-nohost 5.05. Same story: the host KV tier is the difference. (At
    --hw ours the idle-load TTFT is already ~490 ms, so ELORA's 500 ms
    cutoff only makes sense on the H800-class profile; see --ttft-slo.)
  * vLLM at ELORA's own 0.2 ratio (max_loras 6) is far worse than tuned
    vLLM (TTFT 4998 vs 1559 ms) because max_loras also caps distinct
    adapters per batch (phase 5). Comparing against default vLLM inflates
    the gap.

MORE ADAPTERS (--n-adapters, ELORA uses 20/50/100), TTFT in ms, ours HW:
                         paper's vLLM      tuned vLLM
    adapters  burst   (offload, ml=6)   + offload (ml=14)   elora   wos   wom
       20      x1          2575              608             600    601   600
       20      x5          5199              920 (ml=10)     794    794   794
       50      x1          6669              651             646    646   646
       50      x5         13355             1364            1128   1134  1128
      100      x1         10953              681             671    671   671
      100      x5         19605             1633            1387   1390  1387
  Against the paper's own baseline ELORA wins 77-94%. Against the same
  baseline with a well-chosen slab it wins ~1% steady and 14-17% under
  bursts: the unified pool's real value is not having to pick one slab size
  for traffic that changes. The swapper (wos) and the tree (wom) still add
  ~0, which disagrees with the paper's 1.42x / 1.51x.

REAL TRACES (run_on_real_traces.py): see the "Phase 7" block in
REAL_DATA_FINDINGS.md. As loaded, LMSYS and Taskmaster deliver all of a
conversation's turns at once, so no conversation ever comes back after its
KV was evicted, and every design lands within ~2%. --think-ms adds an
(assumed) gap between turns. On Mooncake (433 adapters, ~7k-token prompts)
elora is +10% TTFT vs oracle vLLM (ml 60): the unified pool evicts and
reloads adapters (1014 loads vs 82); vllm-offload ml=60 is best at -30%.

Run:  python phase7_elora.py                          # chat workload, ours HW
      python phase7_elora.py --hw elora-aggressive
      python phase7_elora.py --burst 1 5 10
      python phase7_elora.py --peak                   # max req/s at mean TTFT < 500 ms
"""

import argparse
import heapq
import math
from collections import deque
from dataclasses import dataclass, replace as _replace

from core import ADAPTER_MB, MB_PER_TOKEN, get_profile, make_bursty_workload, percentile

BLOCK_MB = 16.0
BLOCK_TOKENS = int(round(BLOCK_MB / MB_PER_TOKEN))      # 128
STEP_TOKEN_BUDGET = 512
FIXED_STEP_MS = 4.0
PER_SEQ_STEP_MS = 0.05
MAX_BATCH_SEQS = 192
WAIT_THRESHOLD_MS = 30000
SWAPPER_INTERVAL_MS = 100.0
BS_WINDOW_MS = 5000.0
PREFETCH_BELOW = 0.70
PROACTIVE_ABOVE, PROACTIVE_TO = 0.95, 0.70


def _blocks(tokens):
    return (tokens + BLOCK_TOKENS - 1) // BLOCK_TOKENS if tokens > 0 else 0


def _sigmoid(x):
    if x > 60:
        return 1.0
    if x < -60:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


@dataclass(frozen=True)
class Design:
    name: str
    unified: bool = True          # False = vLLM's fixed adapter slab
    host_kv: bool = True          # evicted KV goes to host memory (else deleted)
    dependency: bool = True       # LoRA is the parent of its KV
    ranker: str = "eval"          # "eval" (Eq. 3-6) | "lru"
    swapper: bool = True          # 100 ms timer: prefetch (+ proactive out)
    async_swap: bool = True       # transfers overlap compute
    history: bool = True          # keep a conversation's KV after its turn
    batch_cap: bool = False       # max_loras caps distinct adapters per batch


DESIGNS = {d.name: d for d in [
    Design("vllm", unified=False, host_kv=False, dependency=False, ranker="lru",
           swapper=False, async_swap=False, batch_cap=True),
    Design("vllm-offload", unified=False, host_kv=True, dependency=False,
           ranker="lru", swapper=False, async_swap=True, batch_cap=True),
    Design("slora", host_kv=False, dependency=False, ranker="lru",
           swapper=False, async_swap=False, history=False),
    Design("elora"),
    Design("elora-wos", ranker="lru", swapper=False),
    Design("elora-wom", dependency=False),
    Design("elora-nohost", host_kv=False),
]}


class Stats:
    """Decayed visit counter + last-use time, shared by every node type."""
    __slots__ = ("visits", "stamp", "last_used")

    def __init__(self):
        self.visits = 0.0
        self.stamp = 0.0
        self.last_used = -1e18

    def visit(self, now, halflife):
        self.visits = self.decayed(now, halflife) + 1.0
        self.stamp = now
        self.last_used = now

    def decayed(self, now, halflife):
        if self.visits == 0.0:
            return 0.0
        return self.visits * 0.5 ** ((now - self.stamp) / halflife)


class Lora:
    kind = "lora"

    def __init__(self, aid, mb):
        self.id = aid
        self.mb = mb
        self.tier = None              # None (disk only) | "gpu" | "host"
        self.ready_at = 0.0
        self.pins = 0
        self.gpu_kids = 0             # GPU-resident KV nodes under it
        self.st = Stats()


class Prefix:
    kind = "prefix"

    def __init__(self, key, lora, tokens):
        self.id = key
        self.lora = lora
        self.tokens = tokens
        self.mb = _blocks(tokens) * BLOCK_MB
        self.tier = None              # None (must recompute) | "gpu" | "host"
        self.ready_at = 0.0
        self.pins = 0
        self.gpu_kids = 0             # chains with GPU blocks under it
        self.host_kids = 0
        self.st = Stats()


class Chain:
    """One conversation's KV: blocks [0, g) on GPU, [g, g+h) on host, rest gone."""
    kind = "chain"

    def __init__(self, cid, lora, prefix):
        self.id = cid
        self.lora = lora
        self.prefix = prefix
        self.tokens = 0
        self.g = 0
        self.h = 0
        self.ready_at = 0.0
        self.pins = 0
        self.st = Stats()

    def n(self):
        return _blocks(self.tokens)


class Seq:
    __slots__ = ("req", "chain", "prefill_left", "decode_left", "ready_at")

    def __init__(self, req, chain, to_prefill, ready_at):
        self.req = req
        self.chain = chain
        self.prefill_left = to_prefill
        self.decode_left = req.output_tokens
        self.ready_at = ready_at


class Sim:
    def __init__(self, requests, conversations, design, gpu_mb, host_mb,
                 max_loras, hw="ours", eq4="fastlibra", trigger="hpca",
                 t_unit_ms=1000.0, visit_halflife_ms=10000.0):
        hw = get_profile(hw)
        self.d = design
        self.hw = hw
        self.adapter_mb = hw.adapter_mb
        self.pcie = hw.pcie_ms_per_mb
        self.gpu_mb = gpu_mb
        self.host_mb = host_mb
        self.max_loras = max_loras
        self.slab_mb = max_loras * hw.adapter_mb if not design.unified else 0.0
        self.kv_cap = gpu_mb - self.slab_mb if not design.unified else gpu_mb
        self.eq4 = eq4
        self.trigger = trigger
        self.t_unit = t_unit_ms
        self.halflife = visit_halflife_ms

        self.loras = {}
        self.prefixes = {}
        self.chains = {}
        # chains with blocks on GPU / on host. Dicts, not sets: insertion
        # order is deterministic, so eviction ties break the same every run.
        self.gpu_chains = {}
        self.host_chains = {}
        self.kv_vis = 0.0            # decayed sum of KV-node visits
        self.kv_vis_stamp = 0.0
        self.last_stale = -1e18
        for c in conversations:
            a = c.adapter_id
            if a not in self.loras:
                self.loras[a] = Lora(a, hw.adapter_mb)
            lora = self.loras[a]
            pt = getattr(c, "prefix_tokens", 0) or 0
            prefix = None
            if pt > 0:
                key = getattr(c, "prefix_key", None) or (a, getattr(c, "prefix_group", 0))
                prefix = self.prefixes.get(key)
                if prefix is None:
                    prefix = self.prefixes[key] = Prefix(key, lora, pt)
            self.chains[c.conversation_id] = Chain(c.conversation_id, lora, prefix)

        self.incoming = deque(sorted(requests, key=lambda r: r.arrival_time))
        self.now = 0.0
        self.waiting = deque()
        self.batch = []
        self.finished = []
        self.dropped = []
        self.gpu_lora_mb = 0.0
        self.gpu_kv_mb = 0.0
        self.host_kv_mb = 0.0
        self.h2d_free = 0.0
        self.d2h_free = 0.0
        self.sync_ms = 0.0
        self.last_tick = -1e18
        self.bs_samples = deque()

        # counters
        self.disk_loads = 0
        self.lora_swapins = 0
        self.kv_swapin_blocks = 0
        self.kv_swapout_blocks = 0
        self.kv_deleted_blocks = 0
        self.prefetches = 0
        self.recompute_tokens = 0
        self.transfer_ms = 0.0
        self.disk_ms = 0.0
        self.ttft = {}
        self.tpot_num = 0.0
        self.tpot_den = 0
        self.stale_num = 0.0
        self.stale_den = 0
        self.overcommit_mb = 0.0

    # ------------------------------------------------------------ accounting
    def used(self):
        return self.gpu_lora_mb + self.gpu_kv_mb

    def fill(self):
        return self.used() / self.gpu_mb

    def _kv_free(self):
        return self.kv_cap - self.gpu_kv_mb if not self.d.unified else self.gpu_mb - self.used()

    def _lora_free(self):
        return self.slab_mb - self.gpu_lora_mb if not self.d.unified else self.gpu_mb - self.used()

    def _set_lora_tier(self, lora, tier):
        if lora.tier == "gpu":
            self.gpu_lora_mb -= lora.mb
        if tier == "gpu":
            self.gpu_lora_mb += lora.mb
        lora.tier = tier

    def _set_prefix_tier(self, p, tier):
        if p.tier == tier:
            return
        if p.tier == "gpu":
            self.gpu_kv_mb -= p.mb
            p.lora.gpu_kids -= 1
        elif p.tier == "host":
            self.host_kv_mb -= p.mb
        if tier == "gpu":
            self.gpu_kv_mb += p.mb
            p.lora.gpu_kids += 1
        elif tier == "host":
            self.host_kv_mb += p.mb
        p.tier = tier

    def _set_chain(self, ch, g, h):
        was_g, was_h = ch.g > 0, ch.h > 0
        self.gpu_kv_mb += (g - ch.g) * BLOCK_MB
        self.host_kv_mb += (h - ch.h) * BLOCK_MB
        ch.g, ch.h = g, h
        if g > 0:
            self.gpu_chains[ch.id] = ch
        else:
            self.gpu_chains.pop(ch.id, None)
        if h > 0:
            self.host_chains[ch.id] = ch
        else:
            self.host_chains.pop(ch.id, None)
        parent = ch.prefix
        if was_g != (g > 0):
            dlt = 1 if g > 0 else -1
            if parent is not None:
                parent.gpu_kids += dlt
            else:
                ch.lora.gpu_kids += dlt
        if parent is not None and was_h != (h > 0):
            parent.host_kids += 1 if h > 0 else -1

    # ------------------------------------------------------------ scoring
    def _low_lora(self):
        tot = sum(l.st.decayed(self.now, self.halflife) for l in self.loras.values())
        if tot <= 0:
            return 1.0
        bs = (sum(b for _, b in self.bs_samples) / len(self.bs_samples)
              if self.bs_samples else 1.0)
        bs = max(bs, 1.0)
        return sum(1.0 - (1.0 - l.st.decayed(self.now, self.halflife) / tot) ** bs
                   for l in self.loras.values())

    def _eval_ctx(self):
        now, hl = self.now, self.halflife
        tot = {"lora": sum(l.st.decayed(now, hl) for l in self.loras.values()),
               "kv": self._kv_vis_total()}
        n_gpu = sum(1 for l in self.loras.values() if l.tier == "gpu")
        return {"tot": tot, "low": self._low_lora(), "now_lora": n_gpu}

    def _kv_vis_total(self):
        # every node decays at the same rate, so the sum decays as one value
        return self.kv_vis * 0.5 ** ((self.now - self.kv_vis_stamp) / self.halflife)

    def _note_kv_visit(self):
        self.kv_vis = self._kv_vis_total() + 1.0
        self.kv_vis_stamp = self.now

    def eval_node(self, node, ctx, out=True):
        """ELORA Eq. 3-6. `out`: scoring for swap-out (Now_i = LoRAs left
        after removing it) vs swap-in (Now_i = LoRAs after adding it)."""
        now_i = ctx["now_lora"]
        if node.kind == "lora":
            now_i += -1 if out else 1
            mb = node.mb
            tot = ctx["tot"]["lora"]
        else:
            mb = BLOCK_MB if node.kind == "chain" else node.mb
            tot = ctx["tot"]["kv"]
        now_i = max(now_i, 1)
        low = ctx["low"]
        if self.eq4 == "hpca":
            lora_eval = min(1.0, now_i / low)
        else:
            lora_eval = max(1.0, low / now_i)
        visit = node.st.decayed(self.now, self.halflife) / tot if tot > 0 else 0.0
        t = (self.now - node.st.last_used) / self.t_unit
        retain = (mb * self.pcie) * visit * (1.0 - _sigmoid(t))
        return lora_eval * retain

    def _key(self, node, ctx, out=True):
        if self.d.ranker == "lru":
            return (node.st.last_used, 0.0)
        return (self.eval_node(node, ctx, out), node.st.last_used)

    # ------------------------------------------------------------ eviction
    def _gpu_leaves(self, protect, want_lora, want_kv):
        out = []
        if want_kv:
            for ch in self.gpu_chains.values():
                if ch.pins == 0 and id(ch) not in protect:
                    out.append(ch)
            for p in self.prefixes.values():
                if (p.tier == "gpu" and p.pins == 0 and p.gpu_kids == 0
                        and id(p) not in protect):
                    out.append(p)
        if want_lora:
            for l in self.loras.values():
                if l.tier != "gpu" or l.pins or id(l) in protect:
                    continue
                if self.d.dependency and l.gpu_kids > 0:
                    continue
                if self.now < l.ready_at:
                    continue
                out.append(l)
        return out

    def _transfer_out(self, mb):
        cost = mb * self.pcie
        self.transfer_ms += cost
        if self.d.async_swap:
            self.d2h_free = max(self.now, self.d2h_free) + cost
        else:
            self.sync_ms += cost

    def _transfer_in(self, mb):
        """Returns when the data is usable."""
        cost = mb * self.pcie
        self.transfer_ms += cost
        if self.d.async_swap:
            self.h2d_free = max(self.now, self.h2d_free) + cost
            return self.h2d_free
        self.sync_ms += cost
        return self.now

    def _make_host_room(self, need, protect):
        if self.host_kv_mb + need <= self.host_mb:
            return True
        cands = [c for c in self.host_chains.values() if id(c) not in protect]
        cands.sort(key=lambda c: c.st.last_used)
        for c in cands:
            if self.host_kv_mb + need <= self.host_mb:
                break
            k = min(c.h, math.ceil((self.host_kv_mb + need - self.host_mb) / BLOCK_MB))
            self._set_chain(c, c.g, c.h - k)
            self.kv_deleted_blocks += k
        if self.host_kv_mb + need > self.host_mb:
            for p in self.prefixes.values():
                if (p.tier == "host" and p.host_kids == 0 and p.gpu_kids == 0
                        and id(p) not in protect):
                    self._set_prefix_tier(p, None)
                    if self.host_kv_mb + need <= self.host_mb:
                        break
        return self.host_kv_mb + need <= self.host_mb

    def _evict(self, node, deficit, protect):
        """Evict (part of) one GPU leaf. Returns MB freed."""
        if node.kind == "lora":
            self._set_lora_tier(node, "host")       # host already has a copy
            return node.mb
        if node.kind == "prefix":
            if self.d.host_kv and self._make_host_room(node.mb, protect):
                self._set_prefix_tier(node, "host")
                self._transfer_out(node.mb)
            else:
                self._set_prefix_tier(node, None)
            return node.mb
        ch = node
        k = min(ch.g, max(1, math.ceil(deficit / BLOCK_MB)))
        if self.d.host_kv and self._make_host_room(k * BLOCK_MB, protect | {id(ch)}):
            self._set_chain(ch, ch.g - k, ch.h + k)
            self.kv_swapout_blocks += k
            self._transfer_out(k * BLOCK_MB)
        else:
            # deleted: the blocks after them on host are dropped too (simple)
            self.kv_deleted_blocks += k + ch.h
            self._set_chain(ch, ch.g - k, 0)
        return k * BLOCK_MB

    def make_room(self, need, protect, for_lora=False, target_free=None):
        """Free GPU memory by evicting leaves. In vLLM's split layout an
        adapter needs room in the slab and KV needs room in the KV region."""
        if self.d.unified:
            free = lambda: self.gpu_mb - self.used()
            want_lora, want_kv = True, True
        elif for_lora:
            free = self._lora_free
            want_lora, want_kv = True, False
        else:
            free = self._kv_free
            want_lora, want_kv = False, True
        goal = need if target_free is None else target_free
        if free() >= goal - 1e-9:
            return True
        ctx = self._eval_ctx() if self.d.ranker == "eval" else None
        heap = [(self._key(n, ctx), i, n)
                for i, n in enumerate(self._gpu_leaves(protect, want_lora, want_kv))]
        heapq.heapify(heap)
        seq = len(heap)
        while free() < goal - 1e-9 and heap:
            _, _, node = heapq.heappop(heap)
            # still a valid leaf?
            if node.kind == "chain" and (node.g == 0 or node.pins):
                continue
            if node.kind == "prefix" and (node.tier != "gpu" or node.gpu_kids or node.pins):
                continue
            if node.kind == "lora" and (node.tier != "gpu" or node.pins
                                        or (self.d.dependency and node.gpu_kids)):
                continue
            self._evict(node, goal - free(), protect)
            newly = []
            if node.kind == "chain" and node.g > 0:
                newly.append(node)                         # more blocks left
            if node.kind == "chain" and node.g == 0 and node.prefix is not None:
                newly.append(node.prefix)
            if node.kind in ("chain", "prefix") and want_lora:
                newly.append(node.lora)
            if node.kind == "lora" and ctx is not None:
                ctx = self._eval_ctx()                     # Now_LoRA changed
                heap = [(self._key(n, ctx), i, n) for i, n in
                        enumerate(self._gpu_leaves(protect, want_lora, want_kv))]
                heapq.heapify(heap)
                seq = len(heap)
                continue
            for n in newly:
                seq += 1
                heapq.heappush(heap, (self._key(n, ctx), seq, n))
        return free() >= goal - 1e-9

    # ------------------------------------------------------------ admission
    def _path_need(self, ch, new_in):
        """GPU MB this request needs beyond what is already there,
        split into (lora_mb, kv_mb)."""
        lora_mb = ch.lora.mb if ch.lora.tier != "gpu" else 0.0
        kv = 0.0
        if ch.prefix is not None and ch.prefix.tier != "gpu":
            kv += ch.prefix.mb
        kv += (_blocks(ch.tokens + new_in) - ch.g) * BLOCK_MB
        return lora_mb, kv

    def try_admit(self, r):
        ch = self.chains[r.conversation_id]
        new_in = getattr(r, "input_tokens", 0) or 0
        protect = {id(ch), id(ch.lora)}
        if ch.prefix is not None:
            protect.add(id(ch.prefix))
        lora_mb, kv_mb = self._path_need(ch, new_in)
        if self.d.unified:
            if not self.make_room(lora_mb + kv_mb, protect):
                return None
        else:
            if lora_mb and not self.make_room(lora_mb, protect, for_lora=True):
                return None
            if kv_mb and not self.make_room(kv_mb, protect):
                return None

        ready = self.now
        to_prefill = 0
        lora = ch.lora
        if lora.tier != "gpu":
            if lora.tier is None:
                self.disk_loads += 1
                cost = self.hw.swap_cold_ms
                self.disk_ms += cost
                if self.d.async_swap:
                    lora.ready_at = self.now + cost
                else:
                    self.sync_ms += cost
                    lora.ready_at = self.now
            else:
                self.lora_swapins += 1
                lora.ready_at = self._transfer_in(lora.mb)
            self._set_lora_tier(lora, "gpu")
        ready = max(ready, lora.ready_at)

        p = ch.prefix
        if p is not None:
            if p.tier == "host":
                p.ready_at = self._transfer_in(p.mb)
                self._set_prefix_tier(p, "gpu")
            elif p.tier is None:
                to_prefill += p.tokens
                self.recompute_tokens += p.tokens
                self._set_prefix_tier(p, "gpu")
            ready = max(ready, p.ready_at)

        if ch.h > 0:
            self.kv_swapin_blocks += ch.h
            ch.ready_at = self._transfer_in(ch.h * BLOCK_MB)
        ready = max(ready, ch.ready_at)
        cached_tokens = min(ch.tokens, (ch.g + ch.h) * BLOCK_MB / MB_PER_TOKEN)
        missing = int(ch.tokens - cached_tokens)
        if missing > 0:
            to_prefill += missing
            self.recompute_tokens += missing
        ch.tokens += new_in
        to_prefill += new_in
        self._set_chain(ch, ch.n(), 0)

        for node in (lora, p, ch):
            if node is not None:
                node.pins += 1
                node.st.visit(self.now, self.halflife)
                if node is not lora:
                    self._note_kv_visit()
        return Seq(r, ch, to_prefill, ready)

    def schedule(self):
        if not self.waiting:
            return
        self.waiting = deque(sorted(self.waiting, key=lambda r: r.arrival_time))
        deferred = deque()
        while self.waiting and len(self.batch) < MAX_BATCH_SEQS:
            r = self.waiting.popleft()
            ch = self.chains[r.conversation_id]
            if any(s.chain is ch for s in self.batch):
                deferred.append(r)                  # previous turn still running
                continue
            if self.d.batch_cap:
                active = {s.chain.lora.id for s in self.batch}
                starving = self.now - r.arrival_time > WAIT_THRESHOLD_MS
                if ch.lora.id not in active and len(active) >= self.max_loras and not starving:
                    deferred.append(r)
                    continue
            s = self.try_admit(r)
            if s is None:
                deferred.append(r)
                continue
            self.batch.append(s)
        self.waiting.extendleft(reversed(deferred))

    # ------------------------------------------------------------ swapper
    def _host_roots(self):
        """Host-memory nodes whose parent is on GPU (ELORA swaps in roots)."""
        out = []
        for l in self.loras.values():
            if l.tier == "host":
                out.append(l)
        for p in self.prefixes.values():
            if p.tier == "host" and (p.lora.tier == "gpu" or not self.d.dependency):
                out.append(p)
        for c in self.host_chains.values():
            if c.h == 0:
                continue
            parent_ok = (c.g > 0 or
                         (c.prefix.tier == "gpu" if c.prefix is not None
                          else c.lora.tier == "gpu"))
            if parent_ok or not self.d.dependency:
                out.append(c)
        return out

    def tick(self):
        if not self.d.swapper or self.now - self.last_tick < SWAPPER_INTERVAL_MS:
            return
        self.last_tick = self.now
        pinned = set()
        if self.trigger == "fastlibra" and self.fill() > PROACTIVE_ABOVE:
            self.make_room(0.0, pinned,
                           target_free=self.gpu_mb * (1.0 - PROACTIVE_TO))
        if self.fill() >= PREFETCH_BELOW:
            return
        ctx = self._eval_ctx()
        roots = self._host_roots()
        roots.sort(key=lambda n: self.eval_node(n, ctx, out=False), reverse=True)
        budget = self.gpu_mb * PREFETCH_BELOW - self.used()
        for n in roots:
            if budget <= 0:
                break
            if n.kind == "lora":
                if n.mb > budget:
                    continue
                n.ready_at = self._transfer_in(n.mb)
                self._set_lora_tier(n, "gpu")
                self.lora_swapins += 1
                budget -= n.mb
            elif n.kind == "prefix":
                if n.mb > budget or (self.d.dependency and n.lora.tier != "gpu"):
                    continue
                n.ready_at = self._transfer_in(n.mb)
                self._set_prefix_tier(n, "gpu")
                budget -= n.mb
            else:
                k = min(n.h, int(budget // BLOCK_MB))
                if k <= 0:
                    continue
                n.ready_at = self._transfer_in(k * BLOCK_MB)
                self.kv_swapin_blocks += k
                self._set_chain(n, n.g + k, n.h - k)
                budget -= k * BLOCK_MB
            self.prefetches += 1

    # ------------------------------------------------------------ stepping
    def _free_seq(self, s):
        ch = s.chain
        for node in (ch.lora, ch.prefix, ch):
            if node is not None:
                node.pins -= 1
                node.st.last_used = self.now
        if not self.d.history:
            # S-LoRA: no history-KV reuse -- free the conversation's KV
            self._set_chain(ch, 0, 0)
            p = ch.prefix
            if p is not None and p.pins == 0 and p.gpu_kids == 0:
                self._set_prefix_tier(p, None)

    def _stale_sample(self):
        if self.gpu_kv_mb <= 0:
            return
        stale = sum(c.g * BLOCK_MB for c in self.gpu_chains.values()
                    if c.lora.tier != "gpu")
        stale += sum(p.mb for p in self.prefixes.values()
                     if p.tier == "gpu" and p.lora.tier != "gpu")
        self.stale_num += stale / self.gpu_kv_mb
        self.stale_den += 1

    def step(self):
        self.tick()
        self.schedule()
        runnable = [s for s in self.batch if s.ready_at <= self.now]
        if not runnable:
            nxt = [s.ready_at for s in self.batch]
            if self.incoming:
                nxt.append(self.incoming[0].arrival_time)
            if nxt:
                self.now = max(self.now, min(nxt))
                return
            if self.waiting:
                self.dropped.append(self.waiting.popleft())
            return
        n_decode = sum(1 for s in runnable if s.prefill_left <= 0)
        room = STEP_TOKEN_BUDGET - n_decode
        prefill = 0
        for s in runnable:
            if s.prefill_left > 0 and room > 0:
                c = min(s.prefill_left, room)
                s.prefill_left -= c
                prefill += c
                room -= c
        step_ms = (FIXED_STEP_MS + PER_SEQ_STEP_MS * len(runnable)
                   + self.hw.decode_ms_per_token
                   + prefill * self.hw.prefill_ms_per_token + self.sync_ms)
        self.sync_ms = 0.0
        self.now += step_ms
        self.bs_samples.append((self.now, len(runnable)))
        while self.bs_samples and self.bs_samples[0][0] < self.now - BS_WINDOW_MS:
            self.bs_samples.popleft()

        done = set()
        for s in runnable:
            if s.prefill_left > 0:
                continue
            rid = s.req.request_id
            if rid not in self.ttft:
                self.ttft[rid] = self.now - s.req.arrival_time
            s.decode_left -= 1
            self.tpot_num += step_ms
            self.tpot_den += 1
            ch = s.chain
            before = ch.n()
            ch.tokens += 1
            if ch.n() > before:
                prot = {id(ch), id(ch.lora)}
                if ch.prefix is not None:
                    prot.add(id(ch.prefix))
                self.make_room(BLOCK_MB, prot)
                self._set_chain(ch, ch.n(), ch.h)
                self.overcommit_mb = max(self.overcommit_mb,
                                         self.used() - self.gpu_mb)
            if s.decode_left <= 0:
                s.req.finish(self.now - s.req.arrival_time)
                self.finished.append(s.req)
                self._free_seq(s)
                done.add(id(s))
        if done:
            self.batch = [s for s in self.batch if id(s) not in done]

    def run(self):
        guard = 0
        while self.incoming or self.waiting or self.batch:
            while self.incoming and self.incoming[0].arrival_time <= self.now:
                self.waiting.append(self.incoming.popleft())
            if not self.waiting and not self.batch:
                self.now = self.incoming[0].arrival_time
                continue
            if self.now - self.last_stale >= SWAPPER_INTERVAL_MS:
                self.last_stale = self.now
                self._stale_sample()
            self.step()
            guard += 1
            if guard > 5_000_000:
                raise RuntimeError("step loop did not terminate")
        return self.finished

    def metrics(self):
        lats = [r.latency() for r in self.finished]
        ttfts = list(self.ttft.values())
        n = len(lats)
        return {
            "n": n,
            "mean": sum(lats) / n if n else float("nan"),
            "p50": percentile(lats, 50),
            "p95": percentile(lats, 95),
            "ttft": sum(ttfts) / len(ttfts) if ttfts else float("nan"),
            "tpot": self.tpot_num / self.tpot_den if self.tpot_den else float("nan"),
            "stale": 100.0 * self.stale_num / self.stale_den if self.stale_den else 0.0,
            "disk": self.disk_loads,
            "lora_in": self.lora_swapins,
            "kv_in": self.kv_swapin_blocks,
            "kv_out": self.kv_swapout_blocks,
            "kv_del": self.kv_deleted_blocks,
            "prefetch": self.prefetches,
            "rebuild_s": self.recompute_tokens * self.hw.prefill_ms_per_token / 1000,
            "xfer_s": self.transfer_ms / 1000,
            "disk_s": self.disk_ms / 1000,
            "dropped": len(self.dropped),
            "overcommit": self.overcommit_mb,
        }


# ==========================================================================
# Harness
# ==========================================================================

SEEDS = 5


def chat_workload(seed, burst=1.0, rate_scale=1.0, prefix_tokens=400, n_adapters=20):
    """Chat-sized turns (as phase 5's `chat`), 20 adapters by default, a shared
    system prompt per (adapter, group), bursty arrivals and drifting
    popularity (ELORA drives its datasets with the bursty Azure trace)."""
    # above scale 1, add conversations too: with 15 s think time, 80
    # conversations cannot produce more than ~3 req/s however fast they start
    return make_bursty_workload(
        n_conversations=int(round(80 * max(1.0, rate_scale))), n_adapters=n_adapters,
        turns_per_convo=5, mean_tokens=200,
        mean_input_tokens=200, base_rate=0.0006 * rate_scale, turn_gap=15000,
        shared_prefix_tokens=prefix_tokens, shared_prefix_groups=3,
        burst_factor=burst, burst_period=40000.0, burst_duty=0.25,
        popularity_drift_period=60000.0, seed=seed)


def default_max_loras(gpu_mb, hw):
    """ELORA's vLLM baseline: LoRA memory ratio 0.2."""
    return max(1, int(0.2 * gpu_mb / get_profile(hw).adapter_mb))


def trial(design, seed, gpu_mb=6000.0, host_mb=None, max_loras=None, hw="ours",
          burst=1.0, rate_scale=1.0, prefix_tokens=400, n_adapters=20, workload=None,
          **kw):
    if workload is None:
        convos, reqs = chat_workload(seed, burst, rate_scale, prefix_tokens, n_adapters)
    else:
        convos, reqs = workload(seed)
    host_mb = 3.2 * gpu_mb if host_mb is None else host_mb
    ml = max_loras or default_max_loras(gpu_mb, hw)
    sim = Sim(reqs, convos, DESIGNS[design], gpu_mb, host_mb, ml, hw=hw, **kw)
    sim.run()
    m = sim.metrics()
    if reqs:
        span = max(r.arrival_time for r in reqs) - min(r.arrival_time for r in reqs)
        m["qps"] = 1000.0 * len(reqs) / span if span > 0 else float("nan")
    return m


def avg(**kw):
    rows = [trial(seed=s, **kw) for s in range(SEEDS)]
    return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}


def peak_qps(design, ttft_slo=500.0,
             scales=(0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.25, 1.5, 1.75, 2, 2.5, 3,
                     3.5, 4, 5, 6), **kw):
    """ELORA's metric: the highest request rate with mean TTFT < 500 ms."""
    best = 0.0
    for sc in scales:
        m = avg(design=design, rate_scale=sc, **kw)
        if m["ttft"] >= ttft_slo:
            break
        best = m["qps"]
    return best


HEADER = (f"{'design':>13} {'mean':>7} {'p50':>7} {'TTFT':>7} {'TPOT':>6} {'stale':>6}"
          f" {'disk':>5} {'LoRA in':>7} {'KV in':>6} {'KV out':>6} {'KV del':>6}"
          f" {'prefch':>6} {'rebuild s':>9} {'xfer s':>7}")


def fmt(name, m):
    return (f"{name:>13} {m['mean']:>7.0f} {m['p50']:>7.0f} {m['ttft']:>7.0f}"
            f" {m['tpot']:>6.1f} {m['stale']:>5.1f}% {m['disk']:>5.0f} {m['lora_in']:>7.0f}"
            f" {m['kv_in']:>6.0f} {m['kv_out']:>6.0f} {m['kv_del']:>6.0f}"
            f" {m['prefetch']:>6.0f} {m['rebuild_s']:>9.1f} {m['xfer_s']:>7.1f}"
            + (f"  DROPPED {m['dropped']:.0f} requests (did not fit)"
               if m.get("dropped") else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--designs", nargs="+", default=list(DESIGNS))
    ap.add_argument("--hw", default="ours",
                    choices=["ours", "a100", "elora", "elora-aggressive", "elora-conservative"])
    ap.add_argument("--gpu-mb", type=float, default=6000.0)
    ap.add_argument("--host-mb", type=float, default=None,
                    help="host memory for KV (default 3.2x GPU, ELORA's ratio)")
    ap.add_argument("--max-loras", type=int, default=None,
                    help="vLLM adapter slots (default: 0.2 of GPU, ELORA's baseline)")
    ap.add_argument("--vllm-max-loras", type=int, nargs="+", default=[10, 14, 20],
                    help="extra vLLM slab sizes to try (the 0.2-ratio value is "
                         "always included); the best is reported as oracle vLLM")
    ap.add_argument("--burst", type=float, nargs="+", default=[1.0, 5.0])
    ap.add_argument("--rate-scale", type=float, default=1.0)
    ap.add_argument("--prefix-tokens", type=int, default=400)
    ap.add_argument("--n-adapters", type=int, default=20,
                    help="ELORA evaluates 20, 50 and 100")
    ap.add_argument("--eq4", choices=["fastlibra", "hpca"], default="fastlibra")
    ap.add_argument("--trigger", choices=["hpca", "fastlibra"], default="hpca")
    ap.add_argument("--t-unit-ms", type=float, default=1000.0,
                    help="unit of t_i in sigmoid(t_i) (paper does not say)")
    ap.add_argument("--visit-halflife-ms", type=float, default=10000.0)
    ap.add_argument("--peak", action="store_true",
                    help="also report peak req/s at mean TTFT < --ttft-slo")
    ap.add_argument("--ttft-slo", type=float, default=500.0,
                    help="ELORA's 500 ms was set on H800s; at --hw ours the "
                         "idle-load TTFT is already ~490 ms, so use "
                         "--hw elora-aggressive or raise this")
    args = ap.parse_args()

    hw = get_profile(args.hw)
    ml = args.max_loras or default_max_loras(args.gpu_mb, hw)
    kw = dict(gpu_mb=args.gpu_mb, host_mb=args.host_mb, max_loras=ml, hw=hw,
              prefix_tokens=args.prefix_tokens, n_adapters=args.n_adapters, eq4=args.eq4, trigger=args.trigger,
              t_unit_ms=args.t_unit_ms, visit_halflife_ms=args.visit_halflife_ms)
    print(__doc__)
    print(f"hw={hw.name} | GPU {args.gpu_mb:.0f} MB | host {3.2 * args.gpu_mb if args.host_mb is None else args.host_mb:.0f} MB"
          f" | vLLM max_loras {ml} | eq4={args.eq4} trigger={args.trigger}"
          f" | mean of {SEEDS} seeds\n")
    print("columns: mean/p50 latency, mean TTFT, mean TPOT (ms); stale = share of\n"
          "GPU KV whose LoRA is not on GPU; disk = first-time adapter loads;\n"
          "LoRA in = adapter copies host->GPU; KV in/out/del = 16 MB blocks swapped\n"
          "in / swapped out to host / deleted; prefch = swapper prefetches;\n"
          "rebuild s = seconds of re-prefill for KV that was gone; xfer s = PCIe time\n")
    vsweep = sorted(set(args.vllm_max_loras + [ml]))
    for b in args.burst:
        print(f"== burst x{b:g} (rate scale {args.rate_scale:g}) ==")
        print(HEADER)
        print("-" * len(HEADER))
        rows = {}
        for d in args.designs:
            if d.startswith("vllm"):
                best = None
                for v in vsweep:
                    m = avg(design=d, burst=b, rate_scale=args.rate_scale,
                            **{**kw, "max_loras": v})
                    tag = " <- ELORA's 0.2 ratio" if v == ml else ""
                    print(fmt(f"{d} ml={v}", m) + tag)
                    if m["dropped"]:
                        continue                   # cannot serve this load
                    if best is None or m["ttft"] < best["ttft"]:
                        best, best_ml = m, v
                rows[d] = best
                print(f"{'':>13} best (\"oracle\") {d}: max_loras={best_ml}")
            else:
                rows[d] = avg(design=d, burst=b, rate_scale=args.rate_scale, **kw)
                print(fmt(d, rows[d]))
        if "vllm" in rows:
            o = rows["vllm"]
            print(f"\n  vs oracle vLLM (TTFT / TPOT / mean latency):")
            for d, m in rows.items():
                if d == "vllm":
                    continue
                print(f"  {d:>13}: {100 * (m['ttft'] - o['ttft']) / o['ttft']:+5.0f}% /"
                      f" {100 * (m['tpot'] - o['tpot']) / o['tpot']:+5.0f}% /"
                      f" {100 * (m['mean'] - o['mean']) / o['mean']:+5.0f}%")
        print()
    if args.peak:
        print(f"== peak load: highest req/s with mean TTFT < {args.ttft_slo:.0f} ms (burst x1) ==")
        for d in args.designs:
            if d.startswith("vllm"):
                qs = {v: peak_qps(d, args.ttft_slo, **{**kw, "max_loras": v})
                      for v in vsweep}
                v = max(qs, key=qs.get)
                print(f"{d:>13} {qs[v]:6.2f} req/s  (best max_loras={v}; "
                      f"at 0.2 ratio: {qs[ml]:.2f})")
            else:
                print(f"{d:>13} {peak_qps(d, args.ttft_slo, **kw):6.2f} req/s")


if __name__ == "__main__":
    main()
