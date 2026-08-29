"""Tabulate measured (real vLLM) vs predicted (simulator) for Sweep A.

The simulator side reuses phase 2's own trial() so the predicted curve is
exactly what `python phase2_separate_pools.py` prints -- no reimplementation.
Adapter slab MB = max_loras * ADAPTER_MB (20), matching how phase 2 sizes it.

The measured side reads the run_*.jsonl files that validation/workload.py
wrote, one per (max_loras, seed).

Usage:
  # after running the sweep for max_loras in 12 7 5 3, seeds 0..4:
  python validation/compare.py \
      --results-dir validation/results \
      --max-loras 12 7 5 3 --seeds 0 1 2 3 4

  # sim-only preview (no measured data yet):
  python validation/compare.py --predict-only --max-loras 12 7 5 3
"""

import argparse
import glob
import json
import os
import statistics
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from core import ADAPTER_MB  # noqa: E402
from phase2_separate_pools import trial as phase2_trial  # noqa: E402
from pools import NoComm  # noqa: E402

KV_POOL_MB = 800   # phase 2 Part A default


def predict(max_loras, seeds):
    """Mean sim p50/p95/stale/disk over seeds, at this slab size. Mirrors
    phase2_separate_pools.avg() with a_cpu=k_cpu=0 and NoComm."""
    a_gpu = max_loras * ADAPTER_MB
    rows = [phase2_trial(a_gpu, 0, KV_POOL_MB, 0, NoComm, s) for s in seeds]
    agg = {k: statistics.fmean(r[k] for r in rows) for k in rows[0]}
    return agg


def load_measured(results_dir, max_loras, seeds):
    """Return {seed: {p50, p95, n, errors}} from run_maxloras{ML}_seed{S}.jsonl."""
    out = {}
    for s in seeds:
        # accept a couple of naming variants
        cands = [
            os.path.join(results_dir, f"run_maxloras{max_loras}_seed{s}.jsonl"),
            os.path.join(results_dir, f"run_seed{s}.jsonl"),
        ]
        cands += glob.glob(os.path.join(results_dir, f"*maxloras{max_loras}*seed{s}*.jsonl"))
        path = next((c for c in cands if os.path.exists(c)), None)
        if path is None:
            continue
        recs = [json.loads(l) for l in open(path) if l.strip()]
        ok = [r for r in recs if r.get("error") is None and r.get("e2e_ms") is not None]
        if not ok:
            out[s] = {"p50": float("nan"), "p95": float("nan"),
                      "n": len(recs), "errors": len(recs)}
            continue
        lats = sorted(r["e2e_ms"] for r in ok)
        out[s] = {
            "p50": _pct(lats, 50),
            "p95": _pct(lats, 95),
            "ttft_p50": _pct(sorted(r["ttft_ms"] for r in ok if r.get("ttft_ms")), 50),
            "n": len(ok),
            "errors": len(recs) - len(ok),
        }
    return out


def _pct(s, p):
    if not s:
        return float("nan")
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] * (1 - (k - lo)) + s[hi] * (k - lo)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=os.path.join(_REPO_ROOT, "validation", "results"))
    ap.add_argument("--max-loras", type=int, nargs="+", default=[12, 7, 5, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--predict-only", action="store_true")
    args = ap.parse_args(argv)

    print(f"KV pool held fixed at {KV_POOL_MB}MB; adapter slab = max_loras x {ADAPTER_MB}MB\n")
    hdr = (f"{'max_loras':>9} {'slab_MB':>8} "
           f"{'sim_p50':>9} {'sim_p95':>9} {'sim_stale':>10} {'sim_disk':>9}")
    if not args.predict_only:
        hdr += f"  |  {'meas_p50':>9} {'meas_p95':>9} {'meas_ttft50':>12} {'p50_err%':>9} {'runs':>5}"
    print(hdr)
    print("-" * len(hdr))

    first_sim = first_meas = None
    for ml in args.max_loras:
        sim = predict(ml, args.seeds)
        line = (f"{ml:>9} {ml * ADAPTER_MB:>8} "
                f"{sim['p50']:>9.0f} {sim['p95']:>9.0f} {sim['stale']:>9.1f}% {sim['disk']:>9.1f}")
        if not args.predict_only:
            meas = load_measured(args.results_dir, ml, args.seeds)
            if meas:
                p50s = [m["p50"] for m in meas.values() if m["p50"] == m["p50"]]
                p95s = [m["p95"] for m in meas.values() if m["p95"] == m["p95"]]
                tts = [m.get("ttft_p50", float("nan")) for m in meas.values()]
                tts = [t for t in tts if t == t]
                mp50 = statistics.fmean(p50s) if p50s else float("nan")
                mp95 = statistics.fmean(p95s) if p95s else float("nan")
                mtt = statistics.fmean(tts) if tts else float("nan")
                err = (100 * (mp50 - sim["p50"]) / sim["p50"]) if p50s else float("nan")
                line += (f"  |  {mp50:>9.0f} {mp95:>9.0f} {mtt:>12.0f} "
                         f"{err:>8.1f}% {len(p50s):>5}")
            else:
                line += f"  |  {'(no data)':>9}"
        print(line)

    print("\nSuccess criteria (VALIDATION_PLAN.md section 8):")
    print("  strong : |p50_err%| <= 20 across the sweep AND monotonic degradation")
    print("  useful : shape matches (monotonic, knee at same slab size), recalibrate constants")
    print("  neg.   : pathology absent -> reclaim pool / prefix caching absorbed it; report it")


if __name__ == "__main__":
    main()
