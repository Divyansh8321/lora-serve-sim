"""Phase 7 GPU run: measured (real vLLM) vs predicted (phase7_elora.py).

Reads validation/results/p7_ml{ML}_off{GB}_b{B}_s{S}.jsonl from run_p7.sh and
puts each cell next to the simulator's prediction for the same design
(vllm / vllm-offload), slot count, burst and seeds, at --hw a100 (rough
A100 timings) and --hw ours.

  python validation/compare_p7.py                       # after run_p7.sh
  python validation/compare_p7.py --predict-only        # before the run

The question the run answers: does CPU KV offload cut TTFT on a real GPU by
about what the simulator says, and is the slot count as important as it says?
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
import phase7_elora as P  # noqa: E402


def predict(design, ml, burst, seeds, hw):
    rows = [P.trial(design, s, gpu_mb=6000.0, max_loras=ml, hw=hw, burst=burst)
            for s in seeds]
    return {k: statistics.fmean(r[k] for r in rows) for k in ("ttft", "tpot", "mean")}


def measured(results_dir):
    cells = {}
    pat = re.compile(r"p7_ml(\d+)_off([\d.]+)_b([\d.]+)_s(\d+)\.jsonl$")
    for path in glob.glob(os.path.join(results_dir, "p7_*.jsonl")):
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        ml, off, b, s = int(m[1]), float(m[2]), float(m[3]), int(m[4])
        recs = [json.loads(l) for l in open(path) if l.strip()]
        cells.setdefault((ml, off > 0, b), []).append(recs)
    out = {}
    for key, runs in cells.items():
        ok = [r for recs in runs for r in recs if r["error"] is None and r["ttft_ms"] is not None]
        n_all = sum(len(recs) for recs in runs)
        if not ok:
            out[key] = None
            continue
        out[key] = {
            "ttft": statistics.fmean(r["ttft_ms"] for r in ok),
            "tpot": statistics.fmean(r["tpot_ms"] for r in ok if r["tpot_ms"] is not None),
            "mean": statistics.fmean(r["e2e_ms"] for r in ok),
            "late": statistics.fmean(r["late_ms"] for r in ok),
            "errors": n_all - len(ok), "runs": len(runs),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=os.path.join(_REPO_ROOT, "validation", "results"))
    ap.add_argument("--mls", type=int, nargs="+", default=[6, 10, 14])
    ap.add_argument("--bursts", type=float, nargs="+", default=[1.0, 5.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--predict-only", action="store_true")
    args = ap.parse_args()

    meas = {} if args.predict_only else measured(args.results_dir)
    for b in args.bursts:
        print(f"\n== burst x{b:g}: mean TTFT ms (TPOT ms) ==")
        print(f"{'config':>18} {'sim a100':>16} {'sim ours':>16}"
              + ("" if args.predict_only else f" {'MEASURED':>16} {'errors':>7} {'late ms':>8}"))
        rows = {}
        for off in (False, True):
            design = "vllm-offload" if off else "vllm"
            for ml in args.mls:
                a = predict(design, ml, b, args.seeds, "a100")
                o = predict(design, ml, b, args.seeds, "ours")
                line = (f"{design + ' ml=' + str(ml):>18} {a['ttft']:>8.0f} ({a['tpot']:>4.1f})"
                        f" {o['ttft']:>8.0f} ({o['tpot']:>4.1f})")
                m = meas.get((ml, off, b))
                rows[(off, ml)] = (a["ttft"], m["ttft"] if m else None)
                if not args.predict_only:
                    line += (f" {m['ttft']:>8.0f} ({m['tpot']:>4.1f}) {m['errors']:>7} {m['late']:>8.0f}"
                             if m else f" {'(no data)':>16}")
                print(line)
        # headline: best offload vs best plain, and offload's gain at each slot count
        for col, label in ((0, "sim a100"), (1, "measured")):
            plain = [v[col] for (off, _), v in rows.items() if not off and v[col] is not None]
            offl = [v[col] for (off, _), v in rows.items() if off and v[col] is not None]
            if plain and offl:
                print(f"   {label}: best vLLM+offload vs best vLLM TTFT "
                      f"{100 * (min(offl) / min(plain) - 1):+.0f}%; "
                      f"best vLLM vs ml={args.mls[0]} {100 * (min(plain) / rows[(False, args.mls[0])][col] - 1):+.0f}%"
                      if rows[(False, args.mls[0])][col] else "")
    print("\nRead it as: same SIGN and roughly the same SIZE of the offload gain and of the"
          "\nslot-count effect = the simulator's conclusion holds on real hardware. Absolute"
          "\nms will differ (the a100 profile is a rough guess). Non-zero 'errors' or large"
          "\n'late ms' (client could not keep the schedule) make a cell unreliable.")


if __name__ == "__main__":
    main()
