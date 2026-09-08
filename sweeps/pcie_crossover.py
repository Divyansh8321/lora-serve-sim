"""Standalone analysis: does any single HARDWARE constant flip phase 7's
swapper from net-negative to net-positive?

Imports phase7_cost_swapper as a library and sweeps swap-cold-ms and
decode-ms-per-token across a geometric range at burst x10, from the `ours`
engine baseline. Emits a markdown table + a one-line verdict per constant.

Result (as of writing): NEITHER constant flips the sign -- see
phase7_cost_swapper.py's docstring. The flip needs the POLICY switch
(--swap-out-when full). This script exists to make that negative result
reproducible without re-deriving it by hand.

Run:  python sweeps/pcie_crossover.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import replace
from core import OURS
from phase7_cost_swapper import avg

BURST = 10.0
POOL_MB = 2400.0
PREFIX = 400
MAX_LORAS = 5
DRIFT = 60000.0

SWEEPS = [
    ("swap_cold_ms", [300, 200, 130, 85, 55, 35, 20, 10]),
    ("decode_ms_per_token", [12, 9.5, 7.5, 6, 4.5, 3.5, 2.5, 2]),
]


def delta(base_p50, p50):
    return 100.0 * (base_p50 - p50) / base_p50 if base_p50 else float("nan")


def one(field, values):
    print(f"### `{field}` (from `ours`, burst x{BURST:.0f}, evict-on-92% kept)\n")
    print(f"| {field} | swap-full p50 | react-lru p50 | delta% | sign |")
    print("|---|---|---|---|---|")
    prev = None
    crossover = None
    for v in values:
        hw = replace(OURS, **{field: v})
        sf = avg(burst_factor=BURST, policy_name="swap-full", pool_mb=POOL_MB,
                 prefix_tokens=PREFIX, max_loras=MAX_LORAS, drift_period=DRIFT, hw=hw)
        rl = avg(burst_factor=BURST, policy_name="react-lru", pool_mb=POOL_MB,
                 prefix_tokens=PREFIX, max_loras=MAX_LORAS, drift_period=DRIFT, hw=hw)
        d = delta(rl["p50"], sf["p50"])
        sign = "+" if d >= 0 else "-"
        print(f"| {v:g} | {sf['p50']:.0f} | {rl['p50']:.0f} | {d:+.1f}% | {sign} |")
        if prev is not None and (prev[1] < 0) != (d < 0):
            (v0, d0), (v1, d1) = prev, (v, d)
            crossover = v0 + (v1 - v0) * (0 - d0) / (d1 - d0)
        prev = (v, d)
    print()
    if crossover is not None:
        print(f"**Crossover:** `{field}` ~= {crossover:.4f} (interp).\n")
    else:
        state = "positive" if prev[1] >= 0 else "negative"
        print(f"**Verdict:** no sign flip -- `swap-full` stays {state} across "
              f"the whole range. A hardware constant alone does not rescue the "
              f"proactive-eviction swapper.\n")


def main():
    print("# Phase 7 -- hardware-constant crossover sweep\n")
    print("Question: can faster hardware alone flip the timer-driven swapper "
          "from net-negative to net-positive, with ELORA's *policy* choices "
          "(evict-on-full, async swap) left at our defaults?\n")
    for field, values in SWEEPS:
        one(field, values)
    print("---\n")
    print("See `phase7_cost_swapper.py --sweep` for the switch attribution: "
          "`--swap-out-when full` is the single change that flips the sign "
          "(-21.6% -> +2.2% at burst x10).")


if __name__ == "__main__":
    main()
