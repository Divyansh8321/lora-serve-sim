"""Standalone analysis: which single ENGINE switch flips phase 7's swapper?

Imports phase7_cost_swapper and, from the `ours` baseline, flips ONE switch
at a time to its ELORA value, running swap-full vs react-lru at each burst
level. Emits a markdown table.

Result: `swapout` ("evict only when full") is the switch that carries the
sign flip; `swapmode` (async) compounds it; the hardware constants
(pcie/decode) do not. See phase7_cost_swapper.py's docstring.

Run:  python sweeps/switch_attribution.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import ELORA_AGGRESSIVE, get_profile
from phase7_cost_swapper import avg, _apply_switch

POOL_MB = 2400.0
PREFIX = 400
MAX_LORAS = 5
DRIFT = 60000.0
BURSTS = [1.0, 5.0, 10.0]
SWITCHES = ["none", "pcie", "decode", "swapmode", "prefetch", "swapout", "all"]


def delta(base_p50, p50):
    return 100.0 * (base_p50 - p50) / base_p50 if base_p50 else float("nan")


def main():
    print("# Phase 7 -- engine-switch attribution\n")
    print("From the `ours` baseline, each row flips ONE switch to its ELORA "
          "value. `swap-full` p50 delta vs `react-lru`, mean of 5 seeds.\n")
    base_hw = get_profile("ours")
    for bf in BURSTS:
        tag = "STEADY" if bf <= 1.0 else f"BURST x{bf:.0f}"
        print(f"### {tag}\n")
        print("| switch | swap-full p50 | react-lru p50 | delta% | sign |")
        print("|---|---|---|---|---|")
        for sw in SWITCHES:
            hw = _apply_switch(base_hw, sw, ELORA_AGGRESSIVE)
            sf = avg(burst_factor=bf, policy_name="swap-full", pool_mb=POOL_MB,
                     prefix_tokens=PREFIX, max_loras=MAX_LORAS, drift_period=DRIFT, hw=hw)
            rl = avg(burst_factor=bf, policy_name="react-lru", pool_mb=POOL_MB,
                     prefix_tokens=PREFIX, max_loras=MAX_LORAS, drift_period=DRIFT, hw=hw)
            d = delta(rl["p50"], sf["p50"])
            sign = "+" if d >= 0 else "-"
            mark = "  **<- flip**" if (sw not in ("none",) and d >= 0) else ""
            print(f"| {sw} | {sf['p50']:.0f} | {rl['p50']:.0f} | {d:+.1f}% | {sign} |{mark}")
        print()
    print("---\n")
    print("`swapout` (evict only when full) is the single change that flips "
          "the sign. `decode` alone makes it *worse* -- a fixed proactive-"
          "eviction overhead is a larger fraction of a smaller step.")


if __name__ == "__main__":
    main()
