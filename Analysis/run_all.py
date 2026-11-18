"""Run the full pipeline from raw CSVs to final rankings.

    python run_all.py            # every step
    python run_all.py s05        # start from step s05 (earlier outputs must exist)
"""
import importlib
import sys
import time

STEPS = ["s01_prepare", "s02_win_probability", "s03_expected", "s04_pressure", "s05_ratings",
         "s06_predictive_validation", "s07_robustness", "s08_final", "s09_insights", "s10_figures", "s11_exports"]

if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else STEPS[0]
    begin = [i for i, s in enumerate(STEPS) if s.startswith(start)][0]
    for step in STEPS[begin:]:
        t = time.time()
        print(f"\n===== {step} =====", flush=True)
        importlib.import_module(step).main()
        print(f"----- {step} done in {time.time() - t:.0f}s", flush=True)
