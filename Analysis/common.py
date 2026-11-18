"""Shared paths, constants and small helpers for the SupeRR Selector pipeline."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data"
PROC = ROOT / "processed"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
for p in (PROC, OUT, FIG):
    p.mkdir(exist_ok=True)

SEED = 20260516
rng = np.random.default_rng(SEED)

# ICC Full Members (2022 onwards). Everything else in the international pool is an Associate.
FULL_MEMBERS = {
    "India", "Australia", "England", "Pakistan", "South Africa", "New Zealand",
    "Sri Lanka", "West Indies", "Bangladesh", "Afghanistan", "Ireland", "Zimbabwe",
}

# Extras that are not recorded run-by-run in ball_by_ball.csv. Each value is the
# expected number of runs per event, estimated by robust regression of result
# margins on extras counts (see 01_prepare.calibrate_extras). They are re-estimated
# on every run; these are only fall-backs.
DEFAULT_EXTRA_RUNS = {"wide": 1.14, "byes": 2.03, "legbyes": 1.25}

# Dismissals credited to the bowler.
BOWLER_WICKETS = {"Caught", "Bowled", "LBW", "Stump", "Hit Wicket"}


def save_json(obj, name):
    with open(OUT / name, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def load_json(name):
    with open(OUT / name) as f:
        return json.load(f)


def phase_of(over):
    """T20 phase from 1-indexed over number."""
    return np.where(over <= 6, "Powerplay", np.where(over <= 15, "Middle", "Death"))
