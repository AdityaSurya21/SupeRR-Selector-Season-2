"""Step 8 - Final rankings.

Final rating ("Clutch Rating"): the 20th percentile of the match-bootstrap distribution
of a player's pressure-level RAE per 100 pressure balls (two-component PAPM). Read it as:
"we are 80% confident this player adds at least this many runs per 100 pressure balls
compared with an average player in identical situations".

Eligibility (rating universe = IPL + Full-Member T20Is):
* batters: >= 120 pressure balls faced and >= 10 matches with 3+ pressure balls;
* bowlers: >= 180 pressure balls bowled and >= 10 matches with 6+ pressure balls;
* pressure evidence in at least two different seasons (repeatability needs more than one).
"""
import numpy as np
import pandas as pd
from common import PROC, OUT, RAW, save_json
from s05_ratings import universe

ELIG = {"bat": dict(balls=120, matches=10), "bowl": dict(balls=180, matches=10)}

# Tie band: Clutch Ratings closer than this (runs per 100 pressure balls) are treated as
# statistically tied - it is about one-sixth of a typical bootstrap SD (5-7 runs). Inside the
# band, order is settled by how often each player makes the top 10 across the *core*
# robustness specifications (alternatives we consider equally defensible). Specifications we
# argue against in the report - conventional pressure definitions, unshrunk raw rates - and
# subset diagnostics (single competition or innings) do not vote.
TIE_BAND = 1.0
CORE_SPECS = ["Pressure = leverage top 10%", "Pressure = leverage top 30%", "Pressure = phase-relative leverage",
              "Min sample x0.67", "Min sample x1.25", "Min sample x1.67", "All matches incl. Associates/A-teams",
              "Model: single-currency ridge", "Model: WPA ridge", "Model: posterior mean (no risk adjustment)",
              "Model: 4x stronger shrinkage", "Model: 4x weaker shrinkage", "Baseline: no opponent adjustment",
              "Leave out season 2022", "Leave out season 2023", "Leave out season 2024", "Leave out season 2025",
              "Leave out season 2026"]


def core_share(side):
    R = pd.read_csv(OUT / f"robustness_ranks_{side}.csv", index_col=0)
    return (R[CORE_SPECS] <= 10).mean(axis=1)


def tie_break(el):
    """Stable adjacent swaps inside the tie band (bubble passes until nothing moves)."""
    order = list(el.index)
    moved = True
    while moved:
        moved = False
        for i in range(len(order) - 1):
            a, b = el.loc[order[i]], el.loc[order[i + 1]]
            if a.clutch_rating - b.clutch_rating < TIE_BAND and b.core_top10_share > a.core_top10_share:
                order[i], order[i + 1] = order[i + 1], order[i]
                moved = True
    return el.loc[order]


def matches_in_universe():
    b = pd.read_parquet(PROC / "balls_final.parquet")
    um = set(b[universe(b)].match_id)
    pe = pd.read_csv(RAW / "playing_eleven.csv")
    pe = pe[(pe.is_played.astype(str).str.lower() == "true") & pe.match_id.isin(um)]
    return pe.groupby("player_id").match_id.nunique()


def boot_ranks(side, eligible_ids):
    z = np.load(PROC / f"boot_{side}.npz", allow_pickle=True)
    actors = z["actors"]
    keep = np.isin(actors, eligible_ids)
    P = 100 * z["PRS"][:, keep]
    # rank by each bootstrap draw's pressure level (posterior-style rank distribution)
    R = (-P).argsort(1).argsort(1) + 1
    ids = actors[keep]
    return pd.DataFrame({"player_id": ids, "p_top10": (R <= 10).mean(0), "p_top3": (R <= 3).mean(0),
                         "rank_p10": np.percentile(R, 10, axis=0), "rank_median": np.median(R, axis=0),
                         "rank_p90": np.percentile(R, 90, axis=0)})


def _shift_text(r):
    sh, p = r.shift_rae100, r.p_shift_positive
    if p >= 0.8:
        return f"Lifts his level under pressure: {sh:+.1f} runs/100 balls against his own normal ({100*p:.0f}% of resamples agree)."
    if p <= 0.2:
        return (f"Slightly below his own normal level under pressure ({sh:+.1f} runs/100; {100*(1-p):.0f}% of resamples agree), "
                f"but his normal level is high enough to carry it.")
    return f"Plays to his normal level under pressure (shift {sh:+.1f} runs/100, indistinguishable from zero)."


def explain_bat(r):
    return (f"{r.p_balls:.0f} pressure balls in {r.c_pressure_matches:.0f} matches (average DPI {r.p_LI_mean:.1f}). "
            f"Strikes at {r.p_strike_rate:.0f} where an average batter in the same situations would strike at {r.p_xSR:.0f}; "
            f"dismissed {r.p_outs:.0f} times against {r.p_x_outs:.1f} expected. " + _shift_text(r) +
            f" Above expectation in {100*r.c_raw_positive_share:.0f}% of his pressure innings.")


def explain_bowl(r):
    return (f"{r.p_balls:.0f} pressure balls in {r.c_pressure_matches:.0f} matches (average DPI {r.p_LI_mean:.1f}). "
            f"Concedes {r.p_economy:.2f} an over where {r.p_xEcon:.2f} is expected in the same situations; "
            f"{r.p_wkts:.0f} wickets against {r.p_x_wkts:.1f} expected. " + _shift_text(r) +
            f" Beat expectation in {100*r.c_raw_positive_share:.0f}% of his pressure spells.")


def build(side):
    r = pd.read_parquet(PROC / f"ratings_{side}.parquet")
    e = ELIG[side]
    r["eligible"] = (r.p_balls >= e["balls"]) & (r.c_pressure_matches >= e["matches"]) & (r.pressure_seasons >= 2)
    # Evidence gate: the model may rate a player above average under pressure on the strength of
    # his overall ability, but we only *select* players whose own observed pressure record is also
    # above expectation. Model and evidence must agree.
    r["raw_pressure_rae100"] = 100 * r.p_rae / r.p_balls
    r["evidence_gate"] = r.raw_pressure_rae100 > 0
    r["rank_eligible"] = r.eligible & r.evidence_gate
    r["clutch_rating"] = r.pressure_rae100_p20
    el = r[r.rank_eligible].copy()
    el = el.merge(boot_ranks(side, el.player_id.values), on="player_id", how="left")
    el = el.sort_values("clutch_rating", ascending=False).reset_index(drop=True)
    el["rank_by_rating_only"] = np.arange(1, len(el) + 1)
    el["core_top10_share"] = el.player_id.map(core_share(side)).fillna(0)
    el = tie_break(el).reset_index(drop=True)
    el["rank"] = np.arange(1, len(el) + 1)
    el["matches_played_rating_universe"] = el.player_id.map(matches_in_universe())
    el["explanation"] = el.apply(explain_bat if side == "bat" else explain_bowl, axis=1)
    return r, el


def main():
    out = {}
    for side in ("bat", "bowl"):
        full, el = build(side)
        rob = pd.read_csv(OUT / f"robustness_top10_frequency_{side}.csv").rename(columns={"Unnamed: 0": "player_id"})
        el = el.merge(rob[["player_id", "top10_share_all_specs", "specs_eligible", "median_rank"]], on="player_id", how="left")
        el.to_parquet(PROC / f"final_{side}.parquet", index=False)
        full.to_parquet(PROC / f"full_{side}.parquet", index=False)
        top = el.head(10)
        assert top.player_id.is_unique and top.player_name.notna().all()
        out[side] = top[["rank", "player_id", "player_name", "clutch_rating"]].to_dict("records")
        print(el.head(14)[["rank", "rank_by_rating_only", "player_id", "player_name", "role", "clutch_rating", "pressure_rae100", "p_top10", "core_top10_share", "top10_share_all_specs"]].round(2).to_string())
        mism = el[el.player_name != el.name_in_deliveries][["player_id", "player_name", "name_in_deliveries"]]
        print("name mismatches (mapping vs deliveries):", mism.head(20).to_dict("records"))
    save_json(out, "final_top10.json")


if __name__ == "__main__":
    main()
