"""Step 11 - Deliverable tables: Top_10_Batters.csv, Top_10_Bowlers.csv, Complete_Player_Ratings.csv,
Final_Unstop_Submission_List.txt."""
import numpy as np
import pandas as pd
from common import PROC, ROOT, load_json

DEL = ROOT / "deliverables"
DEL.mkdir(exist_ok=True)


def top_bat(f):
    t = f.head(10)
    return pd.DataFrame({
        "Rank": t["rank"], "Player": t.player_name, "Player_ID": t.player_id, "Role": t.role, "Teams_in_data": t.teams,
        "Matches_played_rating_universe": t.matches_played_rating_universe, "Matches_played_all_competitions": t.matches_played_all_comps,
        "Pressure_balls_faced": t.p_balls.astype(int), "Matches_with_pressure_exposure": t.c_pressure_matches.astype(int),
        "Mean_DPI_on_pressure_balls": t.p_LI_mean.round(2),
        "Pressure_strike_rate": t.p_strike_rate.round(1), "Expected_SR_same_situations": t.p_xSR.round(1),
        "Pressure_average": t.p_average.round(1), "Pressure_dot_pct": t.p_dot_pct.round(1), "Pressure_boundary_pct": t.p_boundary_pct.round(1),
        "Pressure_dismissals": t.p_outs.astype(int), "Expected_dismissals": t.p_x_outs.round(1),
        "Normal_strike_rate": t.n_strike_rate.round(1), "Normal_average": t.n_average.round(1),
        "Raw_pressure_RAE_per100": t.raw_pressure_rae100.round(2),
        "Normal_level_RAE_per100": t.normal_rae100.round(2), "Pressure_level_RAE_per100": t.pressure_rae100.round(2),
        "Pressure_shift_RAE_per100": t.shift_rae100.round(2), "P_shift_positive": t.p_shift_positive.round(2),
        "Pressure_WPA_pctpts_per100": t.pressure_wpa100.round(2),
        "Consistency_share_pressure_innings_above_exp": t.c_raw_positive_share.round(2), "Consistency_shrunk": t.c_consistency.round(2),
        "Pressure_level_SD": t.pressure_rae100_sd.round(2), "Pressure_level_P10": t.pressure_rae100_p10.round(2),
        "Pressure_level_P90": t.pressure_rae100_p90.round(2), "P_top10_bootstrap": t.p_top10.round(2),
        "Top10_share_core_specifications": t.core_top10_share.round(2),
        "Clutch_Rating_final": t.clutch_rating.round(2), "Explanation": t.explanation})


def top_bowl(f):
    t = f.head(10)
    return pd.DataFrame({
        "Rank": t["rank"], "Player": t.player_name, "Player_ID": t.player_id, "Role": t.role, "Teams_in_data": t.teams,
        "Matches_played_rating_universe": t.matches_played_rating_universe, "Matches_played_all_competitions": t.matches_played_all_comps,
        "Pressure_balls_bowled": t.p_balls.astype(int), "Matches_with_pressure_exposure": t.c_pressure_matches.astype(int),
        "Mean_DPI_on_pressure_balls": t.p_LI_mean.round(2),
        "Pressure_economy": t.p_economy.round(2), "Expected_economy_same_situations": t.p_xEcon.round(2),
        "Pressure_bowling_SR": t.p_bowl_sr.round(1), "Pressure_bowling_average": t.p_bowl_avg.round(1),
        "Pressure_dot_pct": t.p_dot_pct.round(1), "Pressure_boundary_conceded_pct": t.p_boundary_pct.round(1),
        "Pressure_wickets": t.p_wkts.astype(int), "Expected_wickets": t.p_x_wkts.round(1),
        "Normal_economy": t.n_economy.round(2), "Normal_bowling_SR": t.n_bowl_sr.round(1),
        "Raw_pressure_RAE_per100": t.raw_pressure_rae100.round(2),
        "Normal_level_RAE_per100": t.normal_rae100.round(2), "Pressure_level_RAE_per100": t.pressure_rae100.round(2),
        "Pressure_shift_RAE_per100": t.shift_rae100.round(2), "P_shift_positive": t.p_shift_positive.round(2),
        "Pressure_WPA_pctpts_per100": t.pressure_wpa100.round(2),
        "Consistency_share_pressure_spells_above_exp": t.c_raw_positive_share.round(2), "Consistency_shrunk": t.c_consistency.round(2),
        "Pressure_level_SD": t.pressure_rae100_sd.round(2), "Pressure_level_P10": t.pressure_rae100_p10.round(2),
        "Pressure_level_P90": t.pressure_rae100_p90.round(2), "P_top10_bootstrap": t.p_top10.round(2),
        "Top10_share_core_specifications": t.core_top10_share.round(2),
        "Clutch_Rating_final": t.clutch_rating.round(2), "Explanation": t.explanation})


def complete():
    rows = []
    for side, lab in (("bat", "Batting"), ("bowl", "Bowling")):
        full = pd.read_parquet(PROC / f"full_{side}.parquet")
        fin = pd.read_parquet(PROC / f"final_{side}.parquet")[["player_id", "rank", "p_top10", "core_top10_share", "top10_share_all_specs"]]
        d = full.merge(fin, on="player_id", how="left")
        conv = ("p_strike_rate", "n_strike_rate") if side == "bat" else ("p_economy", "n_economy")
        out = pd.DataFrame({
            "Discipline": lab, "Player_ID": d.player_id, "Player": d.player_name, "Role": d.role, "Teams_in_data": d.teams,
            "Balls_all": d.all_balls, "Pressure_balls": d.p_balls.fillna(0).astype(int),
            "Matches_with_pressure_exposure": d.c_pressure_matches.fillna(0).astype(int), "Pressure_seasons": d.pressure_seasons.fillna(0).astype(int),
            "Pressure_SR_or_Economy": d[conv[0]].round(2), "Normal_SR_or_Economy": d[conv[1]].round(2),
            "Raw_pressure_RAE_per100": d.raw_pressure_rae100.round(2),
            "Normal_level_RAE_per100": d.normal_rae100.round(2), "Pressure_level_RAE_per100": d.pressure_rae100.round(2),
            "Pressure_shift_RAE_per100": d.shift_rae100.round(2), "P_shift_positive": d.p_shift_positive.round(2),
            "Pressure_WPA_pctpts_per100": d.pressure_wpa100.round(2), "Consistency_shrunk": d.c_consistency.round(2),
            "Pressure_level_SD": d.pressure_rae100_sd.round(2), "Clutch_Rating": d.clutch_rating.round(2),
            "Eligible_sample": d.eligible, "Evidence_gate_pass": d.evidence_gate, "Final_rank": d["rank"],
            "P_top10_bootstrap": d.p_top10.round(2), "Top10_share_core_specs": d.core_top10_share.round(2)})
        rows.append(out.sort_values(["Final_rank", "Clutch_Rating"], ascending=[True, False], na_position="last"))
    return pd.concat(rows)


def main():
    fb = pd.read_parquet(PROC / "final_bat.parquet")
    fw = pd.read_parquet(PROC / "final_bowl.parquet")
    tb, tw = top_bat(fb), top_bowl(fw)
    for t in (tb, tw):
        assert t.Player_ID.is_unique and t.Player.is_unique and list(t.Rank) == list(range(1, 11))
    tb.to_csv(DEL / "Top_10_Batters.csv", index=False)
    tw.to_csv(DEL / "Top_10_Bowlers.csv", index=False)
    complete().to_csv(DEL / "Complete_Player_Ratings.csv", index=False)
    mp = pd.read_csv(ROOT / "data" / "competition_team_player_mapping.csv")
    names = set(mp.player_name)
    lines = ["SupeRR Selector Season 2 - Round 1", "Final Unstop submission list", "",
             "Names are exactly as they appear in competition_team_player_mapping.csv (player ID in brackets for checking).", "",
             "TOP 10 BATTERS UNDER PRESSURE"]
    for _, r in tb.iterrows():
        assert r.Player in names
        lines.append(f"Rank {r.Rank:>2}: {r.Player}   [player_id {r.Player_ID}]")
    lines += ["", "TOP 10 BOWLERS UNDER PRESSURE"]
    for _, r in tw.iterrows():
        assert r.Player in names
        lines.append(f"Rank {r.Rank:>2}: {r.Player}   [player_id {r.Player_ID}]")
    lines += ["", "If a dropdown shows a shorter form of a name (e.g. 'Tilak Varma' for 'Namboori Tilak Varma'),",
              "choose the entry whose player matches the ID above. No player appears twice in either list."]
    (DEL / "Final_Unstop_Submission_List.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
