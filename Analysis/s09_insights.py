"""Step 9 - Discoveries, player profiles and case-study material.

Situation-specific leaderboards use a simple empirical-Bayes shrinkage of raw RAE per
ball toward zero: est = sum(RAE) / (n + k), with k chosen as the prior-strength of the
main model's run component (lambda). This keeps small samples from topping lists.
"""
import numpy as np
import pandas as pd
from common import PROC, OUT, load_json, save_json
from s05_ratings import universe

H = load_json("rating_hyperparameters.json")


def shrink(df, actor, val, k, min_n):
    g = df.groupby(actor)[val].agg(["sum", "size"])
    g = g[g["size"] >= min_n]
    g["est100"] = 100 * g["sum"] / (g["size"] + k)
    g["raw100"] = 100 * g["sum"] / g["size"]
    return g.sort_values("est100", ascending=False)


def main():
    b = pd.read_parquet(PROC / "balls_final.parquet")
    u = b[universe(b)].copy()
    fb = u[u.faced]
    fb_bat = pd.read_parquet(PROC / "final_bat.parquet")
    fw = pd.read_parquet(PROC / "final_bowl.parquet")
    nm = pd.concat([pd.read_parquet(PROC / "full_bat.parquet")[["player_id", "player_name"]],
                    pd.read_parquet(PROC / "full_bowl.parquet")[["player_id", "player_name"]]]).drop_duplicates("player_id").set_index("player_id").player_name
    kb, kw = H["bat"]["runs"]["lam_base"], H["bowl"]["runs"]["lam_base"]
    ins = {}

    def top(df, n=8, extra=None):
        d = df.head(n).copy()
        d["name"] = d.index.map(nm)
        cols = ["name", "est100", "raw100", "size"] + (extra or [])
        return d[cols].round(2).reset_index().to_dict("records")

    # 1. crisis: batting after early collapse (>=3 down inside 8 overs, or >=5 down inside 14)
    crisis = fb[((fb.wkts_before >= 3) & (fb.over <= 8)) | ((fb.wkts_before >= 5) & (fb.over <= 14))]
    ins["crisis_batting"] = top(shrink(crisis, "striker_id", "bat_rae", kb, 60))
    ins["crisis_share_of_balls"] = float(len(crisis) / len(fb))
    # 2. escalating chase: required rate >= 11 with DPI pressure
    esc = fb[(fb.innings == 2) & (fb.rrr >= 11) & fb.pressure]
    ins["escalating_chase"] = top(shrink(esc, "striker_id", "bat_rae", kb, 40))
    # 3. finishers: scoring above expectation without extra dismissal risk (death overs, pressure)
    fin = fb[fb.pressure & (fb.over >= 16)].copy()
    fin["run_res"] = fin.bat_runs - fin.x_bat_runs
    fin["out_res"] = fin.striker_out.astype(float) - fin.x_out
    g = fin.groupby("striker_id").agg(n=("run_res", "size"), runs100=("run_res", "sum"), outs100=("out_res", "sum"))
    g = g[g.n >= 80]
    g["runs100"] = 100 * g.runs100 / (g.n + kb)
    g["outs100"] = 100 * g.outs100 / (g.n + 600)
    g["name"] = g.index.map(nm)
    ins["finishers_scoring_vs_risk"] = g.sort_values("runs100", ascending=False).round(2).reset_index().head(15).to_dict("records")
    g.to_csv(PROC / "finisher_quadrant.csv")
    # 4. bowlers preventing boundaries in high-leverage overs
    pw = u[u.pressure & u.faced]
    gb = pw.groupby("bowler_id").agg(n=("is_boundary", "size"), bnd=("is_boundary", "sum"), xbnd=("x_boundary", "sum"))
    gb = gb[gb.n >= 150]
    gb["bnd_pct"] = 100 * gb.bnd / gb.n
    gb["x_bnd_pct"] = 100 * gb.xbnd / gb.n
    gb["prevented_per100"] = 100 * (gb.xbnd - gb.bnd) / (gb.n + 300)
    gb["name"] = gb.index.map(nm)
    ins["boundary_prevention"] = gb.sort_values("prevented_per100", ascending=False).round(2).reset_index().head(10).to_dict("records")
    # 5. wickets when the batting side is on top (WP >= 60%) under pressure
    hw = u[u.pressure & (u.wp >= 0.6)].copy()
    hw["wres"] = hw.bowler_wicket.astype(float) - hw.x_bowl_wkt
    g5 = hw.groupby("bowler_id").agg(n=("wres", "size"), w=("bowler_wicket", "sum"), xw=("x_bowl_wkt", "sum"), wpa=("bowl_wpa", "sum"))
    g5 = g5[g5.n >= 90]
    g5["wae_per100"] = 100 * (g5.w - g5.xw) / (g5.n + 600)
    g5["name"] = g5.index.map(nm)
    ins["momentum_breakers"] = g5.sort_values("wae_per100", ascending=False).round(2).reset_index().head(10).to_dict("records")
    # 6. underrated / overrated by conventional pressure stats
    for side, df, conv, asc in (("bat", fb_bat, "p_strike_rate", False), ("bowl", fw, "p_economy", True)):
        d = df.copy()
        d["conv_rank"] = d[conv].rank(ascending=asc)
        d["delta_rank"] = d.conv_rank - d["rank"]
        ins[f"underrated_{side}"] = d.sort_values("delta_rank", ascending=False)[["player_name", "rank", "conv_rank", conv, "pressure_rae100"]].head(6).round(2).to_dict("records")
        ins[f"overrated_{side}"] = d.sort_values("delta_rank")[["player_name", "rank", "conv_rank", conv, "pressure_rae100"]].head(6).round(2).to_dict("records")
    # 7. risers and fallers (shrunk shift with resample support)
    for side, df in (("bat", fb_bat), ("bowl", fw)):
        d = df[["player_name", "rank", "normal_rae100", "pressure_rae100", "shift_rae100", "shift_rae100_sd", "p_shift_positive", "p_balls"]]
        ins[f"risers_{side}"] = d.sort_values("shift_rae100", ascending=False).head(8).round(2).to_dict("records")
        ins[f"fallers_{side}"] = d[d.normal_rae100 > d.normal_rae100.quantile(0.6)].sort_values("shift_rae100").head(8).round(2).to_dict("records")
    # 8. exposure to pressure by role / batting position / phase
    ex = fb.groupby(pd.cut(fb.batting_order, [0, 2, 4, 6, 8, 11], labels=["1-2", "3-4", "5-6", "7-8", "9-11"]), observed=True).agg(
        pressure_share=("pressure", "mean"), mean_LI=("LI", "mean"), balls=("LI", "size"))
    ins["exposure_by_position"] = ex.round(3).reset_index().to_dict("records")
    ins["exposure_by_phase_innings"] = u.groupby(["innings", "phase"]).agg(pressure_share=("pressure", "mean"), mean_LI=("LI", "mean")).round(3).reset_index().to_dict("records")
    # 9. IPL vs T20I
    for side, actor, val in (("bat", "striker_id", "bat_rae"), ("bowl", "bowler_id", "bowl_rae")):
        s = fb if side == "bat" else u
        g = s.groupby(["comp_group", "pressure"])[val].mean().unstack()
        ins[f"comp_pressure_gap_{side}"] = (100 * g).round(2).reset_index().to_dict("records")
        pp = s[s.pressure].groupby([actor, "comp_group"])[val].agg(["sum", "size"]).unstack()
        pp = pp[(pp[("size", "IPL")] >= 60) & (pp[("size", "T20I")] >= 60)]
        x = pp[("sum", "IPL")] / (pp[("size", "IPL")] + kb)
        y = pp[("sum", "T20I")] / (pp[("size", "T20I")] + kb)
        ins[f"ipl_vs_t20i_player_corr_{side}"] = {"r": float(np.corrcoef(x, y)[0, 1]), "n": int(len(pp))}
    ins["share_pressure_balls_ipl"] = float(u[u.pressure].comp_group.eq("IPL").mean())
    ins["pressure_rate_by_comp"] = u.groupby("comp_group").pressure.mean().round(3).to_dict()
    # 10. knockouts
    ko = u[u.knock_out == 1]
    ins["knockout"] = {"matches": int(ko.match_id.nunique()), "pressure_share": float(ko.pressure.mean()),
                       "pressure_share_league": float(u[u.knock_out == 0].pressure.mean()), "mean_LI": float(ko.LI.mean()),
                       "mean_LI_league": float(u[u.knock_out == 0].LI.mean())}
    kb_ = shrink(ko[ko.faced & ko.pressure], "striker_id", "bat_rae", kb, 25)
    ins["knockout_batters"] = top(kb_, 6)
    kw_ = shrink(ko[ko.pressure], "bowler_id", "bowl_rae", kw, 36)
    ins["knockout_bowlers"] = top(kw_, 6)
    # 11. pressure mode
    ins["mode_share"] = u[u.pressure].pressure_mode.value_counts(normalize=True).round(3).to_dict()
    sv = fb[fb.pressure & (fb.pressure_mode == "Survival")]
    ins["survival_specialists"] = top(shrink(sv, "striker_id", "bat_rae", kb, 50))
    # 12. how many 'pressure' balls does the top 10 face compared with the pool?
    ins["top10_bat_share_pressure_in_innings2"] = float(fb[fb.pressure & fb.striker_id.isin(fb_bat.head(10).player_id)].innings.eq(2).mean())
    # 13. population: how much does performance change under pressure on average?
    ins["population_pressure_effect"] = {
        "bat_sr_pressure": float(100 * fb[fb.pressure].bat_runs.mean()), "bat_sr_other": float(100 * fb[~fb.pressure].bat_runs.mean()),
        "bat_out_pct_pressure": float(100 * fb[fb.pressure].striker_out.mean()), "bat_out_pct_other": float(100 * fb[~fb.pressure].striker_out.mean()),
        "bat_rae_pressure": float(100 * fb[fb.pressure].bat_rae.mean()), "bat_rae_other": float(100 * fb[~fb.pressure].bat_rae.mean()),
        "bowl_econ_pressure": float(6 * u[u.pressure].bowler_runs.sum() / u[u.pressure].legal.sum()),
        "bowl_econ_other": float(6 * u[~u.pressure].bowler_runs.sum() / u[~u.pressure].legal.sum())}
    save_json(ins, "insights.json")

    # ---------- case-study candidates: biggest single-innings pressure contributions by top-10 players ----------
    rows = []
    for side, df, actor, val in (("bat", fb_bat, "striker_id", "bat_wpa"), ("bowl", fw, "bowler_id", "bowl_wpa")):
        ids = df.head(10).player_id
        s = (fb if side == "bat" else u)
        s = s[s[actor].isin(ids)]
        g = s.groupby([actor, "match_id", "innings"]).agg(wpa=(val, "sum"), pballs=("pressure", "sum"), balls=(val, "size"),
                                                          maxLI=("LI", "max"))
        g = g[g.pballs >= 6].sort_values("wpa", ascending=False).head(8).reset_index()
        g["side"] = side
        g["name"] = g[actor].map(nm)
        rows.append(g)
    cs = pd.concat(rows)
    m = pd.read_parquet(PROC / "matches.parquet").set_index("match_id")
    cs["fixture"] = cs.match_id.map(lambda x: f"{m.loc[x, 'home_team']} v {m.loc[x, 'away_team']}, {m.loc[x, 'comp_name']} ({m.loc[x, 'match_date']})")
    cs["result"] = cs.match_id.map(m.result)
    cs.to_csv(PROC / "case_study_candidates.csv", index=False)
    print(cs[["side", "name", "match_id", "innings", "wpa", "pballs", "fixture", "result"]].to_string())


if __name__ == "__main__":
    main()
