"""Step 5 - Player ratings.

Rating universe: deliveries in IPL matches and in T20Is between ICC Full Members. These
are the competitions the brief describes and the ones in which players overlap enough
for opponent adjustment to work. Associate and A-team matches still train the
match-state models (steps 2-3) but do not enter player ratings. (Sensitivity: s06.)

Outputs, for every batter and bowler:
* conventional statistics (overall and under pressure)
* PAPM estimates: normal level, pressure level, pressure shift (ridge / empirical Bayes)
* the same model on Win Probability Added, as an impact cross-check
* match-level consistency
* match-bootstrap uncertainty (500 resamples)
"""
import numpy as np
import pandas as pd
from common import PROC, OUT, RAW, save_json
import ratings_core as rc

B = 500
GRID_BASE = [100, 200, 400, 800, 1600, 3200]
GRID_DELTA = [200, 800, 3200, 12800, 51200, 1e8]


def universe(b):
    return b.bat_tier.isin(["IPL", "FM"]) & b.bowl_tier.isin(["IPL", "FM"])


def names():
    mp = pd.read_csv(RAW / "competition_team_player_mapping.csv")
    nm = mp.groupby("player_id").player_name.agg(lambda s: s.value_counts().index[0])
    teams = mp.groupby("player_id").team_name.agg(lambda s: ", ".join(pd.unique(s)))
    return nm, teams


def matches_played():
    pe = pd.read_csv(RAW / "playing_eleven.csv")
    pe = pe[pe.is_played.astype(str).str.lower() == "true"]
    return pe.groupby("player_id").match_id.nunique()


def batting_conventional(s):
    g = s.groupby("striker_id")
    out = pd.DataFrame({
        "balls": g.size(), "runs": g.bat_runs.sum(), "outs": g.striker_out.sum(),
        "fours_sixes": g.is_boundary.sum(), "dots": g.is_dot.sum(), "innings": g.apply(lambda d: d.groupby(["match_id", "innings"]).ngroups),
        "rae": g.bat_rae.sum(), "x_runs": g.x_bat_runs.sum(), "x_outs": g.x_out.sum(), "wpa": g.bat_wpa.sum(),
        "LI_mean": g.LI.mean(), "matches": g.match_id.nunique()})
    out["strike_rate"] = 100 * out.runs / out.balls
    out["average"] = out.runs / out.outs.replace(0, np.nan)
    out["dot_pct"] = 100 * out.dots / out.balls
    out["boundary_pct"] = 100 * out.fours_sixes / out.balls
    out["balls_per_dismissal"] = out.balls / out.outs.replace(0, np.nan)
    out["xSR"] = 100 * out.x_runs / out.balls
    out["runs_above_exp"] = out.runs - out.x_runs
    out["outs_above_exp"] = out.outs - out.x_outs
    return out


def bowling_conventional(s):
    g = s.groupby("bowler_id")
    out = pd.DataFrame({
        "balls": g.size(), "legal": g.legal.sum(), "conceded": g.bowler_runs.sum(), "wkts": g.bowler_wicket.sum(),
        "dots": g.is_dot.sum(), "boundaries": g.is_boundary.sum(), "rae": g.bowl_rae.sum(),
        "x_conceded": g.x_bowl_runs.sum(), "x_wkts": g.x_bowl_wkt.sum(), "wpa": g.bowl_wpa.sum(),
        "LI_mean": g.LI.mean(), "matches": g.match_id.nunique()})
    out["economy"] = 6 * out.conceded / out.legal.replace(0, np.nan)
    out["bowl_sr"] = out.legal / out.wkts.replace(0, np.nan)
    out["bowl_avg"] = out.conceded / out.wkts.replace(0, np.nan)
    out["dot_pct"] = 100 * out.dots / out.balls
    out["boundary_pct"] = 100 * out.boundaries / out.balls
    out["xEcon"] = 6 * out.x_conceded / out.legal.replace(0, np.nan)
    out["runs_saved_vs_exp"] = out.x_conceded - out.conceded
    out["wkts_above_exp"] = out.wkts - out.x_wkts
    return out


def consistency(s, actor, val, min_balls):
    """Share of matches with >= min_balls pressure deliveries in which the player was above
    expectation, shrunk toward the population rate with a Beta prior (strength 6 matches)."""
    p = s[s.pressure]
    m = p.groupby([actor, "match_id"]).agg(v=(val, "sum"), n=(val, "size"))
    m = m[m.n >= min_balls]
    g = m.groupby(level=0)
    t = pd.DataFrame({"pressure_matches": g.size(), "positive": g.v.apply(lambda x: (x > 0).sum()),
                      "match_value_sd": g.v.std(), "match_value_median": g.v.median()})
    base = t.positive.sum() / t.pressure_matches.sum()
    k = 6
    t["consistency"] = (t.positive + k * base) / (t.pressure_matches + k)
    t["raw_positive_share"] = t.positive / t.pressure_matches
    return t, float(base)


def season_split(s, actor, val):
    """Pressure value per ball by season, for repeatability."""
    p = s[s.pressure]
    return p.groupby([actor, "season"])[val].agg(["sum", "size"])


def role_batter(s):
    pos = s.groupby("striker_id").batting_order.median()
    return pd.cut(pos, [0, 2.5, 5.5, 8.5, 20], labels=["Opener", "Top/middle order", "Finisher", "Lower order"]).astype(str)


def role_bowler(s):
    """Role from where a bowler's overs fall, relative to the average bowler: a phase is a
    'speciality' if the bowler bowls at least 1.25x the population share of balls in it."""
    ph = s.groupby(["bowler_id", "phase"]).size().unstack(fill_value=0)
    ph = ph.div(ph.sum(axis=1), axis=0)
    pop = s.phase.value_counts(normalize=True)
    idx = ph / pop
    pace = s.groupby("bowler_id").pace_or_spin.agg(lambda x: x.value_counts().index[0]).reindex(ph.index).str.lower()
    pp, mid, dth = idx.get("Powerplay", 0) >= 1.25, idx.get("Middle", 0) >= 1.25, idx.get("Death", 0) >= 1.25
    role = np.select([pp & dth, dth, pp, mid], ["Powerplay & death", "Death", "Powerplay", "Middle overs"], "All-phase")
    return pd.Series(role, index=ph.index) + " " + pace, ph


GRID_RUN_BASE, GRID_RUN_DELTA = [100, 200, 400, 800, 1600], [1600, 6400, 25600, 102400, 409600, 1e8]
GRID_WKT_BASE, GRID_WKT_DELTA = [300, 600, 1250, 2500], [150, 300, 600, 1250, 2500, 10000]


def run_side(s, side):
    if side == "bat":
        actor, opp, y_rae, y_wpa = "striker_id", "bowler_id", "bat_rae", "bat_wpa"
    else:
        actor, opp, y_rae, y_wpa = "bowler_id", "striker_id", "bowl_rae", "bowl_wpa"
    yr, yk = rc.components(s, side)
    cvr, lrb, lrd = rc.cv_select(s, yr, actor, opp, "pressure", GRID_RUN_BASE, GRID_RUN_DELTA)
    cvk, lkb, lkd = rc.cv_select(s, yk, actor, opp, "pressure", GRID_WKT_BASE, GRID_WKT_DELTA)
    cvr.to_csv(OUT / f"cv_lambda_{side}_runs.csv", index=False)
    cvk.to_csv(OUT / f"cv_lambda_{side}_wickets.csv", index=False)
    des, cr, ck = rc.two_component(s, side, (lrb, lrd), (lkb, lkd))
    cp, cn = rc.wicket_price(s, actor, des["actors"])
    normal, pressure = rc.combine(cr, ck, cp, cn)
    # single-component alternative (one ridge on combined RAE) - kept for robustness checks
    cv1, l1b, l1d = rc.cv_select(s, s[y_rae].values, actor, opp, "pressure", GRID_BASE, GRID_DELTA)
    cv1.to_csv(OUT / f"cv_lambda_{side}_single.csv", index=False)
    c1 = rc.fit(des, s[y_rae].values, l1b, l1d)
    # WPA version (percentage points of win probability)
    yw = 100 * s[y_wpa].values
    cvw, lbw, ldw = rc.cv_select(s, yw, actor, opp, "pressure", [400, 1600, 6400, 25600], [1600, 6400, 25600, 102400])
    cvw.to_csv(OUT / f"cv_lambda_{side}_wpa.csv", index=False)
    cw = rc.fit(des, yw, lbw, ldw)
    actors, NOR, PRS, RUNP, WKP, SHIFT = rc.bootstrap_two_component(s, side, (lrb, lrd), (lkb, lkd), B=B)
    _, Aw, PRSw = rc.bootstrap(s, yw, actor, opp, "pressure", lbw, ldw, B=200, seed=7)
    np.savez_compressed(PROC / f"boot_{side}.npz", actors=actors, NOR=NOR, PRS=PRS, RUNP=RUNP, WKP=WKP, SHIFT=SHIFT, Aw=Aw, PRSw=PRSw)
    r = pd.DataFrame({
        "player_id": des["actors"],
        "normal_rae100": 100 * normal, "pressure_rae100": 100 * pressure,
        # behavioural shift: change in runs and dismissals under pressure, valuing wickets at the
        # player's pressure-situation price for both, so a cheaper death-overs wicket is not
        # mistaken for a change in how he plays
        "shift_rae100": 100 * (cr["d"] + cp * ck["d"]),
        "pressure_rae100_sd": 100 * PRS.std(0),
        "pressure_rae100_p10": 100 * np.percentile(PRS, 10, axis=0),
        "pressure_rae100_p20": 100 * np.percentile(PRS, 20, axis=0),
        "pressure_rae100_p80": 100 * np.percentile(PRS, 80, axis=0),
        "pressure_rae100_p90": 100 * np.percentile(PRS, 90, axis=0),
        "normal_rae100_sd": 100 * NOR.std(0),
        "shift_rae100_sd": 100 * SHIFT.std(0),
        "p_shift_positive": (SHIFT > 0).mean(0),
        "price_effect_rae100": 100 * (pressure - normal) - 100 * (cr["d"] + cp * ck["d"]),
        "pressure_runs100": 100 * (cr["a"] + cr["d"]), "normal_runs100": 100 * cr["a"],
        "pressure_wkt100": 100 * (ck["a"] + ck["d"]), "normal_wkt100": 100 * ck["a"],
        "wicket_price_pressure": cp, "wicket_price_normal": cn,
        "pressure_rae100_single": 100 * (c1["a"] + c1["d"]), "normal_rae100_single": 100 * c1["a"],
        "normal_wpa100": 100 * cw["a"], "pressure_wpa100": 100 * (cw["a"] + cw["d"]),
        "pressure_wpa100_sd": 100 * PRSw.std(0),
    })
    hyper = {"runs": {"lam_base": lrb, "lam_delta": lrd, "cv_gain_pct": float(cvr.improvement_vs_null_pct.max())},
             "wickets": {"lam_base": lkb, "lam_delta": lkd, "cv_gain_pct": float(cvk.improvement_vs_null_pct.max())},
             "single": {"lam_base": l1b, "lam_delta": l1d, "cv_gain_pct": float(cv1.improvement_vs_null_pct.max())},
             "wpa": {"lam_base": lbw, "lam_delta": ldw, "cv_gain_pct": float(cvw.improvement_vs_null_pct.max())},
             "residual_sd_runs": float(np.std(yr)), "residual_sd_wkts": float(np.std(yk)),
             "implied_ability_sd_runs_per100": float(100 * np.std(yr) / np.sqrt(lrb)),
             "implied_shift_sd_runs_per100": float(100 * np.std(yr) / np.sqrt(lrd)),
             "implied_ability_sd_wkts_per100": float(100 * np.std(yk) / np.sqrt(lkb)),
             "implied_shift_sd_wkts_per100": float(100 * np.std(yk) / np.sqrt(lkd))}
    return r, hyper


def main():
    b = pd.read_parquet(PROC / "balls_final.parquet")
    u = b[universe(b)].reset_index(drop=True)
    nm, teams = names()
    mp = matches_played()

    # ---------------- batting ----------------
    sb = u[u.faced].reset_index(drop=True)
    conv_all = batting_conventional(sb).add_prefix("all_")
    conv_p = batting_conventional(sb[sb.pressure]).add_prefix("p_")
    conv_n = batting_conventional(sb[~sb.pressure]).add_prefix("n_")
    cons, base_rate_bat = consistency(sb, "striker_id", "bat_rae", 3)
    rb, hyper_b = run_side(sb, "bat")
    bat = rb.set_index("player_id").join([conv_all, conv_p, conv_n, cons.add_prefix("c_")], how="left")
    bat["role"] = role_batter(sb)
    bat["survival_share_pressure"] = sb[sb.pressure].groupby("striker_id").pressure_mode.apply(lambda x: (x == "Survival").mean())
    bat["ipl_share_pressure"] = sb[sb.pressure].groupby("striker_id").comp_group.apply(lambda x: (x == "IPL").mean())
    bat["pressure_seasons"] = sb[sb.pressure].groupby("striker_id").season.nunique()
    bat["knockout_pressure_balls"] = sb[sb.pressure & (sb.knock_out == 1)].groupby("striker_id").size()
    bat["player_name"] = bat.index.map(nm)
    bat["name_in_deliveries"] = sb.groupby("striker_id").striker_name.agg(lambda s: s.value_counts().index[0])
    bat["teams"] = bat.index.map(teams)
    bat["matches_played_all_comps"] = bat.index.map(mp)
    bat.reset_index().to_parquet(PROC / "ratings_bat.parquet", index=False)

    # ---------------- bowling ----------------
    sw = u.reset_index(drop=True)
    conv_all = bowling_conventional(sw).add_prefix("all_")
    conv_p = bowling_conventional(sw[sw.pressure]).add_prefix("p_")
    conv_n = bowling_conventional(sw[~sw.pressure]).add_prefix("n_")
    cons_w, base_rate_bowl = consistency(sw, "bowler_id", "bowl_rae", 6)
    rw, hyper_w = run_side(sw, "bowl")
    bowl = rw.set_index("player_id").join([conv_all, conv_p, conv_n, cons_w.add_prefix("c_")], how="left")
    role, ph = role_bowler(sw)
    bowl["role"] = role
    bowl = bowl.join(ph.add_prefix("phase_share_"))
    bowl["ipl_share_pressure"] = sw[sw.pressure].groupby("bowler_id").comp_group.apply(lambda x: (x == "IPL").mean())
    bowl["pressure_seasons"] = sw[sw.pressure].groupby("bowler_id").season.nunique()
    bowl["knockout_pressure_balls"] = sw[sw.pressure & (sw.knock_out == 1)].groupby("bowler_id").size()
    bowl["player_name"] = bowl.index.map(nm)
    bowl["name_in_deliveries"] = sw.groupby("bowler_id").bowler_name.agg(lambda s: s.value_counts().index[0])
    bowl["teams"] = bowl.index.map(teams)
    bowl["matches_played_all_comps"] = bowl.index.map(mp)
    bowl.reset_index().to_parquet(PROC / "ratings_bowl.parquet", index=False)

    # season-level pressure value for repeatability analysis
    season_split(sb, "striker_id", "bat_rae").to_csv(PROC / "season_pressure_bat.csv")
    season_split(sw, "bowler_id", "bowl_rae").to_csv(PROC / "season_pressure_bowl.csv")
    save_json({"bat": hyper_b, "bowl": hyper_w, "consistency_base_rate_bat": base_rate_bat,
               "consistency_base_rate_bowl": base_rate_bowl, "universe_balls": int(len(u)),
               "universe_matches": int(u.match_id.nunique())}, "rating_hyperparameters.json")
    print(hyper_b, hyper_w)


if __name__ == "__main__":
    main()
