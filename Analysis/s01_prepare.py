"""Step 1 - Data preparation.

Turns the raw delivery file into a clean table with one row per delivery and the
full match state *before* that delivery: score, wickets, legal balls remaining,
target, runs needed, scoring environment, team strength and the eventual result.

Key data facts established during profiling (all re-checked here):
* `runs` holds runs off the bat only. Wides, byes and leg-byes carry no run count,
  so their run value is imputed from result margins (calibrate_extras).
* `extras1_runs` is a code for the extras type, not a run count.
* Illegal deliveries are wides, 'Overstep' and 'other' (both no-balls). With that
  definition 98% of overs have exactly six legal balls; the rest are the final
  over of an innings.
* Innings 3 and 4 are super overs and are excluded from modelling.
"""
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
from common import RAW, PROC, FULL_MEMBERS, DEFAULT_EXTRA_RUNS, BOWLER_WICKETS, save_json, phase_of


def load_raw():
    b = pd.read_csv(RAW / "ball_by_ball.csv")
    m = pd.read_csv(RAW / "matches.csv")
    c = pd.read_csv(RAW / "competitions.csv")
    return b, m, c


def classify_result(m):
    r = m.result.fillna("").str.lower()
    out = pd.Series("decided", index=m.index)
    out[r.str.contains("no result|abandon")] = "no_result"
    out[r.str.contains("super over|eliminator|tied|tie")] = "tie"
    out[m.win_team_id.isna() & (out == "decided")] = "no_result"
    return out


def calibrate_extras(b, m):
    """Estimate runs per wide / bye / leg-bye from 'won by X runs' margins.

    margin_true - margin_reconstructed = a*dWides + b*dByes + c*dLegbyes + noise,
    where d* are first-innings minus second-innings counts. Huber regression keeps
    the few rain-affected rows from dominating.
    """
    x = b[b.innings <= 2].copy()
    x["wd"] = (x.extras_type == "wide").astype(int)
    x["nb"] = x.extras_type.isin(["Overstep", "other"]).astype(int)
    x["by"] = (x.extras_type == "byes").astype(int)
    x["lb"] = (x.extras_type == "legbyes").astype(int)
    x["base"] = x.runs + x.nb + x.extras_2_runs  # wides enter only through the coefficient
    agg = x.groupby(["match_id", "innings"])[["base", "wd", "by", "lb"]].sum().unstack()
    mm = m.set_index("match_id")
    rows = []
    for mid, r in mm.result.str.lower().items():
        mr = re.search(r"by (\d+) run", str(r))
        clean = pd.isna(mm.loc[mid, "revised_target"]) and pd.isna(mm.loc[mid, "first_inn_revised_overs"]) \
            and "d/l" not in str(r) and "dls" not in str(r)
        if mr and clean and mid in agg.index:
            a = agg.loc[mid]
            rows.append(dict(y=int(mr.group(1)) - (a[("base", 1)] - a[("base", 2)]),
                             wd=a[("wd", 1)] - a[("wd", 2)], by=a[("by", 1)] - a[("by", 2)],
                             lb=a[("lb", 1)] - a[("lb", 2)]))
    d = pd.DataFrame(rows)
    fit = sm.RLM(d.y, d[["wd", "by", "lb"]], M=sm.robust.norms.HuberT()).fit()
    est = {"wide": float(fit.params["wd"]), "byes": float(fit.params["by"]), "legbyes": float(fit.params["lb"])}
    resid = d.y - fit.predict(d[["wd", "by", "lb"]])
    diag = {"n_matches": int(len(d)), "estimates": est, "se": {k: float(v) for k, v in zip(est, fit.bse)},
            "median_abs_margin_error": float(resid.abs().median()),
            "share_within_2_runs": float((resid.abs() <= 2).mean()),
            "share_within_5_runs": float((resid.abs() <= 5).mean())}
    return est, diag


def team_tier(name, is_ipl):
    if is_ipl:
        return "IPL"
    if name.endswith(" A"):
        return "A"
    return "FM" if name in FULL_MEMBERS else "ASSOC"


def compute_elo(m, k=24.0):
    """Chronological Elo. Each match's rating uses only earlier matches, so no leakage.
    Starting points reflect tier: Full Members and IPL 1500, A teams 1400, Associates 1300."""
    start = {"IPL": 1500, "FM": 1500, "A": 1400, "ASSOC": 1300}
    rating, pre = {}, {}
    for _, r in m.sort_values(["match_date", "match_id"]).iterrows():
        h, a = r.home_team_id, r.away_team_id
        for t, tier in ((h, r.home_tier), (a, r.away_tier)):
            rating.setdefault(t, start[tier])
        pre[r.match_id] = (rating[h], rating[a])
        if r.result_class == "no_result":
            continue
        e_h = 1 / (1 + 10 ** ((rating[a] - rating[h]) / 400))
        s_h = 0.5 if r.result_class == "tie" else float(r.win_team_id == h)
        rating[h] += k * (s_h - e_h)
        rating[a] -= k * (s_h - e_h)
    return pre


def main():
    b, m, c = load_raw()
    m = m.merge(c, on="comp_id", how="left")
    m["is_ipl"] = m.comp_name.str.startswith("IPL")
    m["comp_group"] = np.where(m.is_ipl, "IPL", np.where(m.match_format == "T20", "A-team T20", "T20I"))
    m["year"] = pd.to_datetime(m.match_date).dt.year
    m["result_class"] = classify_result(m)
    m["home_tier"] = [team_tier(n, i) for n, i in zip(m.home_team, m.is_ipl)]
    m["away_tier"] = [team_tier(n, i) for n, i in zip(m.away_team, m.is_ipl)]
    m["is_dls"] = m.result.str.lower().str.contains("d/l|dls", na=False) | m.revised_target.notna()
    elo = compute_elo(m)
    m["home_elo"] = m.match_id.map(lambda x: elo[x][0])
    m["away_elo"] = m.match_id.map(lambda x: elo[x][1])

    extra_runs, extras_diag = calibrate_extras(b, m)

    raw_n = len(b)
    b = b[b.innings <= 2].copy()
    super_over_balls = raw_n - len(b)
    b = b.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)

    et = b.extras_type.fillna("none")
    b["is_wide"] = et.eq("wide")
    b["is_noball"] = et.isin(["Overstep", "other"])
    b["is_bye"] = et.eq("byes")
    b["is_legbye"] = et.eq("legbyes")
    b["legal"] = ~(b.is_wide | b.is_noball)
    b["faced"] = ~b.is_wide                      # a no-ball counts as a ball faced
    b["bat_runs"] = b.runs.astype(float)
    b["extra_runs"] = (b.is_wide * extra_runs["wide"] + b.is_noball * 1.0 + b.extras_2_runs
                       + b.is_bye * extra_runs["byes"] + b.is_legbye * extra_runs["legbyes"])
    b["total_runs"] = b.bat_runs + b.extra_runs
    # runs charged to the bowler: bat runs + wides + no-ball penalty (not byes/leg-byes)
    b["bowler_runs"] = b.bat_runs + b.is_wide * extra_runs["wide"] + b.is_noball * (1.0 + b.extras_2_runs)
    b["is_wicket"] = b.is_wicket.astype(bool)
    b["striker_out"] = b.is_wicket & (b.dismissed_player_id == b.striker_id) & ~b.dismissal_type.isin(["Retire Out"])
    b["bowler_wicket"] = b.is_wicket & b.dismissal_type.isin(BOWLER_WICKETS)
    b["is_boundary"] = b.is_four | b.is_six
    b["is_dot"] = b.faced & (b.bat_runs == 0) & (b.total_runs == 0)
    b["team_wicket"] = b.is_wicket & ~b.dismissal_type.isin(["Retire Out"])

    g = b.groupby(["match_id", "innings"], sort=False)
    b["runs_before"] = g.total_runs.cumsum() - b.total_runs
    b["wkts_before"] = g.team_wicket.cumsum() - b.team_wicket
    b["legal_before"] = g.legal.cumsum() - b.legal
    b["over"] = b.overid.astype(int)
    b["phase"] = phase_of(b.over)

    mi = m.set_index("match_id")
    inn_tot = b.groupby(["match_id", "innings"]).total_runs.sum().unstack()
    # balls available to each innings (revised overs are stored in balls)
    b["balls_total"] = 120.0
    r1 = b.match_id.map(mi.first_inn_revised_overs)
    r2 = b.match_id.map(mi.second_inn_revised_overs).fillna(r1)
    b.loc[(b.innings == 1) & r1.notna(), "balls_total"] = r1
    b.loc[(b.innings == 2) & r2.notna(), "balls_total"] = r2
    b["balls_left"] = (b.balls_total - b.legal_before).clip(lower=1)

    # ---- chase targets ----
    targets = {}
    tgt_method = {}
    inn2 = b[b.innings == 2]
    first_bat2 = inn2.groupby("match_id").batting_team_id.first().to_dict()
    last_ball2 = inn2.groupby("match_id").total_runs.last().to_dict()
    for mid in inn_tot.index:
        t1 = inn_tot.loc[mid, 1] if 1 in inn_tot.columns else np.nan
        t2 = inn_tot.loc[mid, 2] if 2 in inn_tot.columns else np.nan
        row = mi.loc[mid]
        if pd.notna(row.revised_target):
            targets[mid], tgt_method[mid] = float(row.revised_target), "official revised target"
            continue
        if pd.isna(t1):
            continue
        tgt = np.round(t1) + 1
        res = str(row.result).lower()
        mr = re.search(r"by (\d+) run", res)
        bat2 = first_bat2.get(mid)
        # The target is made consistent with the *reconstructed* chase, because what drives the
        # match state is runs needed = target - reconstructed score. Unrecorded extras make the
        # reconstructed totals a few runs short; anchoring to the chase keeps the final overs exact.
        if mr and not row.is_dls and pd.notna(t2):
            tgt = np.round(t2) + int(mr.group(1)) + 1          # defended by X: chase ended X+1 short
            method = "anchored to chase + margin"
        elif re.search(r"wicket|wkt", res) and bat2 is not None and row.win_team_id == bat2 and pd.notna(t2):
            lo = t2 - last_ball2.get(mid, 0.0)                  # score before the winning ball
            tgt = float(np.clip(tgt, np.floor(lo) + 1, np.floor(t2))) if np.floor(lo) + 1 <= np.floor(t2) else float(np.floor(t2))
            method = "anchored to winning ball"
        elif row.result_class == "tie" and pd.notna(t2):
            tgt = np.round(t2) + 1
            method = "tie: chase total + 1"
        else:
            method = "reconstructed first innings"
        targets[mid], tgt_method[mid] = float(tgt), method
    b["target"] = np.where(b.innings == 2, b.match_id.map(targets), np.nan)
    b["runs_needed"] = b.target - b.runs_before
    b["rrr"] = np.where(b.innings == 2, b.runs_needed * 6 / b.balls_left, np.nan)
    b["crr"] = np.where(b.legal_before > 0, b.runs_before * 6 / b.legal_before.clip(lower=1), np.nan)
    b["wkts_in_hand"] = 10 - b.wkts_before

    # ---- match context ----
    b["comp_id"] = b.match_id.map(mi.comp_id)
    b["comp_group"] = b.match_id.map(mi.comp_group)
    b["season"] = b.match_id.map(mi.year)
    b["match_date"] = b.match_id.map(mi.match_date)
    b["knock_out"] = b.match_id.map(mi.knock_out).astype(int)
    b["result_class"] = b.match_id.map(mi.result_class)
    b["is_dls"] = b.match_id.map(mi.is_dls)
    home = b.match_id.map(mi.home_team_id)
    b["bat_is_home"] = b.batting_team_id == home
    b["bat_tier"] = np.where(b.bat_is_home, b.match_id.map(mi.home_tier), b.match_id.map(mi.away_tier))
    b["bowl_tier"] = np.where(b.bat_is_home, b.match_id.map(mi.away_tier), b.match_id.map(mi.home_tier))
    b["bat_elo"] = np.where(b.bat_is_home, b.match_id.map(mi.home_elo), b.match_id.map(mi.away_elo))
    b["bowl_elo"] = np.where(b.bat_is_home, b.match_id.map(mi.away_elo), b.match_id.map(mi.home_elo))
    b["elo_diff"] = ((b.bat_elo - b.bowl_elo) / 25).round() * 25   # coarse, for the same reason as env
    win = b.match_id.map(mi.win_team_id)
    b["bat_won"] = np.where(b.result_class == "tie", 0.5,
                            np.where(b.result_class == "no_result", np.nan, (win == b.batting_team_id).astype(float)))

    # Scoring environment known *before* the match: run rate of the previous 40 matches in
    # the same pool (IPL / international). Rounded to 0.1 rpo so that it cannot act as a
    # match identifier inside tree models (an early version that used a leave-one-out mean
    # let the trees memorise matches - see validation notes).
    inn = b.groupby("match_id").agg(runs=("total_runs", "sum"), balls=("legal", "sum"),
                                    grp=("comp_group", "first"), date=("match_date", "first")).reset_index()
    inn["pool"] = inn.grp.replace({"A-team T20": "T20I"})
    inn = inn.sort_values(["date", "match_id"])
    inn["rpo"] = 6 * inn.runs / inn.balls.clip(lower=1)
    inn["env"] = inn.groupby("pool").rpo.transform(lambda s: s.shift(1).rolling(40, min_periods=10).mean())
    inn["env"] = inn.env.fillna(inn.rpo.expanding().mean().shift(1)).fillna(8.4).round(1)
    b["env_rpo"] = b.match_id.map(inn.set_index("match_id").env)

    # post-delivery state (used for win-probability changes)
    b["runs_after"] = b.runs_before + b.total_runs
    b["wkts_after"] = b.wkts_before + b.team_wicket
    b["balls_left_after"] = (b.balls_left - b.legal).clip(lower=0)
    # last ball of each innings: after-state is terminal
    b["inn_last"] = ~b.duplicated(["match_id", "innings"], keep="last")

    keep_cols = [c for c in b.columns if c not in ("delivery_ist_time",)]
    b[keep_cols].to_parquet(PROC / "balls.parquet", index=False)
    m.to_parquet(PROC / "matches.parquet", index=False)

    summary = {
        "raw_deliveries": int(raw_n), "super_over_deliveries_excluded": int(super_over_balls),
        "deliveries_modelled": int(len(b)), "matches_in_matches_csv": int(len(m)),
        "matches_with_deliveries": int(b.match_id.nunique()),
        "matches_without_deliveries": int(len(set(m.match_id) - set(b.match_id))),
        "result_classes": m.result_class.value_counts().to_dict(),
        "dls_or_revised_matches": int(m.is_dls.sum()),
        "comp_groups_matches": m.groupby("comp_group").size().to_dict(),
        "knockout_matches": int(m.knock_out.sum()),
        "extras_calibration": extras_diag,
        "target_methods": pd.Series(tgt_method).value_counts().to_dict(),
        "share_overs_with_six_legal": float((b.groupby(["match_id", "innings", "over"]).legal.sum() == 6).mean()),
        "unique_batters": int(b.striker_id.nunique()), "unique_bowlers": int(b.bowler_id.nunique()),
        "date_range": [str(m.match_date.min()), str(m.match_date.max())],
        "tiers_balls": b.bat_tier.value_counts().to_dict(),
    }
    save_json(summary, "data_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
