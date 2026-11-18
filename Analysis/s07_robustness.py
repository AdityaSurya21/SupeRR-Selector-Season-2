"""Step 7 - Robustness: do the rankings survive reasonable changes to every choice we made?

Each specification changes one decision and recomputes the ranking from scratch
(posterior-mean pressure level from the two-component model unless stated). We report
* Spearman correlation with the baseline ranking (players eligible in both),
* overlap between the specification's top 10 and the baseline top 10,
* for every player, the share of specifications in which he makes the top 10.
The baseline ranking statistic is the 20th percentile of the match-bootstrap
distribution (a conservative estimate); for the cheaper specifications we use the
posterior mean minus 0.84 analytical-approximate SDs from the baseline bootstrap,
which gives the same ordering logic without 500 refits per specification.
"""
import numpy as np
import pandas as pd
from common import PROC, OUT, load_json, save_json
import ratings_core as rc
from s05_ratings import universe

H = load_json("rating_hyperparameters.json")
META = load_json("pressure_meta.json")
MIN = {"bat": 120, "bowl": 180}
MIN_MATCHES = 10


def lams(side, key="runs"):
    return (H[side][key]["lam_base"], H[side][key]["lam_delta"])


def rate(s, side, method="2c", lam_scale=1.0, no_opp=False):
    actor, opp = ("striker_id", "bowler_id") if side == "bat" else ("bowler_id", "striker_id")
    des = rc.build(s, actor, opp, "pressure")
    if no_opp:
        big = 1e12
    if method == "2c":
        lr = tuple(x * lam_scale for x in lams(side, "runs"))
        lk = tuple(x * lam_scale for x in lams(side, "wickets"))
        yr, yk = rc.components(s, side)
        kw = dict(lam_opp=big, lam_opp_delta=big) if no_opp else {}
        cr = rc.fit(des, yr, *lr, **kw)
        ck = rc.fit(des, yk, *lk, **kw)
        cp, cn = rc.wicket_price(s, actor, des["actors"])
        _, prs = rc.combine(cr, ck, cp, cn)
        val = 100 * prs
    elif method == "single":
        y = s.bat_rae.values if side == "bat" else s.bowl_rae.values
        c = rc.fit(des, y, *lams(side, "single"))
        val = 100 * (c["a"] + c["d"])
    elif method == "wpa":
        y = 100 * (s.bat_wpa.values if side == "bat" else s.bowl_wpa.values)
        c = rc.fit(des, y, *lams(side, "wpa"))
        val = 100 * (c["a"] + c["d"])
    elif method == "raw":
        y = s.bat_rae if side == "bat" else s.bowl_rae
        g = s[s.pressure].assign(y=y[s.pressure]).groupby(actor).y.mean()
        val = 100 * g.reindex(des["actors"]).values
    elif method == "conventional":
        p = s[s.pressure].groupby(actor)
        if side == "bat":
            val = (100 * p.bat_runs.mean()).reindex(des["actors"]).values          # pressure strike rate
        else:
            val = (-6 * p.bowler_runs.sum() / p.legal.sum()).reindex(des["actors"]).values   # pressure economy
    out = pd.DataFrame({"player_id": des["actors"], "value": val})
    pb = s[s.pressure].groupby(actor).size()
    pm = s[s.pressure].groupby([actor, "match_id"]).size()
    pm = (pm >= (3 if side == "bat" else 6)).groupby(level=0).sum()
    out["p_balls"] = out.player_id.map(pb).fillna(0)
    out["p_matches"] = out.player_id.map(pm).fillna(0)
    return out


def ranked(df, sd, min_balls, conservative=True):
    d = df[(df.p_balls >= min_balls) & (df.p_matches >= MIN_MATCHES)].copy()
    if conservative:
        d["score"] = d.value - 0.84 * d.player_id.map(sd).fillna(sd.median())
    else:
        d["score"] = d.value
    d["rank"] = d.score.rank(ascending=False, method="first")
    return d.set_index("player_id")["rank"]


def spec_list():
    dc = META["def_columns"]
    specs = [("Baseline", {})]
    specs += [("Pressure = leverage top 10%", {"pcol": dc["DPI: leverage top 10%"]}),
              ("Pressure = leverage top 30%", {"pcol": dc["DPI: leverage top 30%"]}),
              ("Pressure = phase-relative leverage", {"pcol": dc["Phase-relative leverage top 20%"]}),
              ("Pressure = close game after over 12", {"pcol": dc["Close game (WP 25-75%) after over 12"]}),
              ("Pressure = death overs (conventional)", {"pcol": dc["Death overs (16-20)"]}),
              ("Pressure = chase RRR >= 10 (conventional)", {"pcol": dc["Chase RRR >= 10"]}),
              ("Min sample x0.67", {"min_scale": 2 / 3}), ("Min sample x1.25", {"min_scale": 1.25}),
              ("Min sample x1.67", {"min_scale": 5 / 3}),
              ("IPL only", {"filter": "ipl"}), ("T20I only", {"filter": "t20i"}),
              ("All matches incl. Associates/A-teams", {"filter": "all"}),
              ("First innings only", {"filter": "inn1"}), ("Second innings only", {"filter": "inn2"}),
              ("Model: single-currency ridge", {"method": "single"}), ("Model: WPA ridge", {"method": "wpa"}),
              ("Model: no shrinkage (raw pressure RAE)", {"method": "raw"}),
              ("Model: posterior mean (no risk adjustment)", {"conservative": False}),
              ("Model: 4x stronger shrinkage", {"lam_scale": 4.0}), ("Model: 4x weaker shrinkage", {"lam_scale": 0.25}),
              ("Baseline: no opponent adjustment", {"no_opp": True}),
              ("Conventional: pressure SR / economy", {"method": "conventional"})]
    for S in (2022, 2023, 2024, 2025, 2026):
        specs.append((f"Leave out season {S}", {"drop_season": S}))
    return specs


def main():
    b = pd.read_parquet(PROC / "balls_final.parquet")
    res = {}
    for side in ("bat", "bowl"):
        boot = np.load(PROC / f"boot_{side}.npz", allow_pickle=True)
        sd = pd.Series(100 * boot["PRS"].std(0), index=boot["actors"])
        ranks, summ = {}, []
        for name, sp in spec_list():
            if sp.get("filter") == "all":
                u = b
            else:
                u = b[universe(b)]
            if sp.get("filter") == "ipl":
                u = u[u.comp_group == "IPL"]
            elif sp.get("filter") == "t20i":
                u = u[u.comp_group == "T20I"]
            elif sp.get("filter") == "inn1":
                u = u[u.innings == 1]
            elif sp.get("filter") == "inn2":
                u = u[u.innings == 2]
            if "drop_season" in sp:
                u = u[u.season != sp["drop_season"]]
            u = u.copy()
            if "pcol" in sp:
                u["pressure"] = u[sp["pcol"]].astype(bool)
            s = u[u.faced].reset_index(drop=True) if side == "bat" else u.reset_index(drop=True)
            r = rate(s, side, sp.get("method", "2c"), sp.get("lam_scale", 1.0), sp.get("no_opp", False))
            mb = MIN[side] * sp.get("min_scale", 1.0)
            if sp.get("filter") in ("ipl", "t20i", "inn1", "inn2") or "drop_season" in sp:
                mb = mb * 0.6     # less data in a subset - scale eligibility down
            if "pcol" in sp:      # some definitions mark more or fewer balls
                share = u[sp["pcol"]].mean() / 0.2
                mb = mb * min(max(share, 0.5), 1.8)
            ranks[name] = ranked(r, sd, mb, sp.get("conservative", True) and sp.get("method", "2c") != "raw")
        R = pd.DataFrame(ranks)
        base = R["Baseline"]
        for name in R.columns:
            both = pd.DataFrame({"Baseline": base, "x": R[name]}).dropna()
            top_b = set(base[base <= 10].index)
            top_s = set(R[name][R[name] <= 10].index)
            summ.append({"specification": name, "spearman_vs_baseline": float(both.Baseline.corr(both.x, method="spearman")) if len(both) > 5 else np.nan,
                         "top10_overlap": len(top_b & top_s), "n_eligible": int(R[name].notna().sum()),
                         "baseline_top10_eligible": int(R.loc[list(top_b), name].notna().sum())})
        S = pd.DataFrame(summ)
        R.to_csv(OUT / f"robustness_ranks_{side}.csv")
        S.to_csv(OUT / f"robustness_summary_{side}.csv", index=False)
        top_freq = (R.drop(columns=["Baseline"]) <= 10).sum(1) / (R.shape[1] - 1)
        elig = R.drop(columns=["Baseline"]).notna().sum(1)
        med_rank = R.drop(columns=["Baseline"]).median(1)
        tf = pd.DataFrame({"top10_share_all_specs": top_freq, "specs_eligible": elig, "median_rank": med_rank,
                           "baseline_rank": base}).sort_values("top10_share_all_specs", ascending=False)
        tf.to_csv(OUT / f"robustness_top10_frequency_{side}.csv")
        res[side] = {"mean_spearman": float(S.spearman_vs_baseline.iloc[1:].mean()),
                     "mean_top10_overlap": float(S.top10_overlap.iloc[1:].mean())}
        print(side)
        print(S.round(2).to_string())
        print(tf.head(20))
    save_json(res, "robustness_overview.json")


if __name__ == "__main__":
    main()
