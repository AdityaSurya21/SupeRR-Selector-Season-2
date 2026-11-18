"""Step 6 - Which measure of 'pressure performance' actually predicts future pressure performance?

This is the test that decides the final rating. For each candidate metric we estimate it
on one block of seasons and check how well it predicts the player's *raw* pressure-ball
RAE per 100 balls in seasons the metric has never seen.

Designs
* Forward chronological: estimate on 2022-24, predict 2025-26 (the real selection problem).
* Leave-one-season-out (LOSO): for each season S, estimate on the other seasons, predict S.
* Split-half repeatability: odd vs even matches (match ids), same player.

Metrics compared: conventional pressure strike rate / average / economy, raw pressure
RAE, raw 'clutch delta' (pressure minus normal), PAPM normal level, PAPM pressure level,
PAPM shift alone, PAPM on WPA, and match-level consistency.
"""
import numpy as np
import pandas as pd
from common import PROC, save_json
import ratings_core as rc
from s05_ratings import universe

from common import load_json
_H = load_json("rating_hyperparameters.json")
LAM = {k: (_H[k]["single"]["lam_base"], _H[k]["single"]["lam_delta"]) for k in ("bat", "bowl")}
LAM2 = {k: {"run": (_H[k]["runs"]["lam_base"], _H[k]["runs"]["lam_delta"]),
            "wkt": (_H[k]["wickets"]["lam_base"], _H[k]["wickets"]["lam_delta"])} for k in ("bat", "bowl")}
MIN_TRAIN, MIN_TEST = {"bat": 60, "bowl": 90}, {"bat": 40, "bowl": 60}


def side_cfg(side):
    return ("striker_id", "bowler_id", "bat_rae", "bat_wpa") if side == "bat" else ("bowler_id", "striker_id", "bowl_rae", "bowl_wpa")


def candidate_metrics(s, side):
    actor, opp, y, yw = side_cfg(side)
    des = rc.build(s, actor, opp, "pressure")
    c = rc.fit(des, s[y].values, *LAM[side])
    cw = rc.fit(des, 100 * s[yw].values, 1600, 6400)
    _, cr, ck = rc.two_component(s, side, LAM2[side]["run"], LAM2[side]["wkt"], des=des)
    cp, cn = rc.wicket_price(s, actor, des["actors"])
    nor2, prs2 = rc.combine(cr, ck, cp, cn)
    r = pd.DataFrame({"PAPM normal level": 100 * nor2, "PAPM pressure level": 100 * prs2,
                      "PAPM pressure shift only": 100 * (cr["d"] + cp * ck["d"]),
                      "PAPM single-currency pressure level": 100 * (c["a"] + c["d"]),
                      "PAPM pressure WPA": 100 * (cw["a"] + cw["d"])},
                     index=des["actors"])
    p, n = s[s.pressure], s[~s.pressure]
    gp, gn = p.groupby(actor), n.groupby(actor)
    r["pressure balls"] = gp.size()
    r["Raw pressure RAE/100"] = 100 * gp[y].mean()
    r["Raw clutch delta (pressure - normal RAE/100)"] = 100 * (gp[y].mean() - gn[y].mean())
    if side == "bat":
        r["Pressure strike rate"] = 100 * gp.bat_runs.mean()
        r["Pressure average"] = gp.bat_runs.sum() / gp.striker_out.sum().replace(0, np.nan)
        r["Overall strike rate"] = 100 * s.groupby(actor).bat_runs.mean()
        r["Overall average"] = s.groupby(actor).bat_runs.sum() / s.groupby(actor).striker_out.sum().replace(0, np.nan)
    else:
        r["Pressure economy (negated)"] = -6 * gp.bowler_runs.sum() / gp.legal.sum()
        r["Pressure wickets per ball"] = gp.bowler_wicket.mean()
        r["Overall economy (negated)"] = -6 * s.groupby(actor).bowler_runs.sum() / s.groupby(actor).legal.sum()
        r["Overall wickets per ball"] = s.groupby(actor).bowler_wicket.mean()
    mm = p.groupby([actor, "match_id"])[y].agg(["sum", "size"])
    mm = mm[mm["size"] >= (3 if side == "bat" else 6)]
    pos = mm.groupby(level=0)["sum"].apply(lambda x: (x > 0).mean())
    cnt = mm.groupby(level=0).size()
    r["Match consistency (share positive)"] = (pos * cnt + 6 * 0.5) / (cnt + 6)
    return r


def target(s, side):
    actor, _, y, _ = side_cfg(side)
    p = s[s.pressure]
    g = p.groupby(actor)[y]
    return pd.DataFrame({"y": 100 * g.mean(), "n": g.size()})


def wcorr(x, y, w):
    m = ~(np.isnan(x) | np.isnan(y))
    x, y, w = x[m], y[m], w[m]
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    return float(np.sum(w * (x - mx) * (y - my)) / np.sqrt(np.sum(w * (x - mx) ** 2) * np.sum(w * (y - my) ** 2)))


def spearman(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    return float(pd.Series(x[m]).rank().corr(pd.Series(y[m]).rank()))


def evaluate(tr, te, side):
    cm = candidate_metrics(tr, side)
    tg = target(te, side)
    j = cm.join(tg, how="inner")
    j = j[(j["pressure balls"] >= MIN_TRAIN[side]) & (j.n >= MIN_TEST[side])]
    out = {}
    for col in cm.columns:
        if col == "pressure balls":
            continue
        out[col] = {"weighted_r": wcorr(j[col].values.astype(float), j.y.values, j.n.values.astype(float)),
                    "spearman": spearman(j[col].values.astype(float), j.y.values)}
    # does consistency or WPA add anything beyond PAPM pressure level? (weighted OLS)
    import statsmodels.api as sm
    X = j[["PAPM pressure level", "Match consistency (share positive)", "PAPM pressure WPA"]].astype(float)
    X = (X - X.mean()) / X.std()
    fit = sm.WLS(j.y, sm.add_constant(X), weights=j.n).fit()
    out["_incremental_regression"] = {"params": fit.params.to_dict(), "pvalues": fit.pvalues.to_dict(), "n_players": int(len(j))}
    return out, j


def main():
    b = pd.read_parquet(PROC / "balls_final.parquet")
    u = b[universe(b)].reset_index(drop=True)
    res = {}
    for side in ("bat", "bowl"):
        s = u[u.faced].reset_index(drop=True) if side == "bat" else u
        r = {}
        fwd, jf = evaluate(s[s.season <= 2024], s[s.season >= 2025], side)
        r["forward_2022_24_to_2025_26"] = fwd
        jf.to_csv(PROC / f"forward_validation_{side}.csv")
        loso_rows = []
        for S in sorted(s.season.unique()):
            o, j = evaluate(s[s.season != S], s[s.season == S], side)
            j["season"] = S
            loso_rows.append(j)
        L = pd.concat(loso_rows)
        loso = {}
        for col in [c for c in L.columns if c not in ("y", "n", "season", "pressure balls")]:
            loso[col] = {"weighted_r": wcorr(L[col].values.astype(float), L.y.values, L.n.values.astype(float)),
                         "spearman": spearman(L[col].values.astype(float), L.y.values)}
        loso["_n_player_seasons"] = int(len(L))
        r["leave_one_season_out_pooled"] = loso
        # split-half repeatability (odd/even match ids)
        odd = s.match_id % 2 == 1
        A = candidate_metrics(s[odd], side)
        Bm = candidate_metrics(s[~odd], side)
        j = A.join(Bm, lsuffix="_A", rsuffix="_B", how="inner")
        j = j[(j["pressure balls_A"] >= MIN_TEST[side]) & (j["pressure balls_B"] >= MIN_TEST[side])]
        sh = {}
        for col in A.columns:
            if col == "pressure balls":
                continue
            x, y = j[col + "_A"].astype(float).values, j[col + "_B"].astype(float).values
            m = ~(np.isnan(x) | np.isnan(y))
            sh[col] = float(np.corrcoef(x[m], y[m])[0, 1])
        sh["_n_players"] = int(len(j))
        r["split_half_repeatability"] = sh
        res[side] = r
        print(side, "forward:", {k: round(v["weighted_r"], 3) for k, v in fwd.items() if not k.startswith("_")})
        print(side, "LOSO:", {k: round(v["weighted_r"], 3) for k, v in loso.items() if not k.startswith("_")})
        print(side, "split-half:", {k: round(v, 3) for k, v in sh.items() if not k.startswith("_")})
        print(side, "incremental:", fwd["_incremental_regression"])
    save_json(res, "predictive_validation.json")


if __name__ == "__main__":
    main()
