"""Step 2 - Win probability, leverage and the Dynamic Pressure Index.

Architecture (one coherent match-state model for both innings):

1. Chase model  C(s):  P(chasing side wins | runs needed, balls left, wickets in hand,
   target, scoring environment, Elo gap). Gradient boosting with *monotone constraints*
   so it can never claim that needing more runs, having fewer balls or losing a
   wicket helps the chasing side.
2. First-innings model: the remaining-runs distribution R | state ~ Normal(mu(s), sigma(s)),
   fitted with two boosted regressions (mean, then variance). WP1(s) is obtained by
   integrating the chase model over the projected target:
        WP1(s) = E_R[ 1 - C(target = runs + R + 1, full resources) ]
   This uses the full information in the final score rather than only win/loss,
   and it makes the WP at the end of innings 1 equal to the WP at the first ball of
   innings 2 by construction.
3. Outcome model: P(dot, 1, 2, 3, 4, 6, wicket, wide/no-ball | state), multinomial boosting.
4. Leverage:  LI_raw(s) = sum_o P(o | s) * |WP(s after o) - WP(s)|, the expected swing in
   win probability carried by this delivery. LI = LI_raw / mean(LI_raw), so LI = 1 is an
   average ball and LI = 3 carries three times the usual match-deciding weight.

Every ball's WP and LI come from models trained on other matches only (5-fold
GroupKFold by match) so that no delivery is scored by a model that has seen its
own outcome. A separate chronological split (train 2022-24, test 2025-26)
measures true forecasting accuracy.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from common import PROC, SEED, save_json

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(24)      # nodes/weights for E[f(Z)], Z~N(0,1)
GH_W = GH_W / GH_W.sum()
T_GRID = np.arange(1, 341)

OUTCOMES = ["0", "1", "2", "3", "4", "6", "W", "X"]
OUT_RUNS = np.array([0, 1, 2, 3, 4, 6, 0, 1.14])
OUT_WKT = np.array([0, 0, 0, 0, 0, 0, 1, 0])
OUT_LEGAL = np.array([1, 1, 1, 1, 1, 1, 1, 0])

# Candidate feature sets. Monotone directions: -1 = can only lower WP, +1 = can only raise it.
MONO = {"runs_needed": -1, "balls_left": 1, "wkts_in_hand": 1, "rrr": -1, "target": 0,
        "env_rpo": 0, "elo_diff": 1, "runs_before": 0, "balls_total": 1, "crr_f": 0,
        "runs_needed_rel": -1, "rrr_rel": -1}
ENV_REF = 8.5   # reference run rate; "_rel" specifications express runs in units of the scoring environment
CHASE_SETS = {
    "state": ["runs_needed", "balls_left", "wkts_in_hand", "rrr"],
    "state_rel": ["runs_needed_rel", "balls_left", "wkts_in_hand", "rrr_rel"],
    "state_rel+elo": ["runs_needed_rel", "balls_left", "wkts_in_hand", "rrr_rel", "elo_diff"],
    "state+env": ["runs_needed", "balls_left", "wkts_in_hand", "rrr", "env_rpo"],
    "state+env+elo": ["runs_needed", "balls_left", "wkts_in_hand", "rrr", "env_rpo", "elo_diff"],
}
RR_SETS = {
    "state": ["balls_left", "wkts_in_hand", "balls_total"],
    "state+env": ["balls_left", "wkts_in_hand", "balls_total", "env_rpo"],
    "state+env+elo": ["balls_left", "wkts_in_hand", "balls_total", "env_rpo", "elo_diff"],
    "state+env+momentum": ["balls_left", "wkts_in_hand", "balls_total", "env_rpo", "crr_f"],
    "state_rel": ["balls_left", "wkts_in_hand", "balls_total"],
    "state_rel+elo": ["balls_left", "wkts_in_hand", "balls_total", "elo_diff"],
}
RR_MONO_OVERRIDE = {"env_rpo": 1}
OUT_FEATS = ["innings", "balls_left", "wkts_in_hand", "runs_needed_f", "rrr_f", "crr_f", "env_rpo", "balls_total"]


def outcome_class(b):
    cls = np.where(~b.legal, "X", np.where(b.team_wicket, "W", ""))
    r = b.total_runs.round().clip(0, 6).astype(int).replace({5: 4})
    cls = np.where(cls == "", r.astype(str), cls)
    return pd.Series(cls, index=b.index)


def _gfit(model, X, y, groups, seed, w=None):
    """Fit with early stopping on a *match-grouped* validation split. Row-level random
    validation lets the booster keep improving by memorising matches it has already seen."""
    r = np.random.default_rng(seed)
    u = np.unique(groups)
    val = np.isin(groups, r.choice(u, max(1, int(0.2 * len(u))), replace=False))
    kw = {} if w is None else {"sample_weight": w[~val], "sample_weight_val": w[val]}
    model.fit(X[~val], y[~val], X_val=X[val], y_val=y[val], **kw)
    return model


class WPModel:
    def __init__(self, seed=SEED, chase_set="state", rr_set="state+env"):
        self.seed = seed
        self.cf, self.rf = CHASE_SETS[chase_set], RR_SETS[rr_set]
        self.rr_rel = rr_set.startswith("state_rel")
        kw = dict(learning_rate=0.05, max_iter=1000, max_leaf_nodes=15, min_samples_leaf=1000,
                  l2_regularization=1.0, early_stopping=True, n_iter_no_change=30, random_state=seed)
        self.chase = HistGradientBoostingClassifier(monotonic_cst=[MONO[f] for f in self.cf], **kw)
        self.rr_mu = HistGradientBoostingRegressor(
            monotonic_cst=[RR_MONO_OVERRIDE.get(f, MONO[f]) for f in self.rf], **kw)
        self.rr_var = HistGradientBoostingRegressor(**kw)
        self.outc = HistGradientBoostingClassifier(**{**kw, "max_leaf_nodes": 31, "min_samples_leaf": 300})

    # ---------------- fitting ----------------
    def fit(self, b):
        ok = b.bat_won.notna() & ~b.is_dls
        c = b[ok & (b.innings == 2) & (b.runs_needed > 0)].copy()
        c["runs_needed_rel"] = c.runs_needed * ENV_REF / c.env_rpo
        c["rrr_rel"] = c.rrr * ENV_REF / c.env_rpo
        _gfit(self.chase, c[self.cf], (c.bat_won >= 0.5).astype(int).values, c.match_id.values, self.seed,
              w=np.where(c.bat_won == 0.5, 0.5, 1.0))
        f = b[ok & (b.innings == 1)].copy()
        f["crr_f"] = f.crr.fillna(-1)
        final = f.groupby("match_id").total_runs.transform("sum")
        y = (final - f.runs_before).values
        if self.rr_rel:
            y = y * ENV_REF / f.env_rpo.values
        _gfit(self.rr_mu, f[self.rf], y, f.match_id.values, self.seed)
        res2 = (y - self.rr_mu.predict(f[self.rf])) ** 2
        _gfit(self.rr_var, f[self.rf], res2, f.match_id.values, self.seed)
        _gfit(self.outc, out_X(b), outcome_class(b).values, b.match_id.values, self.seed)
        return self

    # ---------------- prediction ----------------
    def chase_wp(self, runs_needed, balls_left, wkts_in_hand, target, env, elo):
        runs_needed = np.asarray(runs_needed, float)
        balls_left = np.asarray(balls_left, float)
        wkts = np.asarray(wkts_in_hand, float)
        X = pd.DataFrame({"runs_needed": runs_needed, "balls_left": np.maximum(balls_left, 1),
                          "wkts_in_hand": wkts, "rrr": runs_needed * 6 / np.maximum(balls_left, 1),
                          "target": target, "env_rpo": env, "elo_diff": elo})
        X["runs_needed_rel"] = X.runs_needed * ENV_REF / X.env_rpo
        X["rrr_rel"] = X.rrr * ENV_REF / X.env_rpo
        X = X[self.cf]
        p = np.empty(len(X))
        live = (runs_needed > 0) & (balls_left > 0) & (wkts > 0)
        if live.any():
            p[live] = self.chase.predict_proba(X[live])[:, 1]
        p[runs_needed <= 0] = 1.0
        dead = ~live & (runs_needed > 0)
        p[dead] = np.where(runs_needed[dead] == 1, 0.5, 0.0)   # tie if exactly one short
        return p

    def chase_curve(self, balls2, env, elo_chaser):
        """C(T) for T in T_GRID at the start of a chase."""
        n = len(T_GRID)
        return self.chase_wp(T_GRID, np.full(n, balls2), np.full(n, 10), T_GRID, np.full(n, env), np.full(n, elo_chaser))

    def first_inn_wp(self, runs, wkts_in_hand, balls_left, balls_total, env, elo, curve):
        """Batting-first WP by integrating over the projected final total."""
        runs = np.asarray(runs, float)
        bt = np.asarray(balls_total, float)
        crr = np.where(bt - np.asarray(balls_left) > 0, runs * 6 / np.maximum(bt - np.asarray(balls_left), 1), -1)
        X = pd.DataFrame({"balls_left": balls_left, "wkts_in_hand": wkts_in_hand, "runs_before": runs,
                          "balls_total": balls_total, "env_rpo": env, "elo_diff": elo, "crr_f": crr})
        done = (np.asarray(balls_left) <= 0) | (np.asarray(wkts_in_hand) <= 0)
        k = (np.asarray(env, float) / ENV_REF) if self.rr_rel else 1.0
        mu = np.where(done, 0.0, np.maximum(self.rr_mu.predict(X[self.rf]), 0) * k)
        sd = np.where(done, 0.0, np.sqrt(np.clip(self.rr_var.predict(X[self.rf]), 1, None)) * k)
        T = runs[:, None] + mu[:, None] + sd[:, None] * GH_X[None, :] + 1
        T = np.clip(T, runs[:, None] + 1, T_GRID[-1])
        C = np.interp(T, T_GRID, curve)
        return 1 - (C * GH_W[None, :]).sum(1), mu, sd

    def outcome_probs(self, b):
        P = self.outc.predict_proba(out_X(b))
        cols = list(self.outc.classes_)
        return np.column_stack([P[:, cols.index(o)] if o in cols else np.zeros(len(b)) for o in OUTCOMES])


def out_X(b):
    X = pd.DataFrame({
        "innings": b.innings.values, "balls_left": b.balls_left.values, "wkts_in_hand": b.wkts_in_hand.values,
        "runs_needed_f": np.where(b.innings == 2, b.runs_needed, -1),
        "rrr_f": np.where(b.innings == 2, b.rrr.clip(upper=36), -1),
        "crr_f": b.crr.fillna(-1).values, "env_rpo": b.env_rpo.values, "elo_diff": b.elo_diff.values,
        "balls_total": b.balls_total.values})
    return X[OUT_FEATS]


def evaluate_states(model, b):
    """WP before the ball, WP after the actual outcome, WP after each hypothetical outcome."""
    n = len(b)
    wp_before = np.empty(n)
    wp_after = np.empty(n)
    wp_hyp = np.empty((n, len(OUTCOMES)))
    P = model.outcome_probs(b)
    for mid, idx in b.groupby("match_id").indices.items():
        d = b.iloc[idx]
        first = d[d.innings == 1]
        env = d.env_rpo.iloc[0]
        elo1 = first.elo_diff.iloc[0] if len(first) else -d.elo_diff.iloc[0]
        b2 = d.loc[d.innings == 2, "balls_total"].iloc[0] if (d.innings == 2).any() else 120.0
        curve = model.chase_curve(b2, env, -elo1)
        for inn in (1, 2):
            sel = np.where(d.innings.values == inn)[0]
            if len(sel) == 0:
                continue
            s = d.iloc[sel]
            gi = idx[sel]
            r0, w0, bl0 = s.runs_before.values, s.wkts_in_hand.values, s.balls_left.values
            # after actual
            r1 = s.runs_after.values
            w1 = 10 - s.wkts_after.values
            bl1 = s.balls_left_after.values
            hr = r0[:, None] + OUT_RUNS[None, :]
            hw = w0[:, None] - OUT_WKT[None, :]
            hb = bl0[:, None] - OUT_LEGAL[None, :]
            if inn == 1:
                bt = s.balls_total.values
                f = lambda r, w, bl: model.first_inn_wp(r, w, bl, np.resize(bt, len(r)), env, elo1, curve)[0]
                wp_before[gi] = f(r0, w0, bl0)
                wp_after[gi] = f(r1, w1, bl1)
                wp_hyp[gi] = f(hr.ravel(), hw.ravel(), hb.ravel()).reshape(hr.shape)
            else:
                tgt = s.target.values
                elo2 = s.elo_diff.values
                last = s.inn_last.values
                rn0 = np.where(last, tgt - r0, np.maximum(tgt - r0, 1))
                wp_before[gi] = model.chase_wp(rn0, bl0, w0, tgt, env, elo2)
                rn1 = np.where(last, tgt - r1, np.maximum(tgt - r1, 1))
                wa = model.chase_wp(rn1, bl1, w1, tgt, env, elo2)
                truth = s.bat_won.values
                wa = np.where(last & ~np.isnan(truth), truth, wa)   # the last ball resolves to the real result
                wp_after[gi] = wa
                rep = lambda v: np.repeat(v, len(OUTCOMES))
                hrn = (tgt[:, None] - hr).ravel()
                wp_hyp[gi] = model.chase_wp(hrn, hb.ravel(), hw.ravel(), rep(tgt), env, rep(elo2)).reshape(hr.shape)
    return wp_before, wp_after, wp_hyp, P


def metrics(y, p, w=None):
    keep = ~np.isnan(y)
    y, p = y[keep], np.clip(p[keep], 1e-4, 1 - 1e-4)
    yb = (y >= 0.5).astype(int)
    return {"n": int(len(y)), "brier": float(brier_score_loss(yb, p)), "log_loss": float(log_loss(yb, p)),
            "auc": float(roc_auc_score(yb, p))}


def calibration_table(y, p, bins=10):
    keep = ~np.isnan(y)
    d = pd.DataFrame({"y": (y[keep] >= 0.5).astype(float), "p": p[keep]})
    d["bin"] = pd.cut(d.p, np.linspace(0, 1, bins + 1), include_lowest=True)
    t = d.groupby("bin", observed=True).agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")).reset_index()
    t["bin"] = t.bin.astype(str)
    return t.to_dict("records")


def _logistic_baseline(train, test):
    """Hand-built cricket features + logistic regression: the strongest 'simple' rival."""
    def feats(d):
        X = pd.DataFrame({
            "inn2": (d.innings == 2).astype(int),
            "rrr_minus_crr": np.where(d.innings == 2, d.rrr.clip(upper=36) - d.crr.fillna(d.env_rpo), 0),
            "log_rn": np.where(d.innings == 2, np.log1p(d.runs_needed.clip(lower=0)), 0),
            "log_bl": np.log1p(d.balls_left), "wkts": d.wkts_in_hand,
            "proj_minus_par": np.where(d.innings == 1, d.runs_before + d.crr.fillna(d.env_rpo) * d.balls_left / 6
                                       - d.env_rpo * d.balls_total / 6, 0),
            "elo": d.elo_diff / 100})
        X["wk_x_inn2"] = X.wkts * X.inn2
        return X
    tr = train[train.bat_won.notna() & ~train.is_dls]
    lr = LogisticRegression(max_iter=3000, C=1.0).fit(feats(tr), (tr.bat_won >= 0.5).astype(int))
    return lr.predict_proba(feats(test))[:, 1]


def _score(y, p, inn):
    return {"all": metrics(y, p), "innings_1": metrics(y[inn == 1], p[inn == 1]),
            "innings_2": metrics(y[inn == 2], p[inn == 2])}


def chronological_validation(b):
    """Three-way chronological design, so that the reported test score is honest:
       * selection:  train 2022-23, validate on 2024 -> choose the feature specification
       * test:       train 2022-24 with the chosen specification, forecast 2025-26."""
    res = {"specifications_selection_2024": {}}
    sel_tr, sel_va = b[b.season <= 2023], b[b.season == 2024]
    yv, iv = sel_va.bat_won.values, sel_va.innings.values
    best, best_brier = None, 9
    for cs in ["state", "state+env+elo", "state_rel", "state_rel+elo"]:
        for rs in ["state+env", "state+env+elo", "state_rel", "state_rel+elo"]:
            m = WPModel(chase_set=cs, rr_set=rs).fit(sel_tr)
            wb, _, _, _ = evaluate_states(m, sel_va)
            sc = _score(yv, wb, iv)
            res["specifications_selection_2024"][f"chase[{cs}] | first-innings[{rs}]"] = sc
            print(cs, rs, {k: round(v["brier"], 4) for k, v in sc.items()})
            if sc["all"]["brier"] < best_brier:
                best, best_brier = (cs, rs), sc["all"]["brier"]
    res["selected"] = {"chase_set": best[0], "rr_set": best[1]}
    train = b[b.season <= 2024]
    test = b[b.season >= 2025].copy()
    y, inn = test.bat_won.values, test.innings.values
    m = WPModel(chase_set=best[0], rr_set=best[1]).fit(train)
    best_wb, _, _, best_P = evaluate_states(m, test)
    res["selected_test_2025_26"] = _score(y, best_wb, inn)
    res["selected_calibration"] = calibration_table(y, best_wb)
    pl = _logistic_baseline(train, test)
    res["logistic_handcrafted"] = _score(y, pl, inn)
    f1 = ["runs_before", "wkts_in_hand", "balls_left", "balls_total", "env_rpo"]
    tr = train[train.bat_won.notna() & ~train.is_dls & (train.innings == 1)]
    g = _gfit(HistGradientBoostingClassifier(monotonic_cst=[1, 1, 1, 0, -1], learning_rate=0.05, max_iter=1000,
                                             max_leaf_nodes=15, min_samples_leaf=1000, random_state=SEED),
              tr[f1], (tr.bat_won >= 0.5).astype(int).values, tr.match_id.values, SEED)
    s1 = inn == 1
    res["direct_gbm_first_innings"] = {"innings_1": metrics(y[s1], g.predict_proba(test[s1][f1])[:, 1])}
    res["coin_flip"] = {"all": metrics(y, np.full(len(y), 0.5))}
    cls = outcome_class(test)
    base = outcome_class(train).value_counts(normalize=True).reindex(OUTCOMES).fillna(1e-6).values
    idx = np.array([OUTCOMES.index(c) for c in cls])
    res["outcome_model"] = {"log_loss_model": float(-np.log(np.clip(best_P[np.arange(len(idx)), idx], 1e-9, 1)).mean()),
                            "log_loss_base_rates": float(-np.log(base[idx]).mean())}
    return res


def main():
    b = pd.read_parquet(PROC / "balls.parquet")
    print("chronological validation ...")
    val = chronological_validation(b)
    sel = val["selected"]
    print("selected", sel, "logistic", val["logistic_handcrafted"]["all"])

    print("out-of-fold scoring ...")
    gkf = GroupKFold(n_splits=5)
    n = len(b)
    wb, wa = np.empty(n), np.empty(n)
    wh = np.empty((n, len(OUTCOMES)))
    P = np.empty((n, len(OUTCOMES)))
    for k, (tr, te) in enumerate(gkf.split(b, groups=b.match_id)):
        m = WPModel(seed=SEED + k, chase_set=sel["chase_set"], rr_set=sel["rr_set"]).fit(b.iloc[tr])
        bte = b.iloc[te]
        x1, x2, x3, x4 = evaluate_states(m, bte)
        wb[te], wa[te], wh[te], P[te] = x1, x2, x3, x4
        print(" fold", k, "done")
    b["wp"] = wb
    b["wp_after"] = wa
    b["dwp"] = wa - wb                                     # batting-side win probability added
    swing = np.abs(wh - wb[:, None])
    b["li_raw"] = (P * swing).sum(1)
    b["li_wkt_raw"] = P[:, OUTCOMES.index("W")] * swing[:, OUTCOMES.index("W")]
    b["li_run_raw"] = b.li_raw - b.li_wkt_raw
    b["p_wicket"] = P[:, OUTCOMES.index("W")]
    b["exp_runs_state"] = (P * OUT_RUNS[None, :]).sum(1)
    mean_li = b.li_raw.mean()
    b["LI"] = b.li_raw / mean_li
    b["wicket_share_of_leverage"] = b.li_wkt_raw / b.li_raw.clip(lower=1e-9)
    # OOF metrics
    y = b.bat_won.values
    oof = {"all": metrics(y, wb), "innings_1": metrics(y[b.innings.values == 1], wb[b.innings.values == 1]),
           "innings_2": metrics(y[b.innings.values == 2], wb[b.innings.values == 2]),
           "calibration": calibration_table(y, wb)}
    # continuity at innings break
    last1 = b[(b.innings == 1) & b.inn_last]
    first2 = b[(b.innings == 2)].drop_duplicates("match_id")
    j = last1[["match_id", "wp_after"]].merge(first2[["match_id", "wp"]], on="match_id")
    cont = float(np.abs(j.wp_after - (1 - j.wp)).median())
    # martingale check: mean dWP should be ~0 in live states
    mart = float(b.loc[~b.inn_last, "dwp"].mean())
    val["out_of_fold"] = oof
    val["innings_break_continuity_median_abs_gap"] = cont
    val["mean_dwp_live_balls"] = mart
    val["li_mean_raw"] = float(mean_li)
    val["li_quantiles"] = {str(q): float(b.LI.quantile(q)) for q in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)}
    save_json(val, "wp_validation.json")
    b.to_parquet(PROC / "balls_wp.parquet", index=False)
    print(oof["all"], "continuity", cont, "martingale", mart)


if __name__ == "__main__":
    main()
