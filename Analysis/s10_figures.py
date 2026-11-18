"""Step 10 - Figures for the report. Every chart answers one analytical question (see titles)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from common import PROC, OUT, FIG, load_json
from s05_ratings import universe

BLUE, PINK, GOLD = "#254AA5", "#E5197F", "#B8860B"
INK, MUTED, GRID, GRAY, LIGHT = "#1E1E24", "#5A5A66", "#E6E6EA", "#A3A3AD", "#DCE3F3"
BLUES = LinearSegmentedColormap.from_list("rrblue", ["#F4F6FB", "#B9C6E8", "#6F88CC", BLUE, "#14295E"])

for f in font_manager.findSystemFonts():
    if "Inter" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": "Inter", "font.size": 9, "axes.edgecolor": GRAY, "axes.labelcolor": MUTED,
    "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110, "savefig.dpi": 220,
    "savefig.bbox": "tight", "legend.frameon": False, "axes.axisbelow": True,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", facecolor="white")
    plt.close(fig)


def short(n):
    p = n.split()
    return n if len(n) <= 16 else (p[0][0] + ". " + " ".join(p[1:]))


def f_where_pressure_lives(b):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    for ax, inn in zip(axes, (1, 2)):
        d = b[b.innings == inn]
        t = d.groupby([d.wkts_before.clip(upper=8), "over"]).LI.mean().unstack()
        n = d.groupby([d.wkts_before.clip(upper=8), "over"]).size().unstack()
        t = t.where(n >= 40)
        im = ax.imshow(t.values, aspect="auto", cmap=BLUES, vmin=0, vmax=3.5, origin="lower",
                       extent=[0.5, 20.5, -0.5, t.shape[0] - 0.5])
        ax.set_title(f"{'First' if inn == 1 else 'Second'} innings", pad=6)
        ax.set_xlabel("Over")
        ax.set_xticks([1, 6, 10, 15, 20])
        ax.grid(False)
    axes[0].set_ylabel("Wickets down (8 = 8+)")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("Mean Dynamic Pressure Index (1 = average ball)")
    cb.outline.set_visible(False)
    save(fig, "F01_where_pressure_lives")


def f_definition_test():
    c = pd.read_csv(PROC / "pressure_definition_comparison.csv").sort_values("multiplier")
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    col = [BLUE if ("leverage" in d or "DPI" in d) else GRAY for d in c.definition]
    y = np.arange(len(c))
    ax.barh(y, c.multiplier, color=col, height=0.55)
    ax.errorbar(c.multiplier, y, xerr=1.96 * c.multiplier_se, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
    ax.axvline(1, color=PINK, lw=1.2, ls="--")
    ax.text(1.04, 0.6, "x1 = no more decisive\nthan an ordinary ball", color=PINK, fontsize=7.5, va="center")
    ax.set_yticks(y, c.definition)
    ax.set_xlabel("How much more a run above expectation is worth on a 'pressure' ball (logit coefficient ratio, 95% CI)")
    ax.grid(axis="y", visible=False)
    for yi, (m, s) in enumerate(zip(c.multiplier, c.share_balls)):
        ax.text(0.03, yi, f"{100*s:.0f}% of balls", va="center", fontsize=7, color="white" if m > 0.4 else INK)
    save(fig, "F02_pressure_definition_test")


def f_wp_validation():
    v = load_json("wp_validation.json")
    cal = pd.DataFrame(v["selected_calibration"])
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1, ls="--")
    ax.scatter(cal.pred, cal.obs, s=np.sqrt(cal.n) * 1.2, color=BLUE, zorder=3, edgecolor="white", lw=1.5)
    ax.plot(cal.pred, cal.obs, color=BLUE, lw=1.5)
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Observed win rate")
    ax.set_title("Calibration on unseen 2025-26 matches")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax = axes[1]
    rows = [("No information (50/50)", v["coin_flip"]["all"]["brier"], GRAY),
            ("Logistic, hand-built features", v["logistic_handcrafted"]["all"]["brier"], GRAY),
            ("Selected boosted model", v["selected_test_2025_26"]["all"]["brier"], BLUE)]
    for i, (nm, br, c) in enumerate(rows):
        ax.barh(i, br, color=c, height=0.5)
        ax.text(br + 0.003, i, f"{br:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(3), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.29)
    ax.set_xlabel("Brier score, 2025-26 test seasons (lower is better)")
    ax.set_title("Forecast accuracy against alternatives")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "F03_wp_validation")


def f_behaviour():
    t = pd.read_csv(PROC / "behaviour_by_pressure_decile.csv")
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.8))
    x = np.arange(1, 11)
    for ax, col, lab in zip(axes, ["strike_rate", "dismissal_pct", "share_inn2"],
                            ["Strike rate", "Dismissals per 100 balls", "Share of balls in a chase"]):
        vals = t[col] * (100 if col == "share_inn2" else 1)
        ax.plot(x, vals, color=BLUE, lw=2, marker="o", ms=4.5, mec="white", mew=1)
        ax.set_title(lab)
        ax.set_xticks([1, 5, 10], ["D1\nlowest", "D5", "D10\nhighest"])
        ax.set_xlabel("Pressure decile (DPI)")
    axes[2].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    fig.tight_layout()
    save(fig, "F04_behaviour_by_pressure")


def f_reliability():
    v = load_json("predictive_validation.json")
    keep = ["Raw clutch delta (pressure - normal RAE/100)", "Raw pressure RAE/100", "Match consistency (share positive)",
            "PAPM pressure WPA", "PAPM pressure shift only", "PAPM single-currency pressure level", "PAPM pressure level"]
    lab = {"Raw clutch delta (pressure - normal RAE/100)": "Raw 'clutch' delta", "Raw pressure RAE/100": "Raw pressure runs above exp.",
           "Match consistency (share positive)": "Match-level consistency", "PAPM pressure WPA": "Shrunk pressure WPA",
           "PAPM pressure shift only": "Shrunk pressure shift", "PAPM single-currency pressure level": "Shrunk pressure level (1 currency)",
           "PAPM pressure level": "Shrunk pressure level (final)"}
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.1), sharey=True)
    for ax, side, ttl in zip(axes, ("bat", "bowl"), ("Batters", "Bowlers")):
        sh = v[side]["split_half_repeatability"]
        vals = [sh[k] for k in keep]
        cols = [BLUE if k == "PAPM pressure level" else (PINK if k.startswith("Raw") else GRAY) for k in keep]
        ax.barh(range(len(keep)), vals, color=cols, height=0.55)
        for i, val in enumerate(vals):
            ax.text(val + (0.01 if val >= 0 else -0.01), i, f"{val:+.2f}", va="center", ha="left" if val >= 0 else "right", fontsize=7.5)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_title(f"{ttl}: odd vs even matches (n={sh['_n_players']})")
        ax.set_xlim(-0.25, 0.5)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(keep)), [lab[k] for k in keep])
    fig.supxlabel("Split-half correlation of the same player's measure (higher = more repeatable)", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    save(fig, "F05_repeatability")


def f_shrinkage(fb):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, side, ttl in zip(axes, ("bat", "bowl"), ("Batters", "Bowlers")):
        f = pd.read_parquet(PROC / f"final_{side}.parquet")
        raw = 100 * f.p_rae / f.p_balls
        ax.scatter(raw, f.pressure_rae100, s=f.p_balls / 6, color=GRAY, alpha=0.55, edgecolor="white", lw=0.6)
        t = f.head(10)
        full = pd.read_parquet(PROC / f"full_{side}.parquet")
        gated = full[full.eligible & ~full.evidence_gate & (full.clutch_rating > f.head(10).clutch_rating.min())]
        ax.scatter(gated.raw_pressure_rae100, gated.pressure_rae100, s=gated.p_balls / 6, facecolor="none", edgecolor=PINK, lw=1.2, zorder=3)
        for _, g in gated.iterrows():
            ax.annotate(short(g.player_name) + " (fails evidence gate)", (g.raw_pressure_rae100, g.pressure_rae100), fontsize=6.5,
                        color=PINK, xytext=(5, -10), textcoords="offset points")
        ax.scatter(100 * t.p_rae / t.p_balls, t.pressure_rae100, s=t.p_balls / 6, color=PINK, edgecolor="white", lw=0.8, zorder=3)
        lim = [min(raw.min(), -40), max(raw.max(), 60)]
        ax.plot(lim, lim, color=GRAY, ls="--", lw=0.8)
        ax.axhline(0, color=GRID, lw=1); ax.axvline(0, color=GRID, lw=1)
        offs = [(6, 10), (-60, 12), (6, -12), (-62, -4), (6, -2)]
        for (_, r), o in zip(t.head(5).iterrows(), offs):
            ax.annotate(short(r.player_name), (100 * r.p_rae / r.p_balls, r.pressure_rae100), fontsize=7, color=INK,
                        xytext=o, textcoords="offset points", arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.5))
        ax.set_title(f"{ttl}: raw pressure record vs shrunk estimate")
        ax.set_xlabel("Raw runs above expectation per 100 pressure balls")
        ax.set_ylabel("Model estimate (per 100 pressure balls)")
    fig.text(0.5, -0.03, "Dot size = pressure balls. Points below the dashed line were pulled toward the average because the sample cannot support the raw figure; pink = final top 10.",
             ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    save(fig, "F06_shrinkage")


def f_forest(side):
    f = pd.read_parquet(PROC / f"final_{side}.parquet").head(15).iloc[::-1]
    z = np.load(PROC / f"boot_{side}.npz", allow_pickle=True)
    idx = {a: i for i, a in enumerate(z["actors"])}
    P = 100 * z["PRS"]
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for yi, (_, r) in enumerate(f.iterrows()):
        draws = P[:, idx[r.player_id]]
        p10, p20, p80, p90 = np.percentile(draws, [10, 20, 80, 90])
        top = r["rank"] <= 10
        c = BLUE if top else GRAY
        ax.plot([p10, p90], [yi, yi], color=c, lw=1.2, alpha=0.9)
        ax.plot([p20, p80], [yi, yi], color=c, lw=4.5, solid_capstyle="round", alpha=0.85)
        ax.scatter(r.normal_rae100, yi, marker="o", s=34, facecolor="white", edgecolor=INK, lw=1, zorder=4)
        ax.scatter(r.pressure_rae100, yi, marker="o", s=30, color=c, edgecolor="white", lw=1, zorder=5)
        ax.scatter(r.clutch_rating, yi, marker="D", s=30, color=PINK, edgecolor="white", lw=0.8, zorder=6)
        ax.text(-1.5, yi, f"{int(r['rank'])}. {r.player_name}", ha="right", va="center", fontsize=8,
                color=INK if top else MUTED, fontweight="semibold" if top else "normal")
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks([])
    ax.set_xlim(left=-0.5)
    ax.axhline(len(f) - 10.5, color=GRAY, lw=0.6, ls=":")
    ax.set_xlabel("Runs above expectation per 100 pressure balls")
    ax.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=BLUE, lw=4.5, label="80% / 60% bootstrap range"),
         Line2D([], [], marker="o", color=BLUE, lw=0, label="Pressure level (posterior mean)"),
         Line2D([], [], marker="o", mfc="white", mec=INK, lw=0, label="Normal level"),
         Line2D([], [], marker="D", color=PINK, lw=0, label="Clutch Rating (20th pct, used to rank)")]
    ax.legend(handles=h, loc="lower right", fontsize=7.5)
    save(fig, f"F07_forest_{side}")


def f_normal_vs_pressure():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, side, ttl in zip(axes, ("bat", "bowl"), ("Batters", "Bowlers")):
        f = pd.read_parquet(PROC / f"final_{side}.parquet")
        ax.scatter(f.normal_rae100, f.shift_rae100, s=f.p_balls / 6, color=GRAY, alpha=0.5, edgecolor="white", lw=0.6)
        sig_up = f[f.p_shift_positive >= 0.9]
        sig_dn = f[f.p_shift_positive <= 0.1]
        ax.scatter(sig_up.normal_rae100, sig_up.shift_rae100, s=sig_up.p_balls / 6, color=BLUE, edgecolor="white", lw=0.6, zorder=3)
        ax.scatter(sig_dn.normal_rae100, sig_dn.shift_rae100, s=sig_dn.p_balls / 6, color=PINK, edgecolor="white", lw=0.6, zorder=3)
        for _, r in pd.concat([sig_up.nlargest(5, "shift_rae100"), sig_dn.nsmallest(4, "shift_rae100")]).iterrows():
            ax.annotate(short(r.player_name), (r.normal_rae100, r.shift_rae100), fontsize=6.8, xytext=(4, 2), textcoords="offset points")
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xlabel("Normal level (runs above expectation per 100 balls)")
        ax.set_ylabel("Shift under pressure")
        ax.set_title(f"{ttl}: who rises and who fades?")
    fig.text(0.5, -0.03, "Blue = shift above zero in at least 90% of bootstrap resamples; pink = below zero in at least 90%. Dot size = pressure balls.",
             ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    save(fig, "F08_normal_vs_pressure")


def f_rank_stability(side):
    R = pd.read_csv(OUT / f"robustness_ranks_{side}.csv", index_col=0)
    f = pd.read_parquet(PROC / f"final_{side}.parquet").head(12)
    specs = ["Baseline", "Pressure = leverage top 10%", "Pressure = leverage top 30%", "Pressure = phase-relative leverage",
             "Min sample x1.25", "IPL only", "Second innings only", "Model: single-currency ridge", "Model: WPA ridge",
             "Model: posterior mean (no risk adjustment)", "Model: 4x stronger shrinkage", "Baseline: no opponent adjustment",
             "Leave out season 2024", "Leave out season 2025", "Leave out season 2026"]
    lab = ["Baseline", "Top 10% DPI", "Top 30% DPI", "Phase-relative DPI", "Stricter sample", "IPL only", "Chases only",
           "One-currency model", "WPA model", "No risk adjustment", "4x shrinkage", "No opponent adj.", "Drop 2024", "Drop 2025", "Drop 2026"]
    M = R.loc[f.player_id, specs].values
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    disp = np.where(np.isnan(M), np.nan, np.minimum(M, 25))
    cm = LinearSegmentedColormap.from_list("rk", [BLUE, "#9FB2E0", "#F2F2F5"])
    ax.imshow(disp, cmap=cm, vmin=1, vmax=25, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, "–" if np.isnan(v) else f"{int(v)}", ha="center", va="center", fontsize=7,
                    color="white" if (not np.isnan(v) and v <= 6) else INK)
    ax.set_yticks(range(len(f)), [f"{int(r)}. {n}" for r, n in zip(f["rank"], f.player_name)])
    ax.set_xticks(range(len(lab)), lab, rotation=40, ha="right")
    ax.grid(False)
    ax.set_title(f"{'Batters' if side == 'bat' else 'Bowlers'}: rank under alternative specifications ('–' = below that spec's sample threshold)")
    save(fig, f"F09_rank_stability_{side}")


def f_finisher():
    g = pd.read_csv(PROC / "finisher_quadrant.csv", index_col=0)
    g = g[g.n >= 80]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.scatter(g.outs100, g.runs100, s=g.n / 2.5, color=GRAY, alpha=0.6, edgecolor="white", lw=0.6)
    best = g[(g.runs100 > 4) & (g.outs100 < 0)]
    ax.scatter(best.outs100, best.runs100, s=best.n / 2.5, color=BLUE, edgecolor="white", lw=0.6, zorder=3)
    for _, r in g.nlargest(9, "runs100").iterrows():
        ax.annotate(short(r["name"]), (r.outs100, r.runs100), fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax.axhline(0, color=INK, lw=0.8); ax.axvline(0, color=INK, lw=0.8)
    ax.invert_xaxis()
    ax.set_xlabel("Extra dismissals per 100 balls vs expectation (left = more often out)  →  right = safer")
    ax.set_ylabel("Extra runs per 100 balls vs expectation")
    ax.set_title("Death-overs pressure: scoring faster without getting out more")
    ax.text(0.98, 0.97, "faster AND safer", transform=ax.transAxes, ha="right", va="top", color=BLUE, fontsize=8, fontweight="semibold")
    save(fig, "F10_finisher_quadrant")


def f_prob_top10():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    for ax, side, ttl in zip(axes, ("bat", "bowl"), ("Batters", "Bowlers")):
        f = pd.read_parquet(PROC / f"final_{side}.parquet").head(14).iloc[::-1]
        y = np.arange(len(f))
        ax.barh(y, f.p_top10, color=[BLUE if r <= 10 else GRAY for r in f["rank"]], height=0.6, label="Bootstrap resamples")
        ax.scatter(f.top10_share_all_specs, y, marker="|", s=120, color=PINK, lw=2, zorder=3, label="Model specifications")
        ax.set_yticks(y, [f"{int(r)}. {short(n)}" for r, n in zip(f["rank"], f.player_name)])
        ax.set_title(f"{ttl}: how often in the top 10?")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", visible=False)
    axes[1].legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    save(fig, "F11_top10_probability")


def f_case(b, mid, focus_name, side, fname, title):
    d = b[b.match_id == mid].copy()
    d["ball_ix"] = np.arange(len(d))
    chase_team = d[d.innings == 2].batting_team_id.iloc[0]
    wp_chase = np.where(d.batting_team_id == chase_team, d.wp_after, 1 - d.wp_after)
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 4.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    ax = axes[0]
    ax.plot(d.ball_ix, 100 * wp_chase, color=BLUE, lw=1.6)
    ax.axvline(d[d.innings == 2].ball_ix.min() - 0.5, color=GRAY, lw=0.8, ls=":")
    col = "striker_name" if side == "bat" else "bowler_name"
    f = d[d[col] == focus_name]
    ax.scatter(f.ball_ix, 100 * np.where(f.batting_team_id == chase_team, f.wp_after, 1 - f.wp_after), s=12, color=PINK, zorder=3,
               label=f"Deliveries {'faced' if side == 'bat' else 'bowled'} by {focus_name}")
    w = d[d.team_wicket]
    ax.scatter(w.ball_ix, 100 * np.where(w.batting_team_id == chase_team, w.wp_after, 1 - w.wp_after), marker="x", s=22, color=INK, zorder=4, label="Wicket")
    ax.set_ylabel("Chasing side win prob. (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_title(title)
    ax2 = axes[1]
    ax2.bar(d.ball_ix, d.LI, color=np.where(d[col] == focus_name, PINK, LIGHT), width=1.0)
    ax2.axhline(load_json("pressure_meta.json")["q80_LI"], color=BLUE, lw=0.8, ls="--")
    ax2.set_ylabel("DPI")
    ax2.set_xlabel("Delivery number in the match (first innings | second innings)")
    fig.tight_layout()
    save(fig, fname)


def f_dl():
    tab = np.load(PROC / "wicket_cost_table.npy")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    u = np.arange(121)
    for w in (10, 8, 6, 4, 2):
        ax.plot(120 - u, tab[:, w], color=BLUE if w == 10 else GRAY, lw=1.8 if w == 10 else 1.1)
        ax.text(120 - u[-1] - 1, tab[-1, w], f"{w} wkts in hand", fontsize=7, va="center", ha="right", color=MUTED)
    ax.set_xlabel("Balls bowled")
    ax.set_ylabel("Expected runs still to come")
    ax.set_title("Fitted resource curves (Duckworth-Lewis form) - the price of a wicket")
    ax.set_xlim(-28, 120)
    save(fig, "F12_resource_curves")


def f_exposure():
    ins = load_json("insights.json")
    e = pd.DataFrame(ins["exposure_by_position"])
    p = pd.DataFrame(ins["exposure_by_phase_innings"])
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.8))
    axes[0].bar(e.batting_order.astype(str), 100 * e.pressure_share, color=BLUE, width=0.6)
    axes[0].set_title("Share of balls faced that are pressure balls")
    axes[0].set_xlabel("Batting position")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    order = ["Powerplay", "Middle", "Death"]
    x = np.arange(3)
    for k, (inn, c) in enumerate(((1, GRAY), (2, BLUE))):
        v = p[p.innings == inn].set_index("phase").reindex(order).pressure_share * 100
        axes[1].bar(x + (k - 0.5) * 0.36, v, width=0.34, color=c, label=f"{'First' if inn == 1 else 'Second'} innings")
    axes[1].set_xticks(x, order)
    axes[1].set_title("Pressure balls by phase and innings")
    axes[1].legend(fontsize=7.5)
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    for a in axes:
        a.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "F13_exposure")


def f_robust_summary():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 5.2), sharey=True)
    for ax, side, ttl in zip(axes, ("bat", "bowl"), ("Batters", "Bowlers")):
        s = pd.read_csv(OUT / f"robustness_summary_{side}.csv").iloc[1:]
        y = np.arange(len(s))
        ax.barh(y, s.top10_overlap, color=[GRAY if ("Conventional" in n or "raw" in n or "conventional" in n) else BLUE for n in s.specification], height=0.6)
        for yi, (o, r) in enumerate(zip(s.top10_overlap, s.spearman_vs_baseline)):
            ax.text(o + 0.15, yi, f"{o}/10  ρ={r:.2f}" if not np.isnan(r) else f"{o}/10", va="center", fontsize=6.8)
        ax.set_xlim(0, 13.5)
        ax.set_title(f"{ttl}: overlap with the final top 10")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(np.arange(len(s)), s.specification, fontsize=7.2)
    axes[0].invert_yaxis()
    fig.tight_layout()
    save(fig, "F14_robustness_summary")


def main():
    b = pd.read_parquet(PROC / "balls_final.parquet")
    u = b[universe(b)]
    f_where_pressure_lives(b)
    f_definition_test()
    f_wp_validation()
    f_behaviour()
    f_reliability()
    f_shrinkage(u[u.faced])
    for s in ("bat", "bowl"):
        f_forest(s)
        f_rank_stability(s)
    f_normal_vs_pressure()
    f_finisher()
    f_prob_top10()
    f_case(b, 31250, "Jos Buttler", "bat", "F15_case_buttler",
           "Case 1 - KKR v RR, Eden Gardens, 16 Apr 2024: Buttler's 107* in a 224 chase")
    f_case(b, 40735, "Josh Hazlewood", "bowl", "F16_case_hazlewood",
           "Case 2 - RCB v RR, Bengaluru, 24 Apr 2025: Hazlewood's 19th over")
    f_case(b, 20193, "Rinku Singh", "bat", "F17_case_rinku",
           "Case 3 - GT v KKR, Ahmedabad, 9 Apr 2023: Rinku Singh's five sixes")
    f_dl()
    f_exposure()
    f_robust_summary()
    print("figures written:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
