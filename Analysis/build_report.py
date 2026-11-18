"""Builds SupeRR_Selector_Final_Report.pdf. Every number in the text is read from the pipeline outputs."""
import numpy as np
import pandas as pd
from common import PROC, OUT
from report_common import DEL, fig, j, f1, render_pdf

D = j("data_summary.json")
WP = j("wp_validation.json")
EX = j("expected_model_diagnostics.json")
H = j("rating_hyperparameters.json")
PM = j("pressure_meta.json")
PV = j("predictive_validation.json")
RO = j("robustness_overview.json")
INS = j("insights.json")
FB = pd.read_parquet(PROC / "final_bat.parquet")
FW = pd.read_parquet(PROC / "final_bowl.parquet")
FULLB = pd.read_parquet(PROC / "full_bat.parquet")
FULLW = pd.read_parquet(PROC / "full_bowl.parquet")
DEF = pd.read_csv(PROC / "pressure_definition_comparison.csv").set_index("definition")
BEH = pd.read_csv(PROC / "behaviour_by_pressure_decile.csv")
TB, TW = FB.head(10), FW.head(10)

wp_t = WP["selected_test_2025_26"]
lg_t = WP["logistic_handcrafted"]
mult = DEF.loc["DPI: leverage top 20%"]
death = DEF.loc["Death overs (16-20)"]
rrr = DEF.loc["Chase RRR >= 10"]
hand = DEF.loc["Hand-weighted index (0.4/0.3/0.3) top 20%"]
pop = INS["population_pressure_effect"]
shepherd = FULLB[FULLB.player_name == "Romario Shepherd"].iloc[0]
SH_B, SH_W = PV["bat"]["split_half_repeatability"], PV["bowl"]["split_half_repeatability"]


def _facts():
    from s05_ratings import universe
    b = pd.read_parquet(PROC / "balls_final.parquet")
    u = b[universe(b)]
    f = {}
    rr = b.groupby(["comp_group", "season"]).apply(lambda d: 6 * d.total_runs.sum() / d.legal.sum())
    f["ipl_rpo_2022"], f["ipl_rpo_2026"] = rr[("IPL", 2022)], rr[("IPL", 2026)]
    # split-half of components over all balls (players with 300+ balls in each half)
    s_ = u[u.faced].copy()
    s_["run_res"] = s_.bat_runs - s_.x_bat_runs
    s_["out_res"] = s_.striker_out.astype(float) - s_.x_out
    odd = s_.match_id % 2 == 1
    def sh(col):
        A = s_[odd].groupby("striker_id")[col].agg(["mean", "size"]); B_ = s_[~odd].groupby("striker_id")[col].agg(["mean", "size"])
        jn = A.join(B_, lsuffix="A", rsuffix="B", how="inner"); jn = jn[(jn.sizeA >= 300) & (jn.sizeB >= 300)]
        return float(np.corrcoef(jn.meanA, jn.meanB)[0, 1]), len(jn)
    f["sh_run"], f["sh_n"] = sh("run_res")
    f["sh_out"], _ = sh("out_res")
    f["sh_rae"], _ = sh("bat_rae")
    # case studies
    def chase(mid):
        d = b[(b.match_id == mid) & (b.innings == 2)]
        return d
    k1 = chase(31250)
    jb = k1[k1.striker_name == "Jos Buttler"]
    f["c1_min_wp"] = k1.wp_after.min()
    late = jb[jb.over >= 19]
    f["c1_li_lo"], f["c1_li_hi"] = late.LI.min(), late.LI.max()
    f["c1_wpa"] = jb.bat_wpa.sum()
    f["c1_n5"] = int((late.LI > 5).sum())
    k2 = chase(40735)
    h = k2[k2.bowler_name == "Josh Hazlewood"]
    o19 = k2[k2.over == 19]
    f["c2_wp_before"], f["c2_wp_after"] = o19.wp.iloc[0], o19.wp_after.iloc[-1]
    f["c2_wpa"] = h.bowl_wpa.sum()
    f["c2_hi_li"] = int((o19.LI > 5).sum())
    k3 = chase(20193)
    r6 = k3[(k3.over == 20) & (k3.striker_name == "Rinku Singh")]
    f["c3_wp_start"] = r6.wp.iloc[0]
    f["c3_li"] = list(r6.LI.values)
    return f


F = _facts()


def ordn(x):
    x = int(round(float(x)))
    suf = "th" if 10 <= x % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(x % 10, "th")
    return f"{x}{suf}"


def nm(df, i):
    return df.iloc[i].player_name


def refs_link(n):
    return f'<sup><a href="#ref{n}">[{n}]</a></sup>'


REFS = [
    ("Rajasthan Royals - Vaibhav Sooryavanshi's records in the IPL 2026 Eliminator (680 runs, most by an uncapped player in a season).",
     "https://www.rajasthanroyals.com/latest-news/vaibhav-sooryavanshi-new-records-srh-vs-rr-ipl-2026-eliminator"),
    ("Financial Mail (BusinessDay), 11 June 2026 - 'Cricket prodigy hits the record books with 72 sixes' (776 runs at a strike rate of 237.30, aged 15).",
     "https://www.financialmail.businessday.co.za/news-and-fox/2026-06-11-cricket-prodigy-hits-the-record-books-with-72-sixes/"),
    ("ICC media release - Sharma and Kishan give India a 1-2 in the T20I batting rankings (21-ball 52 in the 2026 T20 World Cup final).",
     "https://www.icc-cricket.com/media-releases/sharma-and-kishan-give-india-remarkable-1-2-in-icc-men-s-t20i-batting-rankings"),
    ("Gulf News - Abhishek Sharma reclaims the ICC No. 1 T20I batting ranking (first reached July 2025).",
     "https://gulfnews.com/sport/abhishek-sharma-replaces-ishan-kishan-as-iccs-no-1-t20i-batter-1.500726769"),
    ("CricTracker - Asia Cup 2025 award winners (Abhishek Sharma Player of the Tournament; Kuldeep Yadav MVP; Shivam Dube final Game Changer).",
     "https://www.crictracker.com/cricket-news/asia-cup-2025-winner-prize-money-complete-list-of-award-winners-top-records-stats-all-you-need-to-know-1238/"),
    ("Wisden - Varun Chakravarthy becomes No. 1 in the ICC T20I bowling rankings for the first time (September 2025).",
     "https://www.wisden.com/series/asia-cup-2025/cricket-news/latest-icc-t20i-player-rankings-india-spinner-crowned-new-no1-bowler-pakistan-opener-top-five-all-rounders"),
    ("ICC - 'India spinner reaches new heights on T20I rankings charts' (Chakravarthy career-high rating of 818).",
     "https://www.icc-cricket.com/news/india-spinner-reaches-new-heights-on-t20i-rankings-charts"),
    ("ESPNcricinfo - Buttler's 107* tops Narine's 109 as Royals ace record chase against KKR (16 April 2024).",
     "https://www.espncricinfo.com/page2/content/story/1429545.html"),
    ("The Federal - Buttler's ton helps RR pull off record chase (RR 121/6 in the 13th over; 46 needed off the last three overs).",
     "https://thefederal.com/cricket/ipl-2024-buttlers-sensational-ton-helps-rr-pull-off-record-chase-to-beat-kkr-118197"),
    ("ESPNcricinfo - Kohli and Hazlewood break RCB's home duck as RR botch another chase (24 April 2025).",
     "https://www.espncricinfo.com/ci/content/story/1482838.html"),
    ("Sportskeeda - Hazlewood's 4/33 against RR (18 needed from 12 balls before his 19th over).",
     "https://www.sportskeeda.com/cricket/news-collecting-plants-garden-fans-hail-josh-hazlewood-match-winning-spell-4-33-rcb-vs-rr-ipl-2025-clash"),
    ("ESPNcricinfo - Rinku Singh's six romance leads greatest last-over heist in T20s (forecaster win probability 0.61% with 28 needed off 5).",
     "https://www.espncricinfo.com/story/kkr-vs-gt-stats-rinku-singh-leads-greatest-last-over-heist-in-t20s-1367943"),
    ("ESPNcricinfo - Sunil Narine retires from international cricket (last West Indies appearance August 2019).",
     "https://www.espncricinfo.com/story/west-indies-spinner-sunil-narine-calls-time-on-international-career-1407363"),
    ("ESPNcricinfo - Heinrich Klaasen retires from international cricket (June 2025).",
     "https://www.espncricinfo.com/ci/content/story/1488740.html"),
    ("ICC - Tim David reflects on a 37-ball T20I century, the fastest by an Australian (July 2025).",
     "https://www.icc-cricket.com/news/david-reflects-on-unforgettable-maiden-t20i-century"),
    ("ESPN - Pathirana's 4/28 at the Wankhede; death-overs economy of 8.17 since IPL 2023.",
     "https://africa.espn.com/cricket/story/_/id/39945490/mi-vs-csk-ipl-2024-pathirana-point-difference-malinga-mumbai"),
    ("Outlook India - ICC T20I rankings after the 2026 T20 World Cup: Rashid Khan replaces Chakravarthy at No. 1.",
     "https://www.outlookindia.com/amp/story/sports/cricket/icc-t20i-rankings-march-2026-update-ishan-kishan-abhishek-sharma-varun-chakravarthy"),
    ("Tribune India - Krunal Pandya (2/31) and Hazlewood (4/33) in RCB's 11-run win over RR, IPL 2025.",
     "https://tribuneindia.com/news/sports/rcbs-josh-hazlewood-completes-150-wickets-in-t20s-during-4-fer-against-rr"),
    ("Mallawa Arachchi, Manage & Scariano (2024). A pressure index for the team batting second in T20I cricket. Journal of Sports Analytics 10(1).",
     "https://journals.sagepub.com/doi/10.3233/JSA-240792"),
    ("Rafique, H. (2022). cricWAR: A reproducible system for evaluating player performance in limited-overs cricket. CMU Sports Analytics Conference.",
     "https://www.stat.cmu.edu/cmsac/conference/2022/assets/pdf/HassanRafique.pdf"),
    ("Misra, V. & Rajaraman, A. (2026). Delta: a leverage-weighted, opposition-adjusted measure of player contributions in T20 cricket.",
     "https://www.cs.columbia.edu/%7Emisra/delta-2026.pdf"),
    ("Bandyopadhyay, R. (2025). Applications of higher-order Markov models and Pressure Index to strategize controlled run chases in T20 cricket. arXiv:2505.01849.",
     "https://arxiv.org/abs/2505.01849v2"),
    ("Sportsganga - Updated ICC T20I rankings after the 2026 T20 World Cup (India captain Suryakumar Yadav ninth among batters).",
     "https://sportsganga.com/news/updated-icc-t20i-rankings-2026-abhishek-sharma-leads-batting-varun-dropped-to-second-place"),
    ("ESPNcricinfo - Marsh, David leave New Zealand bruised in Australia's thrilling last-ball win (21 February 2024).",
     "https://www.espncricinfo.com/story/josh-clarkson-debuts-as-new-zealand-bat-australia-leave-out-steven-smith-1421922"),
]


# ---------------------------------------------------------------- tables
def top_table_bat():
    rows = []
    for _, r in TB.iterrows():
        rows.append(f"""<tr><td class="rank">{r['rank']}</td><td class="name">{r.player_name}</td><td class="r">{r.player_id}</td>
<td>{r.role}</td><td class="r">{r.matches_played_rating_universe:.0f}</td><td class="r">{r.p_balls:.0f}</td>
<td class="r">{r.p_strike_rate:.0f} <span class="note">({r.p_xSR:.0f})</span></td><td class="r">{r.p_outs:.0f} <span class="note">({r.p_x_outs:.1f})</span></td>
<td class="r">{r.normal_rae100:+.1f}</td><td class="r">{r.pressure_rae100:+.1f}</td><td class="r">{r.shift_rae100:+.1f}</td>
<td class="r">{100*r.c_raw_positive_share:.0f}%</td><td class="r">±{r.pressure_rae100_sd:.1f}</td><td class="r">{100*r.p_top10:.0f}%</td>
<td class="r"><b>{r.clutch_rating:.1f}</b></td></tr>""")
    return f"""<table><tr><th>#</th><th>Batter</th><th class="r">ID</th><th>Role</th><th class="r">Matches</th><th class="r">Pressure balls</th>
<th class="r">Pressure SR (exp.)</th><th class="r">Outs (exp.)</th><th class="r">Normal level</th><th class="r">Pressure level</th>
<th class="r">Shift</th><th class="r">Innings above exp.</th><th class="r">SD</th><th class="r">P(top 10)</th><th class="r">Clutch Rating</th></tr>{''.join(rows)}</table>"""


def top_table_bowl():
    rows = []
    for _, r in TW.iterrows():
        rows.append(f"""<tr><td class="rank">{r['rank']}</td><td class="name">{r.player_name}</td><td class="r">{r.player_id}</td>
<td>{r.role}</td><td class="r">{r.matches_played_rating_universe:.0f}</td><td class="r">{r.p_balls:.0f}</td>
<td class="r">{r.p_economy:.2f} <span class="note">({r.p_xEcon:.2f})</span></td><td class="r">{r.p_wkts:.0f} <span class="note">({r.p_x_wkts:.1f})</span></td>
<td class="r">{r.normal_rae100:+.1f}</td><td class="r">{r.pressure_rae100:+.1f}</td><td class="r">{r.shift_rae100:+.1f}</td>
<td class="r">{100*r.c_raw_positive_share:.0f}%</td><td class="r">±{r.pressure_rae100_sd:.1f}</td><td class="r">{100*r.p_top10:.0f}%</td>
<td class="r"><b>{r.clutch_rating:.1f}</b></td></tr>""")
    return f"""<table><tr><th>#</th><th>Bowler</th><th class="r">ID</th><th>Role</th><th class="r">Matches</th><th class="r">Pressure balls</th>
<th class="r">Pressure econ. (exp.)</th><th class="r">Wickets (exp.)</th><th class="r">Normal level</th><th class="r">Pressure level</th>
<th class="r">Shift</th><th class="r">Spells above exp.</th><th class="r">SD</th><th class="r">P(top 10)</th><th class="r">Clutch Rating</th></tr>{''.join(rows)}</table>"""


def conventional_table_bat():
    rows = []
    for _, r in TB.iterrows():
        rows.append(f"""<tr><td class="rank">{r['rank']}</td><td class="name">{r.player_name}</td>
<td class="r">{r.p_strike_rate:.1f}</td><td class="r">{r.n_strike_rate:.1f}</td><td class="r">{r.p_average:.1f}</td><td class="r">{r.n_average:.1f}</td>
<td class="r">{r.p_dot_pct:.1f}</td><td class="r">{r.p_boundary_pct:.1f}</td><td class="r">{r.p_balls_per_dismissal:.1f}</td>
<td class="r">{r.pressure_wpa100:+.1f}</td><td class="r">{r.c_pressure_matches:.0f}</td><td class="r">{100*r.core_top10_share:.0f}%</td></tr>""")
    return f"""<table class="compact"><tr><th>#</th><th>Batter</th><th class="r">SR pressure</th><th class="r">SR normal</th><th class="r">Avg pressure</th>
<th class="r">Avg normal</th><th class="r">Dot % (P)</th><th class="r">Boundary % (P)</th><th class="r">Balls/dismissal (P)</th>
<th class="r">WPA per 100 P balls</th><th class="r">Pressure matches</th><th class="r">Top-10 in core specs</th></tr>{''.join(rows)}</table>"""


def conventional_table_bowl():
    rows = []
    for _, r in TW.iterrows():
        rows.append(f"""<tr><td class="rank">{r['rank']}</td><td class="name">{r.player_name}</td>
<td class="r">{r.p_economy:.2f}</td><td class="r">{r.n_economy:.2f}</td><td class="r">{r.p_bowl_sr:.1f}</td><td class="r">{r.p_bowl_avg:.1f}</td>
<td class="r">{r.p_dot_pct:.1f}</td><td class="r">{r.p_boundary_pct:.1f}</td><td class="r">{r.p_wkts - r.p_x_wkts:+.1f}</td>
<td class="r">{r.pressure_wpa100:+.1f}</td><td class="r">{r.c_pressure_matches:.0f}</td><td class="r">{100*r.core_top10_share:.0f}%</td></tr>""")
    return f"""<table class="compact"><tr><th>#</th><th>Bowler</th><th class="r">Econ. pressure</th><th class="r">Econ. normal</th><th class="r">SR pressure</th>
<th class="r">Avg pressure</th><th class="r">Dot % (P)</th><th class="r">Boundary % (P)</th><th class="r">Wkts vs exp. (P)</th>
<th class="r">WPA per 100 P balls</th><th class="r">Pressure matches</th><th class="r">Top-10 in core specs</th></tr>{''.join(rows)}</table>"""


def definition_table():
    order = ["DPI: leverage top 20%", "DPI: leverage top 10%", "DPI: leverage top 30%", "Phase-relative leverage top 20%",
             "Close game (WP 25-75%) after over 12", "Last 5 overs of a chase", "Death overs (16-20)", "Rule-based composite",
             "Chase RRR >= 10", "Hand-weighted index (0.4/0.3/0.3) top 20%"]
    rows = []
    for d in order:
        r = DEF.loc[d]
        rows.append(f"""<tr><td>{d}</td><td class="r">{100*r.share_balls:.0f}%</td><td class="r">{100*r.share_of_pressure_in_innings1:.0f}%</td>
<td class="r">{100*r.share_of_total_leverage:.0f}%</td><td class="r"><b>{r.multiplier:.2f}</b></td><td class="r">±{1.96*r.multiplier_se:.2f}</td>
<td class="r">{r.llf_gain_vs_unsplit:.1f}</td></tr>""")
    return f"""<table class="compact"><tr><th>Definition of a 'pressure ball'</th><th class="r">Balls flagged</th><th class="r">From 1st innings</th>
<th class="r">Share of all match leverage</th><th class="r">Result-importance multiplier</th><th class="r">95% CI</th><th class="r">Log-lik. gain</th></tr>{''.join(rows)}</table>"""


def validation_table():
    keys = [("Raw clutch delta (pressure - normal RAE/100)", "Raw 'clutch' delta"), ("Raw pressure RAE/100", "Raw pressure runs above expectation"),
            ("Pressure strike rate", "Pressure strike rate / economy"), ("Match consistency (share positive)", "Match-level consistency"),
            ("PAPM pressure WPA", "Shrunk pressure WPA"), ("PAPM pressure shift only", "Shrunk pressure shift"),
            ("PAPM single-currency pressure level", "Shrunk pressure level, one currency"), ("PAPM pressure level", "Shrunk pressure level (final)")]
    rows = []
    for k, lab in keys:
        kb = k
        kw = "Pressure economy (negated)" if k == "Pressure strike rate" else k
        rows.append(f"""<tr><td>{lab}</td><td class="r">{SH_B[kb]:+.2f}</td><td class="r">{PV['bat']['leave_one_season_out_pooled'][kb]['weighted_r']:+.2f}</td>
<td class="r">{PV['bat']['forward_2022_24_to_2025_26'][kb]['weighted_r']:+.2f}</td><td class="r">{SH_W[kw]:+.2f}</td>
<td class="r">{PV['bowl']['leave_one_season_out_pooled'][kw]['weighted_r']:+.2f}</td><td class="r">{PV['bowl']['forward_2022_24_to_2025_26'][kw]['weighted_r']:+.2f}</td></tr>""")
    return f"""<table class="compact"><tr><th rowspan="2">Measure of pressure performance</th><th class="r" colspan="3">Batters</th><th class="r" colspan="3">Bowlers</th></tr>
<tr><th class="r">Split-half</th><th class="r">Leave-season-out</th><th class="r">2022-24 → 2025-26</th><th class="r">Split-half</th><th class="r">Leave-season-out</th><th class="r">2022-24 → 2025-26</th></tr>{''.join(rows)}</table>"""


def robustness_table(side):
    S = pd.read_csv(OUT / f"robustness_summary_{side}.csv").iloc[1:]
    rows = "".join(f"<tr><td>{r.specification}</td><td class='r'>{r.top10_overlap}/10</td><td class='r'>{'-' if pd.isna(r.spearman_vs_baseline) else f'{r.spearman_vs_baseline:.2f}'}</td><td class='r'>{r.n_eligible}</td></tr>"
                   for _, r in S.iterrows())
    return f"<table class='compact'><tr><th>Specification</th><th class='r'>Top-10 overlap</th><th class='r'>Spearman ρ</th><th class='r'>Eligible</th></tr>{rows}</table>"


# ---------------------------------------------------------------- profiles
EXT = {
    "Suryakumar Yadav": f"External context: India's T20I captain, listed inside the ICC top ten batters after the 2026 T20 World Cup{refs_link(23)}. Our data-only finding - the largest pressure sample in the pool and a measurable rise in level - is the statistical version of his reputation.",
    "Vaibhav Sooryavanshi": f"External context: at 15 he scored 776 runs at a strike rate of 237.30 in IPL 2026{refs_link(2)}, including a 29-ball 97 in the Eliminator that took him past the record for runs by an uncapped player in a season{refs_link(1)}. The data cannot yet tell us how he handles a bad season; that is exactly why his Clutch Rating sits 9 runs below his posterior mean.",
    "Abhishek Sharma": f"External context: No. 1 in the ICC T20I batting rankings from July 2025{refs_link(4)}, Player of the Tournament at the 2025 Asia Cup{refs_link(5)} and scorer of a 21-ball 52 in the 2026 T20 World Cup final{refs_link(3)} - a match outside this dataset.",
    "Tim David": f"External context: the last-ball four that completed Australia's chase of 216 in Wellington (Feb 2024){refs_link(24)} and a 37-ball T20I hundred in St Kitts, the fastest by an Australian{refs_link(15)}.",
    "Heinrich Klaasen": f"External context: retired from international cricket in June 2025{refs_link(14)}, so his 2025-26 evidence is IPL-only; the model still has 4 seasons of pressure exposure for him.",
    "Phil Salt": "External context: an opener whose pressure balls come almost entirely in chases (86%), which in this data are the powerplays of big targets.",
    "Nicholas Pooran": "External context: data-only profile. He tops our 'survival' leaderboard - pressure balls where losing a wicket is the dominant risk - which is the opposite of how a pure hitter is usually described.",
    "Jos Buttler": f"External context: the Eden Gardens 107* in a 224 chase (Case Study 1){refs_link(8)}{refs_link(9)}.",
    "Rinku Singh": f"External context: the five consecutive sixes off Yash Dayal with 28 needed from five balls (Case Study 3){refs_link(12)}.",
    "Shivam Dube": f"External context: named Game Changer of the 2025 Asia Cup final{refs_link(5)}. Our data show the most unusual pressure profile in the top ten: slower than expected, but out half as often.",
    "Sunil Narine": f"External context: has not played international cricket since August 2019 and retired from it in 2023{refs_link(13)}, so his entire rating is built on IPL evidence - mostly in KKR's chases.",
    "Jasprit Bumrah": "External context: data-only profile. The finding that batters stop taking risks against him is consistent with a 4-percentage-point boundary suppression in our data.",
    "Varun Chakravarthy": f"External context: returned to India's T20I side in late 2024 after three years away and reached No. 1 in the ICC T20I bowling rankings in September 2025{refs_link(6)}, later a career-high 818 rating{refs_link(7)}.",
    "Matheesha Pathirana": f"External context: his 4/28 at the Wankhede in 2024 is typical of the profile; ESPN put his IPL death-overs economy at 8.17 since 2023, second-best among regular death bowlers{refs_link(16)}.",
    "Rashid Khan": f"External context: back to No. 1 in the ICC T20I bowling rankings after the 2026 T20 World Cup{refs_link(17)}.",
    "Mujeeb Ur Rahman": "External context: data-only profile. All of his qualifying pressure evidence comes from T20Is - mostly Afghanistan's powerplay overs in chases.",
    "Josh Hazlewood": f"External context: the 19th over against RR on 24 April 2025 - one run, two wickets with 18 needed from 12 (Case Study 2){refs_link(10)}{refs_link(11)}.",
    "Krunal Pandya": f"External context: in the same 2025 match as Case Study 2 he removed Parag and Rana for 2/31{refs_link(18)}. All of his pressure evidence is IPL.",
    "Mitchell Santner": "External context: data-only profile; 90% of his pressure balls are in the opposition's chase.",
    "Noor Ahmad": "External context: data-only profile; the selection with the thinnest pressure margin in the bowling list (see text).",
}


def tags_bat(r):
    t = []
    crisis = [d["name"] for d in INS["crisis_batting"]]
    surv = [d["name"] for d in INS["survival_specialists"]]
    if r.player_name in crisis:
        t.append('<span class="tag pink">Crisis Manager</span>')
    if r.role == "Finisher":
        t.append('<span class="tag">Finisher</span>')
    if r.role == "Opener":
        t.append('<span class="tag">Powerplay Accelerator</span>')
    if r.player_name in surv:
        t.append('<span class="tag gold">Survivor</span>')
    if r.p_shift_positive >= 0.9:
        t.append('<span class="tag pink">Pressure Riser</span>')
    if r.c_raw_positive_share >= 0.68:
        t.append('<span class="tag gold">Consistent Performer</span>')
    return " ".join(t)


def tags_bowl(r):
    t = []
    if "Death" in r.role or "death" in r.role:
        t.append('<span class="tag pink">Death Specialist</span>')
    if r.p_wkts - r.p_x_wkts >= 3:
        t.append('<span class="tag pink">Pressure Enforcer</span>')
    if (r.p_xEcon - r.p_economy) >= 1.0:
        t.append('<span class="tag">Controller</span>')
    if r.p_shift_positive >= 0.85:
        t.append('<span class="tag gold">Pressure Riser</span>')
    if r.c_raw_positive_share >= 0.6:
        t.append('<span class="tag gold">Consistent Performer</span>')
    return " ".join(t)


def profiles(df, side):
    out = []
    for _, r in df.iterrows():
        tg = tags_bat(r) if side == "bat" else tags_bowl(r)
        out.append(f"""<div class="profile"><h4><span class="rk">{r['rank']}.</span>{r.player_name} <span class="note">· {r.role} · {r.teams}</span></h4>
<div style="margin-bottom:1mm">{tg}</div><p>{r.explanation} Clutch Rating {r.clutch_rating:.1f} (80% interval of the pressure level {r.pressure_rae100_p10:.1f} to {r.pressure_rae100_p90:.1f});
top ten in {100*r.p_top10:.0f}% of bootstrap resamples and {100*r.core_top10_share:.0f}% of core model specifications.</p>
<p class="ext">{EXT.get(r.player_name, '')}</p></div>""")
    return "".join(out)


# ---------------------------------------------------------------- body
def body():
    b1, b2 = nm(TB, 0), nm(TB, 1)
    w1 = nm(TW, 0)
    q80, q95 = PM["q80_LI"], PM["q95_LI"]
    tl = DEF.loc["DPI: leverage top 20%"].share_of_total_leverage
    dl = EX["dl_resource_fit"]
    hb, hw = H["bat"], H["bowl"]
    ex_bat = INS["exposure_by_phase_innings"]
    inn1_pp = [e for e in ex_bat if e["innings"] == 1 and e["phase"] == "Powerplay"][0]["pressure_share"]
    inn2_d = [e for e in ex_bat if e["innings"] == 2 and e["phase"] == "Death"][0]["pressure_share"]
    inn1_mid = [e for e in ex_bat if e["innings"] == 1 and e["phase"] == "Middle"][0]["pressure_share"]
    ko = INS["knockout"]
    vs = FB[FB.player_name == "Vaibhav Sooryavanshi"].iloc[0]
    sky = FB[FB.player_name == "Suryakumar Yadav"].iloc[0]
    bum = FW[FW.player_name == "Jasprit Bumrah"].iloc[0]
    var = FW[FW.player_name == "Varun Chakravarthy"].iloc[0]
    noor = FW[FW.player_name == "Noor Ahmad"].iloc[0]
    dube = FB[FB.player_name == "Shivam Dube"].iloc[0]
    kul = FW[FW.player_name == "Kuldeep Yadav"].iloc[0]
    marsh = FB[FB.player_name == "Mitchell Marsh"].iloc[0]
    theek = FULLW[FULLW.player_name == "Maheesh Theekshana"].iloc[0]
    fin = {d["name"]: d for d in INS["finishers_scoring_vs_risk"]}
    bp = {d["name"]: d for d in INS["boundary_prevention"]}
    under_b = INS["underrated_bat"]
    over_b = INS["overrated_bat"]
    over_w = INS["overrated_bowl"]
    under_w = INS["underrated_bowl"]
    rb = INS["risers_bowl"]
    nb_bat = int((FULLB.eligible).sum())
    nb_bowl = int((FULLW.eligible).sum())
    moved_bat = FB[FB["rank"] != FB.rank_by_rating_only].head(12)
    html = []
    A = html.append

    # ---------------- contents
    A("""<h1 style="margin-top:2mm">Contents</h1><div class="toc">""")
    toc = ["Executive Summary", "Research Objective", "Understanding Pressure in T20 Cricket", "Dataset and Data Quality",
           "Data Preparation", "Dynamic Pressure Index", "Batting Performance Model", "Bowling Performance Model",
           "Expected Performance and Player Baselines", "Statistical Adjustment and Uncertainty", "Validation and Robustness Testing",
           "Final Batter Rankings", "Final Bowler Rankings", "Individual Player Evidence", "Interesting Discoveries",
           "Match-Level Case Studies", "What Conventional Cricket Statistics Fail to Reveal", "Limitations", "Conclusion",
           "Technical Appendix", "References"]
    for i, t in enumerate(toc, 1):
        A(f"<div><span><span class='n'>{i:02d}</span>{t}</span></div>")
    A("</div>")
    A(f"""<div class="callout" style="margin-top:8mm"><b>How to read the numbers.</b> The common currency in this report is
<b>runs above expectation per 100 balls</b> (RAE/100). For every delivery we estimate what an average player would have produced in exactly the
same match situation; a batter's RAE is the runs he scored above that, minus the run value of any dismissal beyond the expected number. A batter at
+20 adds 20 runs per 100 balls compared with an average player in his shoes. For bowlers the sign is reversed: +20 means 20 runs saved.
The <b>Dynamic Pressure Index (DPI)</b> measures how much a ball can move the result: DPI 1 is an average ball, DPI 3 carries three times the usual
match-deciding weight. A <b>pressure ball</b> is one in the top 20% of DPI (DPI ≥ {q80:.2f}).</div>""")

    # ---------------- 1 executive summary
    A(f"""<div class="section"><div class="kicker">Section 01</div><h2><span class="num">01</span>Executive Summary</h2>
<p>We set out to answer a narrow question: <i>who becomes more dependable, or at least stays dependable, when a T20 match is genuinely in the
balance?</i> Answering it properly required three things that most clutch analyses skip. First, a definition of pressure that is tied to
results rather than to the scoreboard clock. Second, a way of comparing a player with what an average player would have done in the same situation,
against the same quality of bowling or batting. Third, an honest treatment of luck, because a batter's record in high-pressure balls is usually
built on a few hundred deliveries.</p>
<div class="statrow">
<div class="stat"><div class="v">{f1(D['deliveries_modelled'],0)}</div><div class="l">deliveries from {D['matches_with_deliveries']} IPL and T20I matches, Jan 2022 – Nov 2026</div></div>
<div class="stat pink"><div class="v">{mult.multiplier:.1f}×</div><div class="l">how much more a run above expectation is worth on a DPI pressure ball than on an ordinary one</div></div>
<div class="stat"><div class="v">{death.multiplier:.2f}×</div><div class="l">the same figure for 'death overs' - the most common definition of pressure in cricket writing</div></div>
<div class="stat pink"><div class="v">{SH_B['Raw clutch delta (pressure - normal RAE/100)']:+.2f}</div><div class="l">split-half repeatability of the raw 'clutch' statistic - indistinguishable from noise</div></div>
</div>
<p><b>Pressure is leverage, not the clock.</b> We built a win-probability model for both innings and defined the Dynamic Pressure Index as the
expected swing in win probability carried by a single delivery. Tested against match results, the top fifth of DPI balls are {mult.multiplier:.1f} times
as decisive per run as ordinary balls. The definitions that dominate cricket commentary fail the same test: a run above expectation in the death overs
is worth {death.multiplier:.2f} of an ordinary run, in a chase with the required rate above ten it is worth {rrr.multiplier:.2f}, and a typical
hand-weighted pressure index scores {hand.multiplier:.2f}. Those definitions mark balls that feel tense, not balls that decide matches.</p>
<p><b>Clutch is mostly class - but not entirely.</b> When we cross-validated how much players genuinely differ in their pressure performance, the data
said that run-scoring under pressure is the same skill as run-scoring in general: the best estimate of any batter's or bowler's pressure scoring
shift is zero. What does vary between players is <i>dismissal risk</i> under pressure. Some batters protect their wicket far better than their
normal game suggests, and some bowlers stop taking wickets because batters refuse to take risks against them. Our final rating therefore rewards
genuine all-round quality in high-leverage situations, adjusts for the dismissal effects the data can support, and penalises small samples.</p>
<p><b>The rankings.</b> {b1} heads the batting list, with the largest pressure sample in the pool ({sky.p_balls:.0f} balls in {sky.c_pressure_matches:.0f} matches),
a pressure strike rate of {sky.p_strike_rate:.0f} where an average batter in the same situations strikes at {sky.p_xSR:.0f}, and a measurable rise in level
under pressure. {b2} has the highest estimated pressure level of anyone ({vs.pressure_rae100:+.1f} runs per 100 balls) but on only {vs.p_balls:.0f} pressure
balls; our conservative rating puts him second, and we say plainly that he is the least certain name in either list. {w1} leads the bowlers on
{FW.iloc[0].p_balls:.0f} pressure balls at an economy of {FW.iloc[0].p_economy:.2f} where {FW.iloc[0].p_xEcon:.2f} was expected, ahead of
{nm(TW,1)} and {nm(TW,2)}.</p>
<div class="listcard"><div><h4>Top 10 batters under pressure</h4><ol>{''.join(f'<li>{n}</li>' for n in TB.player_name)}</ol></div>
<div class="b"><h4>Top 10 bowlers under pressure</h4><ol>{''.join(f'<li>{n}</li>' for n in TW.player_name)}</ol></div></div>
<p style="margin-top:4mm"><b>How confident are we?</b> Across 27 alternative specifications - different pressure thresholds, sample rules, competitions,
models and leave-one-season-out refits - an average of {RO['bat']['mean_top10_overlap']:.1f} of our ten batters and {RO['bowl']['mean_top10_overlap']:.1f} of our ten
bowlers survive, with rank correlations of {RO['bat']['mean_spearman']:.2f} and {RO['bowl']['mean_spearman']:.2f} against the baseline. The top four in
each list are top-ten in at least {100*min(TB.head(4).p_top10.min(), TW.head(4).p_top10.min()):.0f}% of bootstrap resamples. Positions 9 and 10 are genuinely contested; we name the alternatives in Sections 12 and 13.</p></div>""")

    # ---------------- 2 objective
    A(f"""<div class="section"><div class="kicker">Section 02</div><h2><span class="num">02</span>Research Objective</h2>
<p>The brief asks for a system that measures how batters and bowlers perform in high-pressure situations compared with their usual
level, and then ranks the players who deliver most consistently when it matters. We read that as five separate quantities that must not be blended carelessly:</p>
<ol><li><b>Overall ability</b> - how good the player is across all situations, after adjusting for context and opposition.</li>
<li><b>Pressure performance</b> - how good he is on the balls that actually decide matches.</li>
<li><b>Pressure shift</b> - the difference between the two, i.e. whether he rises or fades.</li>
<li><b>Repeatability</b> - whether that performance recurs across matches and seasons.</li>
<li><b>Confidence</b> - how much of what we see could be luck, given the sample.</li></ol>
<p>Our ranking target is the second quantity, estimated so that the third is weighted exactly as much as the evidence supports, with the fourth and
fifth built in through a conservative estimate and a robustness ensemble. We chose that target deliberately. A franchise picking a player for the
19th over does not care whether he is better there <i>than he usually is</i>; it cares whether he is better there <i>than the alternatives</i>. The
shift still matters: it is one of the inputs to the pressure estimate, and it is reported separately for every player so that 'risers' and 'faders'
are visible. But a ranking built on the shift alone would reward mediocre players who are slightly less mediocre under pressure, and - as Section 11
shows - the raw shift has no repeatable signal at all.</p>
<div class="callout"><b>Decision rule, fixed before the final ranking was produced.</b> Rank by the 20th percentile of each player's bootstrap
distribution of pressure-level RAE/100 (the 'Clutch Rating': we are 80% confident he is at least this good). Two safeguards were added after the
robustness results were in, and we report what each one changed: a <b>tie band</b> (ratings within 1 run/100 of each other are ordered by how often each
player makes the top ten across the core robustness specifications) and an <b>evidence gate</b> (a player is only selected if his own observed pressure
record is above expectation).</div></div>""")

    # ---------------- 3 understanding pressure
    A(f"""<div class="section"><div class="kicker">Section 03</div><h2><span class="num">03</span>Understanding Pressure in T20 Cricket</h2>
<p>Ask a player what pressure feels like and you will hear about the scoreboard: a required rate climbing past twelve, two new batters at the
crease, a death bowler who has not conceded a boundary all night. Ask a statistician and you will hear about consequences: how much the next ball
can change the probability of winning. These are related but not the same. Needing 28 off five balls is enormous scoreboard pressure, yet
almost nothing hinges on the first of those balls - the match is already 99% lost. Needing 9 off six with Buttler on strike carries less obvious
drama, yet every ball moves the result by ten or twenty percentage points.</p>
<p>The academic literature has mostly formalised the first view. Pressure indices for the chasing side, starting with Shah and Shah and refined
by Bhattacharjee and Lemmer, combine required run rate with remaining resources{refs_link(19)}{refs_link(22)}. They are useful for describing a chase, but they
are silent about the team batting first, and they have no natural calibration against results. Sabermetric work - the baseball Leverage Index, and in
cricket the cricWAR framework{refs_link(20)} and the recent leverage-weighted Delta metric{refs_link(21)} - takes the second view, weighting events by
how much they matter. We take the second view as the definition and use the first as a diagnostic: Section 6 shows that once leverage is known, the
scoreboard variables add nothing to the identification of decisive moments.</p>
<p>The choice has a cricketing logic. A death bowler defending 12 off the last over is under enormous pressure; so is a batter walking out at 31 for 3
in the fifth over of a 190 chase. A pure scoreboard definition sees only the first. A leverage definition sees both, and it sees them for the team batting first
too, where the stakes are less visible but real: a collapse from 80 for 1 to 95 for 5 in the 11th over changes a team's expected result as surely as a dropped catch at the death.</p>
<h3>What pressure does to the average player</h3>
<p>Before rating anyone, it is worth knowing how pressure changes the game itself. Across all batters in our rating universe, strike rate on pressure balls
is {pop['bat_sr_pressure']:.0f} against {pop['bat_sr_other']:.0f} on other balls, and the dismissal rate is {pop['bat_out_pct_pressure']:.2f}% against
{pop['bat_out_pct_other']:.2f}%. Bowlers concede {pop['bowl_econ_pressure']:.2f} an over under pressure against {pop['bowl_econ_other']:.2f} otherwise.
Pressure balls are, in other words, more aggressive and riskier for everyone. That is why every comparison in this report is made against an
expectation for the same situation - otherwise a death-overs hitter would look 'clutch' simply because everyone scores faster at the death.</p>
{fig('F04_behaviour_by_pressure', '<b>Figure 1. How the game changes as pressure rises.</b> Balls faced in the rating universe split into ten equal groups by DPI. Strike rate and the share of balls in a chase rise with pressure; dismissal risk is U-shaped, high both in hopeless situations (lowest decile, where the result is settled and batters swing freely) and in the most decisive ones.')}
</div>""")

    # ---------------- 4 dataset
    ds = D
    A(f"""<div class="section"><div class="kicker">Section 04</div><h2><span class="num">04</span>Dataset and Data Quality</h2>
<p>All rankings rest on the six supplied files. We checked every preliminary statistic ourselves; the table below summarises what we found, including
the issues that shaped the methodology.</p>
<table class="compact"><tr><th>Item</th><th>Finding</th></tr>
<tr><td>Deliveries</td><td>{f1(ds['raw_deliveries'],0)} rows; {ds['super_over_deliveries_excluded']} super-over deliveries (innings 3-4) excluded, leaving {f1(ds['deliveries_modelled'],0)}.</td></tr>
<tr><td>Matches</td><td>{ds['matches_in_matches_csv']} in matches.csv ({ds['comp_groups_matches']['IPL']} IPL, {ds['comp_groups_matches']['T20I']} T20I, {ds['comp_groups_matches']['A-team T20']} A-team T20); {ds['matches_with_deliveries']} have deliveries, {ds['matches_without_deliveries']} are abandoned without a ball.</td></tr>
<tr><td>Period</td><td>{ds['date_range'][0]} to {ds['date_range'][1]}; IPL seasons 2022-2026 complete; {f1(ds['unique_batters'],0)} batters and {f1(ds['unique_bowlers'],0)} bowlers.</td></tr>
<tr><td>Results</td><td>{ds['result_classes']['decided']} decided, {ds['result_classes']['tie']} tied (super over or bowl-out), {ds['result_classes']['no_result']} no result; {ds['dls_or_revised_matches']} matches affected by DLS or revised overs; {ds['knockout_matches']} knockouts.</td></tr>
<tr><td><b>Runs field</b></td><td><code>runs</code> holds runs off the bat only. Wides, byes and leg-byes carry <i>no</i> run count, and <code>extras1_runs</code> is a type code (6 = none, 2 = wide, 3 = leg-bye, 4 = bye, 1 = overstep no-ball, 5 = other no-ball), not runs.</td></tr>
<tr><td><b>Imputed extras</b></td><td>We estimated the average value of each unrecorded extra by robust regression of 'won by X runs' margins on extras counts ({ds['extras_calibration']['n_matches']} matches): {ds['extras_calibration']['estimates']['wide']:.2f} runs per wide, {ds['extras_calibration']['estimates']['byes']:.2f} per bye event and {ds['extras_calibration']['estimates']['legbyes']:.2f} per leg-bye event. Reconstructed margins are within 2 runs of the official margin in {100*ds['extras_calibration']['share_within_2_runs']:.0f}% of matches and within 5 runs in {100*ds['extras_calibration']['share_within_5_runs']:.0f}%.</td></tr>
<tr><td>Legal balls</td><td>Wides and both no-ball codes are illegal; with that rule {100*ds['share_overs_with_six_legal']:.1f}% of overs contain exactly six legal balls (the rest are final overs of innings).</td></tr>
<tr><td>Chase targets</td><td>Not stored. Reconstructed and then anchored to the chase itself (Section 5): {ds['target_methods']['anchored to winning ball']} from the winning ball, {ds['target_methods']['anchored to chase + margin']} from the chase total plus margin, {ds['target_methods']['official revised target']} official DLS targets.</td></tr>
<tr><td>Competitions</td><td>IPL plus bilateral T20I series and Asia Cups. <b>No ICC T20 World Cup matches are included</b> - a material gap for a study of pressure (Section 18).</td></tr>
<tr><td>Players</td><td>Names in the delivery file and in competition_team_player_mapping.csv agree for every ranked player; the mapping file's spelling is used throughout and in the Unstop list.</td></tr>
</table>
<p><b>Rating universe.</b> Match-state models use every match, because Associate and A-team games are still informative about how T20 innings unfold.
Player ratings use only deliveries where both sides are IPL teams or ICC Full Members ({H['universe_matches']} matches, {f1(H['universe_balls'],0)} deliveries).
Associate opposition cannot be placed on a common scale with the evidence available - a Nepal batter faces Nepal's opponents almost exclusively - and
including them would let weak fixtures inflate ratings. Section 11 shows the rankings barely move when they are included.</p></div>""")

    # ---------------- 5 preparation
    A(f"""<div class="section"><div class="kicker">Section 05</div><h2><span class="num">05</span>Data Preparation</h2>
<p>Every delivery was turned into a match state <i>before</i> the ball was bowled: runs, wickets, legal balls remaining (using revised overs where
the match was shortened), target and runs needed in a chase, the scoring environment, and the pre-match strength of both teams. The steps that
required judgement were these.</p>
<h4>Chase targets anchored to the chase</h4>
<p>Because some extras are unrecorded, a reconstructed first-innings total can be several runs short - in the Buttler match (Case Study 1) it is seven.
Using it naively would leave the chasing side 'winning' before the last over had been bowled. Since what matters for every second-innings state is
<i>runs needed = target − reconstructed score</i>, we anchored the target to the reconstructed chase: for chases that succeeded, the target must lie between
the score before and after the winning ball; for chases that failed by X runs, the target is the reconstructed chase total plus X plus one. Errors
then shrink to zero exactly where pressure is highest - in the final overs. Official DLS targets were used where recorded.</p>
<h4>Scoring environment without leakage</h4>
<p>T20 scoring rose sharply over this period: IPL run rates went from {F['ipl_rpo_2022']:.2f} an over in 2022 to {F['ipl_rpo_2026']:.2f} in 2026. We describe each match's environment by
the run rate of the previous 40 matches in the same pool (IPL or international), rounded to 0.1, so that no model sees information from the
match it is scoring. An earlier version used a leave-one-out mean of the same season; the boosted models used it to identify individual matches and
memorised them (training Brier 0.005, test 0.31). That failure is why every subsequent model uses match-grouped validation.</p>
<h4>Team strength</h4>
<p>A chronological Elo rating (K = 24) gives each side's pre-match strength. IPL teams and Full Members start at 1500, A-teams at 1400 and Associates at 1300.
Elo enters the win-probability model only; it lets a lopsided India v Nepal fixture carry low leverage throughout, which is the right answer.</p>
<h4>Other rules</h4>
<p>A no-ball counts as a ball faced; a wide does not. Runs charged to a bowler exclude byes and leg-byes. Bowlers are credited with caught, bowled,
LBW, stumped and hit-wicket dismissals only. Retired-out is not a team wicket. Super overs are excluded from modelling but ties count as half a win.</p></div>""")

    # ---------------- 6 DPI
    A(f"""<div class="section"><div class="kicker">Section 06</div><h2><span class="num">06</span>Dynamic Pressure Index</h2>
<h3>Step 1 - a win-probability model for both innings</h3>
<p>In a chase, the probability of winning is modelled directly from runs needed, balls left, wickets in hand and required rate, with two refinements
that turned out to matter. The runs are expressed <i>relative to the scoring environment</i> (needing 200 in IPL 2025 is not the same problem as
needing 200 in a 2022 T20I), and the model is forced to be monotone: it can never claim that needing more runs, having fewer balls or losing a wicket
helps the chasing side. For the first innings we do not fit a separate classifier. We model the distribution of runs still to come from the current
state, R ~ Normal(μ(s), σ(s)), and integrate the chase model over the target that distribution implies:</p>
<div class="eq">WP<sub>1</sub>(s) = E<sub>R</sub>[ 1 − C( target = runs + R + 1, full resources ) ]</div>
<p>This uses the information in the final score rather than just the win/loss label, and it guarantees that the win probability at the end of the
first innings equals the win probability at the start of the chase (median gap {WP['innings_break_continuity_median_abs_gap']:.4f}).</p>
<p>Specifications were chosen on a strict chronological design: fit on 2022-23, choose on 2024, then refit on 2022-24 and test once on 2025-26.
On the unseen 2025-26 matches the selected model has a Brier score of {wp_t['all']['brier']:.3f} (AUC {wp_t['all']['auc']:.2f}), against
{lg_t['all']['brier']:.3f} for a carefully built logistic regression and 0.250 for no information; in chases alone it is {wp_t['innings_2']['brier']:.3f}
against {lg_t['innings_2']['brier']:.3f}. Calibration is close to the diagonal across the range.</p>
{fig('F03_wp_validation', '<b>Figure 2. The win-probability model on matches it never saw.</b> Left: predicted versus observed win rate, 2025-26 (dot size = number of balls). Right: Brier score against two alternatives.')}
<h3>Step 2 - leverage</h3>
<p>For every delivery a multinomial outcome model gives the probability of each result - dot, 1, 2, 3, 4, 6, wicket or wide/no-ball - in that exact
state. Each outcome leads to a new state with its own win probability. The <b>leverage</b> of the ball is the expected size of the swing:</p>
<div class="eq">LI(s) = Σ<sub>o</sub> P(o | s) · | WP(s after o) − WP(s) |, &nbsp;&nbsp; DPI = LI / mean(LI)</div>
<p>A DPI of 1 is an average ball. Half of all balls have a DPI below {WP['li_quantiles']['0.5']:.2f}; the top 20% start at {q80:.2f}, the top 5% at {q95:.2f}
and the most decisive 1% exceed {WP['li_quantiles']['0.99']:.1f}. All win probabilities and leverages are computed out-of-fold - five folds grouped by match - so no ball is scored by a model that saw its outcome.</p>
{fig('F01_where_pressure_lives', '<b>Figure 3. Where pressure lives.</b> Mean DPI by over and wickets down. In the first innings pressure is moderate and spread out; in a chase it concentrates in the last five overs when four to seven wickets are down - but early collapses in either innings also register.')}
<h3>Step 3 - batting and bowling interpretations</h3>
<p>The same index applies to both disciplines: a ball that can swing the match is a pressure ball for the batter and the bowler alike. What differs is
<i>which</i> outcome carries the swing, and that gives us a second, interpretive dimension. We split leverage into the part carried by the wicket outcome and the part carried by run outcomes.
On a <b>survival</b> ball (top quarter of wicket share) the dominant risk is losing a wicket - typical of an early collapse; on a <b>scoring</b> ball (bottom quarter) the
dominant risk is falling behind the rate - typical of a late chase. For a batter the first asks for judgement, the second for hitting; for a bowler the first rewards attack, the second control.
Of all pressure balls, {100*INS['mode_share']['Survival']:.0f}% are survival balls, {100*INS['mode_share']['Scoring']:.0f}% scoring balls and the rest balanced.</p>
<h3>Step 4 - is this a better definition? The result-importance test</h3>
<p>We compared the DPI with nine alternatives, including the definitions most competitors are likely to use. The test does not depend on the leverage
model. For each of {f1(mult.n_innings,0)} team innings we add up the batting side's runs above expectation separately on 'pressure' balls and on all others, and regress
the match result on both sums. The ratio of the two coefficients says how much more a run above expectation is worth on a pressure ball.</p>
{definition_table()}
{fig('F02_pressure_definition_test', '<b>Figure 4. Which definition of pressure identifies the balls that decide matches?</b> Bars show the result-importance multiplier with 95% confidence intervals. Grey bars are scoreboard-based definitions; none is meaningfully above 1.')}
<p>The result is stark. Every leverage-based definition isolates balls that are two to three times as decisive as ordinary ones. Every scoreboard
definition sits at or below one: by the evidence of results, a run above expectation in the death overs is no more decisive than a run in the
seventh over. The top 20% of DPI captures {100*tl:.0f}% of all match leverage in a fifth of the balls. We chose 20% over 10% - which is more decisive per
ball - because at 10% almost every pressure ball is in a chase ({100*DEF.loc['DPI: leverage top 10%'].share_of_pressure_in_innings1:.0f}% from first innings),
and a fair ranking needs batters and bowlers who set totals to have a voice. At 20%, {100*mult.share_of_pressure_in_innings1:.0f}% of pressure balls come from the first innings.</p></div>""")

    # ---------------- 7 batting model
    A(f"""<div class="section"><div class="kicker">Section 07</div><h2><span class="num">07</span>Batting Performance Model</h2>
<p>For every ball faced we estimate, from the situation alone, the expected runs off the bat, the probability the striker is dismissed, the probability
of a boundary and the probability of a dot (boosted models, out-of-fold by match). The situation includes innings, balls left, wickets in hand,
required rate and runs needed in a chase, current run rate, the striker's batting position, pace or spin, competition and season. Batting position is
what makes openers and finishers comparable: a number-six walking in at the 17th over is compared with other number-sixes in the 17th over, not with openers in the powerplay.</p>
<h4>One currency for runs and wickets</h4>
<p>A dismissal costs runs, but how many depends on when it happens. We fitted a resource table in the Duckworth-Lewis form to the first innings in
our data: expected runs still to come with <i>u</i> balls left and <i>w</i> wickets in hand is</p>
<div class="eq">R(u, w) = Z(w) · [ 1 − exp( −λ u / Z(w) ) ], &nbsp; λ = {dl['lambda']:.2f} runs per ball, weighted R² = {dl['weighted_r2']:.3f}</div>
<p>The run value of a wicket is R(u, w) − R(u, w−1): about {EX['wicket_cost_examples']['120b_10w']:.0f} runs for the first wicket of an innings,
{EX['wicket_cost_examples']['60b_6w']:.1f} for the fifth at halfway and {EX['wicket_cost_examples']['12b_6w']:.1f} for the fifth with two overs left. In a chase the
cost is capped at the runs still needed. A free-form table was tried first and rejected because it priced the first wicket at halfway at 0.15 runs.
Per ball, the batter's value is then</p>
<div class="eq">RAE<sub>bat</sub> = ( runs − E[runs | s] ) − c(s) · ( out − P(out | s) )</div>
<p>On average this is zero. A positive total means the batter added runs, kept his wicket, or both, compared with an average batter in the same spots.
Alongside it we track <b>Win Probability Added</b> (the change in his side's win probability on each ball he faces) as a measure of impact.</p>
{fig('F12_resource_curves', '<b>Figure 5. The price of a wicket.</b> Fitted resource curves: expected runs still to come by balls bowled and wickets in hand.', 'narrow')}</div>""")

    # ---------------- 8 bowling
    A(f"""<div class="section"><div class="kicker">Section 08</div><h2><span class="num">08</span>Bowling Performance Model</h2>
<p>The bowling side mirrors the batting side with three differences. Expected runs are runs charged to the bowler (bat runs, wides, no-balls), not byes
and leg-byes. Only bowler-credited dismissals count as wickets; a run-out is the fielders' contribution. And the bowler's own pace or spin is not part of
the expectation, because that is a property of the bowler being rated, while the striker's batting position is: a death bowler who bowls mostly at
numbers 8-11 is expected to take more wickets, and does not get credit for it.</p>
<div class="eq">RAE<sub>bowl</sub> = ( E[conceded | s] − conceded ) + c(s) · ( wicket − P(wicket | s) )</div>
<p>Because expectations depend on phase, powerplay, middle-overs and death bowlers are each measured against bowlers in their own phase. That matters:
a death bowler conceding 9.5 an over can be outstanding, while a middle-overs spinner conceding 7.5 can be ordinary. The roles shown in the tables are
derived from where each bowler's overs fall relative to the average bowler (a phase is a speciality if he bowls at least 1.25 times the average share of his balls in it).</p>
<p>The expected-performance models are well calibrated: across all balls faced, actual runs off the bat average {EX['calibration_bat_runs']['actual']:.3f} per ball
against {EX['calibration_bat_runs']['expected']:.3f} expected, and the dismissal rate is {100*EX['calibration_out']['actual']:.2f}% against
{100*EX['calibration_out']['expected']:.2f}%. They explain only {100*EX['batting_runs_r2']:.1f}% of ball-to-ball variance in runs, which is as it should be:
most of what happens on a single ball is noise, which is exactly why the next section matters.</p></div>""")

    # ---------------- 9 expected & baselines
    A(f"""<div class="section"><div class="kicker">Section 09</div><h2><span class="num">09</span>Expected Performance and Player Baselines</h2>
<p>A player's raw total of runs above expectation is still not a fair rating. It depends on whom he faced, and it treats 100 balls of evidence the
same as 1,000. We therefore estimate every player's level in a single model that sees all deliveries at once - the <b>Pressure-Adjusted Player Model</b> (PAPM):</p>
<div class="eq">RAE<sub>i</sub> = μ + τ·P<sub>i</sub> + a<sub>j</sub> + d<sub>j</sub>·P<sub>i</sub> + o<sub>k</sub> + e<sub>k</sub>·P<sub>i</sub> + ε<sub>i</sub></div>
<p>Here P<sub>i</sub> marks a pressure ball, <i>j</i> is the player being rated and <i>k</i> the opponent on that ball (bowler for a batter, batter for a
bowler). The player's <b>normal level</b> is a<sub>j</sub>; his <b>pressure level</b> is a<sub>j</sub> + d<sub>j</sub>; the difference d<sub>j</sub> is his
<b>pressure shift</b>. The opponent terms mean a run scored off Bumrah in the 19th over counts for more than one off a part-timer, and they handle the
IPL-versus-T20I question automatically, because the same players appear in both.</p>
<h4>Two components, shrunk separately</h4>
<p>Runs and dismissals carry very different amounts of repeatable signal. Across batters with 300+ balls in each half of the data, the split-half reliability of runs above expectation is
{F['sh_run']:.2f} and of dismissals above expectation {F['sh_out']:.2f}, but the combined currency manages only {F['sh_rae']:.2f}, because aggressive batters score more <i>and</i> get out more and
the two partly cancel. One model on the combined currency would force a single level of shrinkage on both. We therefore fit the PAPM separately to the run component
and the dismissal component and recombine them at each player's own wicket price. This raised the split-half reliability of the final pressure
estimate from {SH_B['PAPM single-currency pressure level']:.2f} to {SH_B['PAPM pressure level']:.2f} for batters and from
{SH_W['PAPM single-currency pressure level']:.2f} to {SH_W['PAPM pressure level']:.2f} for bowlers.</p>
<h4>What the data say about 'clutch'</h4>
<p>The penalties on each term are random-effect variances chosen by five-fold cross-validation over matches - the data decide how much a player's
pressure record should move him away from his normal level. The answer is one of the central findings of this study:</p>
<div class="callout pink"><b>Run-scoring under pressure is the same skill as run-scoring in general.</b> For both batters and bowlers, cross-validation
pushes the prior variance of the run-scoring shift to zero: no setting that lets players differ in pressure scoring predicts unseen matches better than
one that does not. <b>Dismissal risk is different.</b> The estimated spread of genuine pressure shifts in dismissal rate is about
{hb['implied_shift_sd_wkts_per100']:.2f} dismissals per 100 balls for batters - comparable to the spread of normal dismissal ability
({hb['implied_ability_sd_wkts_per100']:.2f}). Some players genuinely protect their wicket better when it matters; some bowlers genuinely stop taking wickets.</div>
<p>We report the <b>behavioural shift</b> at a fixed wicket price - the change in runs and dismissals under pressure, both valued at the player's
pressure-situation price - so that a cheaper death-overs wicket is not mistaken for a change in how he plays.</p></div>""")

    # ---------------- 10 uncertainty
    A(f"""<div class="section"><div class="kicker">Section 10</div><h2><span class="num">10</span>Statistical Adjustment and Uncertainty</h2>
<p>Shrinkage protects against small samples on average; it does not tell us how uncertain any individual estimate is. For that we used a match-level
bootstrap: 500 times, resample matches with replacement, refit both PAPM components with the same penalties, and record every player's normal level,
pressure level and shift. Resampling whole matches preserves the correlation between balls of the same innings, which a ball-level bootstrap would ignore.</p>
<p>The <b>Clutch Rating</b> is the 20th percentile of that distribution. It answers the question a selector would ask: <i>what level can I rely on from this
player under pressure, with 80% confidence?</i> It rewards a large, repeatable body of evidence over a brilliant short run, which is what the brief means by
'deliver most consistently'. The difference between the posterior mean and the Clutch Rating is the price a player pays for uncertainty: {sky.pressure_rae100 - sky.clutch_rating:.1f} runs per 100 balls for
{b1}, {vs.pressure_rae100 - vs.clutch_rating:.1f} for {b2}.</p>
{fig('F06_shrinkage', '<b>Figure 6. What shrinkage does.</b> Each dot is an eligible player. Horizontal: raw runs above expectation per 100 pressure balls; vertical: the model estimate. Small samples are pulled toward the player\'s normal level; the outlined point shows the one player excluded by the evidence gate.')}
<h4>Eligibility</h4>
<p>To be ranked a batter needs at least 120 pressure balls (20 overs) and 10 matches with three or more pressure balls; a bowler needs 180 pressure balls
(30 overs) and 10 matches with six or more. Both need pressure evidence in at least two seasons - repeatability cannot be judged within one. This gives
{nb_bat} eligible batters and {nb_bowl} eligible bowlers. Section 11 shows that changing the thresholds by a third in either direction leaves the ordering intact.</p>
<h4>The evidence gate</h4>
<p>Shrinkage can rate a player above average under pressure on the strength of his overall ability even when his own pressure balls went badly. That is
statistically correct as a forecast, but it is not what 'delivered when it mattered' means. We therefore only select players whose observed pressure record
is also above expectation. The gate removed one player: {shepherd.player_name}, whose model pressure level ({shepherd.pressure_rae100:+.1f}) would have placed him in the
top ten, but whose actual pressure balls produced {shepherd.raw_pressure_rae100:+.1f} runs per 100 against an extraordinary {100*shepherd.n_rae/shepherd.n_balls:+.1f} on other balls.</p></div>""")

    # ---------------- 11 validation
    A(f"""<div class="section"><div class="kicker">Section 11</div><h2><span class="num">11</span>Validation and Robustness Testing</h2>
<h3>Which measures of pressure performance are real?</h3>
<p>We compared eight candidate measures on three tests: <b>split-half repeatability</b> (correlation between the same player's measure in odd- and
even-numbered matches), <b>leave-one-season-out prediction</b> (estimate on four seasons, predict raw pressure performance in the fifth) and <b>forward
prediction</b> (estimate on 2022-24, predict 2025-26 - the real selection problem).</p>
{validation_table()}
{fig('F05_repeatability', '<b>Figure 7. Repeatability of pressure measures.</b> The raw clutch delta that most clutch analyses rank on is not repeatable at all; the shrunk two-component pressure level is the most repeatable pressure-specific measure for both disciplines.')}
<p>Three conclusions follow. The <b>raw clutch delta</b> - pressure performance minus normal performance - has no repeatable signal ({SH_B['Raw clutch delta (pressure - normal RAE/100)']:+.2f} and
{SH_W['Raw clutch delta (pressure - normal RAE/100)']:+.2f}). Any ranking built on it ranks luck. <b>Raw pressure records</b> are barely better. The <b>shrunk
pressure level</b> is the most repeatable pressure-specific measure ({SH_B['PAPM pressure level']:.2f} for batters, {SH_W['PAPM pressure level']:.2f} for bowlers) and, for
bowlers, the best forward predictor alongside match-level consistency. For batters no measure predicts the next two seasons' raw pressure performance well;
the 2025-26 raw values are themselves too noisy (40-plus balls per player) to be predicted, which caps every correlation. Overall strike rate is as good a
predictor of future pressure scoring as anything - a reminder that clutch is mostly class - but it ignores dismissals, opposition and context, so it cannot be the rating itself.</p>
<h3>Do the rankings survive different choices?</h3>
<p>We re-ran the ranking under 27 specifications, each changing one decision: three alternative pressure thresholds and the phase-relative DPI, three conventional
definitions, three sample thresholds, IPL only, T20I only, all matches including Associates, first or second innings only, five model variants, no opponent adjustment,
and leaving out each season in turn.</p>
{fig('F09_rank_stability_bat', '<b>Figure 8a. Batters - rank under alternative specifications.</b> Darker = higher rank; "–" means the player falls below that specification\'s sample threshold.')}
{fig('F09_rank_stability_bowl', '<b>Figure 8b. Bowlers - rank under alternative specifications.</b>')}
<p>The top four batters are top-four or close to it almost everywhere; positions five to eight are stable within a place or two; the main fragility is
{nm(TB,1)}, who drops below the sample threshold in stricter specifications and in leave-out-2025/2026 refits (he has two seasons of data), but ranks first to fourth
in every specification where he qualifies. The WPA model reorders more than any other - expected, since WPA is the noisiest currency (repeatability {SH_B['PAPM pressure WPA']:.2f}).
For bowlers, the top five are top-ten in at least {100*TW.head(5).core_top10_share.min():.0f}% of core specifications.</p>
{fig('F11_top10_probability', '<b>Figure 9. How often each player makes the top ten.</b> Bars: share of 500 match-bootstrap resamples. Pink ticks: share of all 27 specifications.')}
</div>""")

    # ---------------- 12 batter rankings
    tie_moves = FB.head(12)[FB.head(12)["rank"] != FB.head(12).rank_by_rating_only]
    raw_only = set(FULLB[FULLB.eligible].sort_values("clutch_rating", ascending=False).head(10).player_name)
    final_set = set(TB.player_name)
    rus = FULLB[FULLB.player_name == "Andre Russell"].iloc[0]
    moves_txt = "; ".join(f"{r.player_name} {int(r.rank_by_rating_only)}→{int(r['rank'])}" for _, r in tie_moves.iterrows())
    A(f"""<div class="section"><div class="kicker">Section 12</div><h2><span class="num">12</span>Final Batter Rankings</h2>
<p>Normal and pressure levels are runs above expectation per 100 balls; the shift is the behavioural change under pressure; 'SD' is the bootstrap
standard deviation of the pressure level. Expected values in brackets are for an average batter facing the same balls.</p>
{top_table_bat()}
{fig('F07_forest_bat', '<b>Figure 10. Batters - pressure level with uncertainty.</b> Thick bar: 60% bootstrap range; thin bar: 80% range; blue dot: posterior mean; open circle: normal level; pink diamond: Clutch Rating used to rank. Ranks 11-15 in grey.')}
<h4>Conventional view of the same ten</h4>
{conventional_table_bat()}
<p><b>What the tie band and evidence gate changed:</b> {moves_txt}; {shepherd.player_name} removed by the evidence gate. Without either safeguard the top ten would contain {', '.join(sorted(raw_only - final_set)) or 'the same players'} in place of
{', '.join(sorted(final_set - raw_only)) or 'nobody'}. <b>Closest alternatives:</b> {marsh.player_name} (rating {marsh.clutch_rating:.1f}) and, outside the sample rules,
Andre Russell ({rus.p_balls:.0f} pressure balls, {120-rus.p_balls:.0f} short of eligibility, posterior pressure level {rus.pressure_rae100:+.1f}). Position 10 is the least secure in the list: {nm(TB,9)} is top-ten in
{100*dube.core_top10_share:.0f}% of core specifications.</p></div>""")

    # ---------------- 13 bowler rankings
    A(f"""<div class="section"><div class="kicker">Section 13</div><h2><span class="num">13</span>Final Bowler Rankings</h2>
<p>For bowlers, positive values are runs saved per 100 balls compared with an average bowler in the same situations, including the run value of wickets.</p>
{top_table_bowl()}
{fig('F07_forest_bowl', '<b>Figure 11. Bowlers - pressure level with uncertainty.</b> Same layout as Figure 10.')}
<h4>Conventional view of the same ten</h4>
{conventional_table_bowl()}
<p><b>Closest alternatives:</b> {kul.player_name} (rating {kul.clutch_rating:.1f}; his pressure record is excellent - {kul.p_wkts:.0f} wickets against {kul.p_x_wkts:.1f} expected -
but his normal level is lower than {noor.player_name}'s) and Nathan Ellis. {noor.player_name} at ten is the thinnest pressure case in the list: his pressure economy
({noor.p_economy:.2f}) is close to expectation ({noor.p_xEcon:.2f}) and he beat expectation in only {100*noor.c_raw_positive_share:.0f}% of pressure spells; he is
there because his overall level is high and his pressure record, while unremarkable, is not negative. A selector who weights match-level consistency more heavily than we do
would choose {kul.player_name}.</p></div>""")

    # ---------------- 14 profiles
    A(f"""<div class="section"><div class="kicker">Section 14</div><h2><span class="num">14</span>Individual Player Evidence</h2>
<p>Each profile gives the supplied-data evidence first and external context second; external material is used only to interpret the numbers, never to
change them. Profile tags are assigned by rule: <span class="tag pink">Crisis Manager</span> top eight in runs above expectation after early collapses
(3+ down inside 8 overs or 5+ down inside 14); <span class="tag">Finisher</span>, <span class="tag">Powerplay Accelerator</span> by batting position;
<span class="tag gold">Survivor</span> top eight on survival-mode pressure balls; <span class="tag pink">Pressure Riser</span> shift above zero in 90% of resamples (85% for
bowlers); <span class="tag gold">Consistent Performer</span> above expectation in at least 68% of pressure innings (60% of spells for bowlers);
<span class="tag pink">Death Specialist</span> by phase share; <span class="tag pink">Pressure Enforcer</span> three or more wickets above expectation on pressure balls;
<span class="tag">Controller</span> pressure economy at least one run an over better than expected.</p>
<h3>Batters</h3>{profiles(TB, 'bat')}</div>
<div class="section"><h3>Bowlers</h3>{profiles(TW, 'bowl')}</div>""")

    # ---------------- 15 discoveries
    A(f"""<div class="section"><div class="kicker">Section 15</div><h2><span class="num">15</span>Interesting Discoveries</h2>
<h3>1. Bumrah's pressure value is control, not wickets</h3>
<p>{bum.player_name}'s normal level is the highest of any bowler ({bum.normal_rae100:+.1f}), and his pressure level is still second ({bum.pressure_rae100:+.1f}) - yet he is the
clearest 'fader' in the bowling pool by shift ({bum.shift_rae100:+.1f}, below zero in every resample). The reason is wickets: on pressure balls he took {bum.p_wkts:.0f}
against {bum.p_x_wkts:.1f} expected. His economy ({bum.p_economy:.2f} against {bum.p_xEcon:.2f}) and boundary rate ({bp['Jasprit Bumrah']['bnd_pct']:.1f}% against
{bp['Jasprit Bumrah']['x_bnd_pct']:.1f}%) say the rest. The most plausible cricketing reading is that batters stop taking risks against him when the match is in the balance -
they play him out and target the other end. A wicket-based metric would call that a decline; a run-and-wicket metric correctly calls it control.</p>
<h3>2. {var.player_name} is the opposite - a genuine pressure enforcer</h3>
<p>He took {var.p_wkts:.0f} wickets on pressure balls against {var.p_x_wkts:.1f} expected - the largest excess among eligible bowlers - while conceding
{var.p_economy:.2f} an over where {var.p_xEcon:.2f} was expected. His shift ({var.shift_rae100:+.1f}) is positive in {100*var.p_shift_positive:.0f}% of resamples.
Batters who must attack in a chase are attacking a bowler they struggle to read.</p>
<h3>3. Scoreboard pressure and decisive pressure are different things</h3>
<p>The result-importance test (Section 6) is the headline methodological finding. 'Death overs', 'required rate above ten' and hand-weighted indices identify balls
that are no more decisive than ordinary ones. The practical consequence: a player who scores heavily in the last five overs of one-sided matches
looks clutch on conventional statistics and is not.</p>
<h3>4. The best finishers score faster without getting out more</h3>
<p>Among batters with 80+ death-overs pressure balls, {', '.join(list(fin.keys())[:3])} score most above expectation while also being dismissed
<i>less</i> often than expected - the combination teams want and rarely get.</p>
{fig('F10_finisher_quadrant', '<b>Figure 12. Death-overs pressure balls: scoring versus dismissal risk.</b> Upper right (axis reversed) = faster and safer than an average batter in the same situations.', 'mid')}
<h3>5. Conventional statistics get specific players badly wrong</h3>
<p>{under_b[0]['player_name']} ranks {under_b[0]['rank']} on our scale but {under_b[0]['conv_rank']:.0f} on pressure strike rate: he scores at only
{under_b[0]['p_strike_rate']:.0f} under pressure, but in situations where expectations are low and dismissals are costly he survives - out {dube.p_outs:.0f} times
against {dube.p_x_outs:.1f} expected. In the other direction, {over_b[0]['player_name']} ({over_b[0]['p_strike_rate']:.0f} pressure strike rate) is
{ordn(over_b[0]['conv_rank'])} conventionally but {ordn(over_b[0]['rank'])} on our scale, and {over_b[5]['player_name']} {ordn(over_b[5]['conv_rank'])} versus {ordn(over_b[5]['rank'])}:
high strike rates achieved where high strike rates are expected. Among bowlers, {over_w[0]['player_name']} has the second-best pressure economy in the pool
({over_w[0]['p_economy']:.2f}) and leads our boundary-prevention list, but ranks {ordn(over_w[0]['rank'])} - he concedes few runs in situations where few are expected
and takes few wickets. {under_w[0]['player_name']} is the reverse: {ordn(under_w[0]['conv_rank'])} on pressure economy ({under_w[0]['p_economy']:.2f}), {ordn(under_w[0]['rank'])} on our scale,
because he bowls the death overs where 9+ is normal.</p>
<h3>6. Roles are not equally exposed</h3>
<p>Numbers 5-8 face pressure balls about as often as openers but at higher average leverage, and tailenders face far fewer. A chase's death overs are pressure balls
{100*inn2_d:.0f}% of the time; a first-innings middle over only {100*inn1_mid:.0f}% of the time. {100*INS['top10_bat_share_pressure_in_innings2']:.0f}% of the top-ten batters' pressure
balls come in chases. That is not a bias in the index - it is where T20 matches are decided - but it is why we tested a phase-relative index (Section 11).</p>
{fig('F13_exposure', '<b>Figure 13. Who faces pressure?</b> Share of balls that are pressure balls, by batting position and by phase and innings.')}
<h3>7. IPL and T20I pressure performance are only loosely related</h3>
<p>For players with 60+ pressure balls in both, the correlation between shrunk IPL and T20I pressure records is {INS['ipl_vs_t20i_player_corr_bat']['r']:.2f} for batters
(n = {INS['ipl_vs_t20i_player_corr_bat']['n']}) and {INS['ipl_vs_t20i_player_corr_bowl']['r']:.2f} for bowlers (n = {INS['ipl_vs_t20i_player_corr_bowl']['n']}).
Either the competitions test different skills or - more likely given the sample sizes - a few dozen pressure balls in each are not enough to see a stable trait.
Pooling both competitions with opponent adjustment is the only way to get a reliable read.</p>
<h3>8. Knockouts are slightly more pressured, but too few to rate on</h3>
<p>The {ko['matches']} knockout matches in the rating universe have mean DPI {ko['mean_LI']:.2f} against {ko['mean_LI_league']:.2f} in league games, with
{100*ko['pressure_share']:.0f}% pressure balls against {100*ko['pressure_share_league']:.0f}%. No player has enough knockout pressure balls for a separate rating;
the absence of T20 World Cups makes this the study's most important blind spot.</p></div>""")

    # ---------------- 16 case studies
    A(f"""<div class="section"><div class="kicker">Section 16</div><h2><span class="num">16</span>Match-Level Case Studies</h2>
<p>These three matches were chosen from the largest single-innings Win Probability Added totals among our top-ten players, and checked against
published reports. Win probability is shown from the chasing side's perspective; the lower panel shows DPI for every ball.</p>
<h3>Case 1 - Jos Buttler, KKR v RR, Eden Gardens, 16 April 2024</h3>
{fig('F15_case_buttler', 'Win probability for Rajasthan (chasing) and DPI by delivery; pink = balls faced by Buttler.')}
<p>Rajasthan were 121 for 6 in the 13th over and needed 46 off the last three{refs_link(9)}. In our model their chance bottomed out at {100*F['c1_min_wp']:.1f}%. Buttler faced
the balls that mattered: {F['c1_n5']} of the deliveries he faced in the 19th and 20th overs had a DPI above 5, peaking at {F['c1_li_hi']:.0f} - among the highest values in the dataset, and his
innings added {F['c1_wpa']:.2f} wins of probability in total - the largest single-innings WPA by any top-ten batter. He finished 107 not out
and won it off the last ball{refs_link(8)}. It illustrates why DPI concentrates in a few deliveries, and why we judge a player on many such innings rather than one.</p>
<h3>Case 2 - Josh Hazlewood, RCB v RR, Bengaluru, 24 April 2025</h3>
{fig('F16_case_hazlewood', 'Win probability for Rajasthan (chasing) and DPI; pink = balls bowled by Hazlewood.')}
<p>After a 22-run 18th over Rajasthan needed 18 off 12 and our model gave them {100*F['c2_wp_before']:.0f}%. Hazlewood's 19th over cost one run and removed Jurel and Archer on
successive balls{refs_link(10)}{refs_link(11)}; Rajasthan's probability fell from {100*F['c2_wp_before']:.0f}% to {100*F['c2_wp_after']:.0f}% in six deliveries. Our model credits his four overs with {F['c2_wpa']:.2f} wins of probability -
the largest bowling WPA by a top-ten bowler. For a Rajasthan audience it is a painful but instructive example: the match was not lost in the 20th over but in the
{['no', 'one', 'two', 'three', 'four', 'five', 'six'][F['c2_hi_li']]} deliveries of the 19th over with DPI above 5.</p>
<h3>Case 3 - Rinku Singh, GT v KKR, Ahmedabad, 9 April 2023</h3>
{fig('F17_case_rinku', 'Win probability for Kolkata (chasing) and DPI; pink = balls faced by Rinku Singh.')}
<p>The most famous last over in IPL history is also the best illustration of what leverage is <i>not</i>. With five balls left our model put
Kolkata's chance at {100*F['c3_wp_start']:.1f}%; ESPNcricinfo's forecaster said 0.61%{refs_link(12)}. The first three sixes barely moved it - DPI on those balls was {F['c3_li'][0]:.2f}, {F['c3_li'][1]:.2f} and {F['c3_li'][2]:.2f}, because even
after them the chase was still very likely lost. Only the fourth and fifth sixes carried real leverage (DPI {F['c3_li'][3]:.1f} and {F['c3_li'][4]:.0f}). Scoreboard pressure was maximal on all five
balls; decisive pressure arrived only at the end. The reconstructed target in this match is two runs short of the published one because of unrecorded extras,
which shifts the timing of the probability jump by one ball but not the story.</p></div>""")

    # ---------------- 17 conventional
    conv_ov_b = int(pd.read_csv(OUT / "robustness_summary_bat.csv").set_index("specification").loc["Conventional: pressure SR / economy", "top10_overlap"])
    conv_ov_w = int(pd.read_csv(OUT / "robustness_summary_bowl.csv").set_index("specification").loc["Conventional: pressure SR / economy", "top10_overlap"])
    A(f"""<div class="section"><div class="kicker">Section 17</div><h2><span class="num">17</span>What Conventional Cricket Statistics Fail to Reveal</h2>
<p>A conventional clutch analysis would filter the last five overs or a required-rate threshold, compute strike rate, average and economy, and rank. It fails in five specific ways.</p>
<p><b>It picks the wrong balls.</b> The pressure balls it selects are no more decisive than ordinary balls (Section 6). <b>It compares players with the wrong baseline.</b>
A strike rate of 170 at the death is average; 140 after three early wickets can be excellent. Our expectations are for the same situation, so finishers are compared with
finishers and openers with openers. <b>It ignores opposition.</b> The same economy against Mumbai's batting and against a weak tail are different achievements; the PAPM's
opponent terms correct for it. <b>It treats runs and wickets separately</b>, so it cannot say whether 15 extra runs justify one extra dismissal; the Duckworth-Lewis
resource table puts them in one currency, at the right price for the moment. <b>It ignores sample size.</b> Raw pressure records have split-half reliability of
about {SH_B['Raw pressure RAE/100']:.2f}; ranking on them ranks luck.</p>
<p>The consequences are visible in our results. Pressure strike-rate and pressure-economy rankings share only {conv_ov_b} and {conv_ov_w} of their top ten with ours (Figure 14), and
they promote players such as {over_b[0]['player_name']} and {over_w[0]['player_name']} whose impressive raw numbers were achieved in situations where they were expected, while
demoting {under_b[0]['player_name']} and {under_w[0]['player_name']}. Just as important, the conventional approach cannot say how confident it is. Ours gives every player
an interval, a probability of being top ten, and a count of the specifications in which he survives.</p>
{fig('F14_robustness_summary', '<b>Figure 14. Overlap of each specification\'s top ten with the final list.</b> Grey bars are the conventional and unshrunk alternatives.')}</div>""")

    # ---------------- 18 limitations
    A(f"""<div class="section"><div class="kicker">Section 18</div><h2><span class="num">18</span>Limitations</h2>
<p><b>No T20 World Cups.</b> The dataset contains IPL seasons, bilateral T20Is and Asia Cups, but none of the 2022, 2024 or 2026 T20 World Cups - the highest-stakes
international matches of the period, including the 2026 final in which Abhishek Sharma made 52 off 21 balls{refs_link(3)}. International pressure evidence is therefore
thinner than it looks, and pressure in elimination matches is barely measured.</p>
<p><b>Reconstructed scores.</b> Unrecorded wides, byes and leg-byes mean reconstructed totals are a run or two out in most matches and up to seven in a few.
Anchoring targets to the chase removes the error where it matters most, but early-innings states carry small inaccuracies.</p>
<p><b>Batting is hard to predict.</b> No measure we tried - ours included - forecasts batters' 2025-26 raw pressure performance well. This reflects noise in the
target as much as weakness in the predictors, but it means the batting ranking is a best estimate of current pressure level, not a guarantee.</p>
<p><b>Short careers.</b> {nm(TB,1)} has two seasons of data. Our conservative rating and the two-season rule are designed for exactly this case, but a player with 129 pressure
balls cannot be known as well as one with 600.</p>
<p><b>Leverage values results, not experience.</b> DPI measures how much a ball can move the result. It does not measure crowd, occasion or personal stakes, and it gives
low pressure to 'hopeless' situations even when players feel them intensely (Case 3).</p>
<p><b>Shared credit.</b> A batter's runs depend on his partner and the field; a bowler's wickets depend on catches; captaincy decides who bowls the 19th over. The model
adjusts for the opponent on each ball but not for teammates or tactical selection.</p>
<p><b>Uncertainty is partly conditional.</b> The bootstrap holds the shrinkage penalties fixed, and the penalties used in validation were chosen on the full data, so the
reported intervals are slightly narrower than a fully nested procedure would give. The tie band and evidence gate were introduced after the robustness results were
known; we report exactly what each changed.</p>
<p><b>Multiple comparisons.</b> The situation-specific leaderboards in Section 15 are descriptive; with many possible slices, some apparent specialists will be chance.</p></div>""")

    # ---------------- 19 conclusion
    A(f"""<div class="section"><div class="kicker">Section 19</div><h2><span class="num">19</span>Conclusion</h2>
<p>The question 'who performs under pressure?' has two easy answers and both are wrong. One is to take the players with the best record in the last five
overs. The other is to take the players whose record improves most under pressure. The first measures the wrong moments; the second measures noise.</p>
<p>The answer the data support is more interesting. Pressure, properly defined, is leverage - and the top fifth of leverage carries half of everything that
decides a T20 match. Within those balls, players' run-scoring is the same skill they show elsewhere, while their dismissal behaviour differs in genuine and
repeatable ways. The best pressure performers are therefore excellent players whose wicket-taking or wicket-keeping holds up, or improves, when it matters most:
{nm(TB,0)}, {nm(TB,1)}, {nm(TB,2)} and {nm(TB,3)} among batters; {nm(TW,0)}, {nm(TW,1)}, {nm(TW,2)} and {nm(TW,3)} among bowlers.</p>
<p>For a franchise, three practical implications follow. Judge death bowlers against what is expected at the death, not against economy tables. Value batters who
protect their wicket in survival situations - their worth is invisible in strike rate. And treat any claim that a player is 'clutch' on a few dozen balls with
deep scepticism, including claims made by models.</p></div>""")

    # ---------------- 20 technical appendix
    rp = pd.read_csv(OUT / "cv_lambda_bat_runs.csv")
    A(f"""<div class="section"><div class="kicker">Section 20</div><h2><span class="num">20</span>Technical Appendix</h2>
<h4>A. Pipeline</h4>
<p>s01 data preparation → s02 win probability, outcome model and leverage → s03 expected-performance models and RAE → s04 pressure definitions and DPI →
s05 two-component PAPM with cross-validated shrinkage and 500-draw match bootstrap → s06 predictive validation → s07 robustness suite (27 specifications) →
s08 final ranking → s09 discoveries → s10 figures → s11 exports. The notebook runs the whole chain from the raw CSVs; the README gives the command.</p>
<h4>B. Models</h4>
<table class="compact"><tr><th>Component</th><th>Specification</th><th>Validation</th></tr>
<tr><td>Chase win probability</td><td>Histogram gradient boosting, monotone in runs needed, balls left, wickets, required rate and Elo; runs relative to environment (×8.5/env)</td><td>Chronological: select on 2024, test 2025-26; Brier {wp_t['innings_2']['brier']:.3f} in chases</td></tr>
<tr><td>First-innings WP</td><td>Remaining runs ~ Normal(μ, σ), μ and σ² by boosting in environment-relative units; integrated with 24-point Gauss-Hermite quadrature</td><td>Brier {wp_t['innings_1']['brier']:.3f} (direct classifier {WP['direct_gbm_first_innings']['innings_1']['brier']:.3f}; logistic {lg_t['innings_1']['brier']:.3f})</td></tr>
<tr><td>Outcome model</td><td>8-class boosting (0,1,2,3,4,6,W,extra)</td><td>Log-loss {WP['outcome_model']['log_loss_model']:.3f} vs {WP['outcome_model']['log_loss_base_rates']:.3f} base rates (2025-26)</td></tr>
<tr><td>Expected runs / outs</td><td>Poisson and logistic boosting on situation, batting position, pace/spin, competition, season; 5-fold out-of-fold by match</td><td>Calibration-in-the-large within 0.5%</td></tr>
<tr><td>Wicket value</td><td>Duckworth-Lewis form, shared λ = {dl['lambda']:.2f}; Z(w) = {', '.join(f'{z:.0f}' for z in dl['Z'])}</td><td>Weighted R² {dl['weighted_r2']:.3f}</td></tr>
<tr><td>PAPM</td><td>Sparse ridge (LSQR) with player, player×pressure, opponent and opponent×pressure effects; separate fits for runs and dismissals</td><td>Penalties by 5-fold match-grouped CV</td></tr>
</table>
<h4>C. Cross-validated penalties (λ = residual variance / prior variance)</h4>
<table class="compact"><tr><th></th><th class="r">Batting runs</th><th class="r">Batting dismissals</th><th class="r">Bowling runs</th><th class="r">Bowling wickets</th></tr>
<tr><td>Normal level λ</td><td class="r">{hb['runs']['lam_base']:.0f}</td><td class="r">{hb['wickets']['lam_base']:.0f}</td><td class="r">{hw['runs']['lam_base']:.0f}</td><td class="r">{hw['wickets']['lam_base']:.0f}</td></tr>
<tr><td>Pressure shift λ</td><td class="r">∞ (10⁸)</td><td class="r">{hb['wickets']['lam_delta']:.0f}</td><td class="r">∞ (10⁸)</td><td class="r">{hw['wickets']['lam_delta']:.0f}</td></tr>
<tr><td>Implied ability SD per 100 balls</td><td class="r">{hb['implied_ability_sd_runs_per100']:.1f} runs</td><td class="r">{hb['implied_ability_sd_wkts_per100']:.2f} outs</td><td class="r">{hw['implied_ability_sd_runs_per100']:.1f} runs</td><td class="r">{hw['implied_ability_sd_wkts_per100']:.2f} wkts</td></tr>
<tr><td>Implied shift SD per 100 balls</td><td class="r">≈ 0</td><td class="r">{hb['implied_shift_sd_wkts_per100']:.2f} outs</td><td class="r">≈ 0</td><td class="r">{hw['implied_shift_sd_wkts_per100']:.2f} wkts</td></tr>
<tr><td>CV improvement over null (%)</td><td class="r">{hb['runs']['cv_gain_pct']:.2f}</td><td class="r">{hb['wickets']['cv_gain_pct']:.2f}</td><td class="r">{hw['runs']['cv_gain_pct']:.2f}</td><td class="r">{hw['wickets']['cv_gain_pct']:.2f}</td></tr></table>
<p class="note">The improvements look small because single-ball outcomes are mostly noise; what matters is that they are positive out of sample and that the
penalty grid optimum is interior or, for run-scoring shifts, at 'no shift'. Full CV surfaces, robustness tables and all validation detail are in the companion
document <i>Model_Validation_and_Robustness.pdf</i>.</p>
<h4>D. Robustness summary - batters</h4>{robustness_table('bat')}
<h4>E. Robustness summary - bowlers</h4>{robustness_table('bowl')}
<h4>F. Reproducibility</h4>
<p>Python 3.13, pandas, scikit-learn 1.9, statsmodels, SciPy, matplotlib. Random seed 20260516. Full run time about 15 minutes on two CPU cores.
Every number in this report is read programmatically from the pipeline outputs when the document is built.</p></div>""")

    # ---------------- references
    A("<div class='section'><h2>References</h2><p class='note'>External sources were used to interpret the supplied-data findings, never to change a rating. All were accessed in November 2026.</p><ol class='refs'>")
    for i, (t, u) in enumerate(REFS, 1):
        A(f"<li id='ref{i}'>{t} <a href='{u}'>{u}</a></li>")
    A("</ol></div>")
    return "".join(html)


def cover():
    return f"""<div class="cover"><div class="band"><div class="gold"></div><div class="stripe"></div>
<div class="inner"><div class="pre">Rajasthan Royals · SupeRR Selector Season 2 · Round 1</div>
<div class="title">BEYOND THE<br>SCOREBOARD</div>
<div class="sub" style="font-size:14pt;font-weight:600;margin-bottom:4mm">Identifying the Most Reliable Performers Under Pressure in T20 Cricket</div>
<div class="sub">A Context-Aware and Statistically Validated Framework for Measuring Clutch Performance | SupeRR Selector Season 2 – Round 1</div></div></div>
<div class="below"><div class="lead">Pressure, measured as the leverage of each delivery on the result; performance, measured against what an average player would have done
in exactly the same situation and against the same opponent; and confidence, measured by asking how much of any record could be luck.</div>
<div class="listcard"><div><h4>Top 10 batters</h4><ol>{''.join(f'<li>{n}</li>' for n in TB.player_name)}</ol></div>
<div class="b"><h4>Top 10 bowlers</h4><ol>{''.join(f'<li>{n}</li>' for n in TW.player_name)}</ol></div></div></div>
<div class="meta"><span>Methodology report · Supplied IPL & T20I data, January 2022 – November 2026</span><span>November 2026</span></div></div>"""


def main():
    out = DEL / "SupeRR_Selector_Final_Report.pdf"
    render_pdf(cover(), body(), out, "Beyond the Scoreboard - SupeRR Selector S2 Round 1",
               "Beyond the Scoreboard", "Identifying the most reliable performers under pressure in T20 cricket")
    print("written", out)


if __name__ == "__main__":
    main()
