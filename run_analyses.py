"""
YouTube Urticaria Infodemiology Study — Q1-Ready Analysis Pipeline
Produces tables, figures, and statistical tests per study protocol.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

try:
    from scipy.stats import mannwhitneyu, kruskal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATS = True
except ImportError:
    HAS_STATS = False

try:
    import statsmodels.api as sm
    from statsmodels.discrete.discrete_model import Logit
    HAS_LOGIT = True
except ImportError:
    HAS_LOGIT = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_CSV = "topic_outputs/comments_with_unmet_needs.csv"
OUTPUT_DIR = Path("topic_outputs/analysis_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Column mappings (dataset uses different names)
UNMET_NEEDS = [
    "uncontrolled_symptoms", "airway_fear", "treatment_failure",
    "medication_side_effect_fear", "trigger_uncertainty", "emotional_distress",
    "sleep_daily_impairment", "access_cost_barrier", "diagnostic_confusion",
    "misinformation_vulnerability",
]

UNMET_NEED_LABELS = {
    "uncontrolled_symptoms": "Uncontrolled symptoms",
    "airway_fear": "Airway/ER fear",
    "treatment_failure": "Treatment failure",
    "medication_side_effect_fear": "Medication side-effect fear",
    "trigger_uncertainty": "Trigger uncertainty",
    "emotional_distress": "Emotional distress",
    "sleep_daily_impairment": "Sleep/daily impairment",
    "access_cost_barrier": "Access/cost barrier",
    "diagnostic_confusion": "Diagnostic confusion",
    "misinformation_vulnerability": "Misinformation vulnerability",
}

TREATMENT_COLS = [
    "mentioned_antihistamine", "mentioned_steroid", "mentioned_h2_blocker",
    "mentioned_biologic", "mentioned_montelukast",
]
TREATMENT_LABELS = {
    "mentioned_antihistamine": "Antihistamine",
    "mentioned_steroid": "Steroid",
    "mentioned_h2_blocker": "H2 blocker",
    "mentioned_biologic": "Biologic (Xolair)",
    "mentioned_montelukast": "Montelukast",
}


def load_and_prepare():
    """Load data and standardize column names."""
    df = pd.read_csv(INPUT_CSV)
    n = len(df)

    # Standardize columns
    df["gender_inferred"] = df["gender"].fillna("unknown").astype(str).str.lower().replace({"": "unknown", "nan": "unknown"})
    df["gender_inferred"] = df["gender_inferred"].apply(lambda x: "unknown" if str(x) not in ["male", "female"] else x)

    df["topic_10"] = df["theme_name"].fillna("Other")
    df["sentiment_class"] = df["sentiment_vader_label"].fillna("neutral")
    df["sentiment_compound"] = pd.to_numeric(df["sentiment_vader_compound"], errors="coerce").fillna(0)
    df["comment_length_tokens"] = df["word_count"].fillna(0).astype(int)
    df["comment_date"] = pd.to_datetime(df["comment_date"], errors="coerce")

    # Collapse speaker_type for sparse cells
    speaker_map = {
        "patient": "patient", "caregiver": "caregiver", "clinician": "clinician",
        "clinician_educator": "clinician", "advertiser": "advertiser_spam",
        "advertiser_spam": "advertiser_spam", "general": "general_viewer",
        "general_viewer": "general_viewer", "unclear": "unclear",
    }
    df["speaker_collapsed"] = df["speaker_type"].fillna("unclear").astype(str).str.lower()
    df["speaker_collapsed"] = df["speaker_collapsed"].map(lambda x: speaker_map.get(x, "other"))
    df.loc[~df["speaker_collapsed"].isin(["patient", "caregiver", "clinician", "advertiser_spam", "general_viewer", "unclear"]), "speaker_collapsed"] = "other"

    return df


def fdr_correct(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction."""
    if not HAS_STATS:
        return pvals, [False] * len(pvals)
    rej, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    return p_adj, rej


def cliffs_delta(x, y):
    """Cliff's delta effect size (ordinal)."""
    n1, n2 = len(x), len(y)
    dominates = sum(1 for a in x for b in y if a > b)
    delta = (dominates - sum(1 for a in x for b in y if a < b)) / (n1 * n2)
    return delta


# ===========================================================================
# BLOCK 1 — Corpus characterization
# ===========================================================================
def block1_corpus(df):
    n_comments = len(df)
    n_videos = df["video_id"].nunique()
    period_start = df["comment_date"].min()
    period_end = df["comment_date"].max()

    comments_per_video = df.groupby("video_id").size()
    likes = pd.to_numeric(df["like_count"], errors="coerce").fillna(0)
    lengths = df["comment_length_tokens"]

    speaker_dist = df["speaker_type"].value_counts(normalize=True).mul(100).round(1)
    gender_dist = df["gender_inferred"].value_counts(normalize=True).mul(100).round(1)
    spam_pct = (df["speaker_collapsed"] == "advertiser_spam").mean() * 100 if "advertiser_spam" in df["speaker_collapsed"].values else 0

    rows = [
        ("N videos", n_videos),
        ("N comments", n_comments),
        ("Study period", f"{period_start.strftime('%Y-%m') if pd.notna(period_start) else '?'} – {period_end.strftime('%Y-%m') if pd.notna(period_end) else '?'}"),
        ("Comments per video (median, IQR)", f"{comments_per_video.median():.0f} ({comments_per_video.quantile(0.25):.0f}–{comments_per_video.quantile(0.75):.0f})"),
        ("Likes per comment (median, IQR)", f"{likes.median():.0f} ({likes.quantile(0.25):.0f}–{likes.quantile(0.75):.0f})"),
        ("Comment length tokens (median, IQR)", f"{lengths.median():.0f} ({lengths.quantile(0.25):.0f}–{lengths.quantile(0.75):.0f})"),
        ("Speaker type distribution (%)", speaker_dist.to_dict()),
        ("Gender inferred distribution (%)", gender_dist.to_dict()),
        ("Spam/advertiser proportion (%)", f"{spam_pct:.1f}"),
    ]
    tab = pd.DataFrame(rows, columns=["Metric", "Value"])
    tab.to_csv(OUTPUT_DIR / "table1_corpus_characteristics.csv", index=False)
    return tab


# ===========================================================================
# BLOCK 2 — Topics
# ===========================================================================
def block2_topics(df):
    topic_dist = df["topic_10"].value_counts()
    topic_pct = df["topic_10"].value_counts(normalize=True).mul(100)

    # 2.1 Topic prevalence + stratification
    tab2_rows = []
    for topic in topic_dist.index:
        n = topic_dist[topic]
        pct = topic_pct[topic]
        tab2_rows.append({"Topic": topic, "N": int(n), "Percent": f"{pct:.1f}%"})
    tab2 = pd.DataFrame(tab2_rows)

    # Chi-square: topic × speaker_collapsed (collapse to patient/caregiver/other)
    df_t = df.copy()
    df_t["speaker_3"] = df_t["speaker_collapsed"].replace({
        "general_viewer": "other", "clinician": "other", "advertiser_spam": "other", "unclear": "other"
    })
    if df_t["speaker_3"].nunique() >= 2 and df_t["topic_10"].nunique() >= 2:
        ct_speaker = pd.crosstab(df_t["topic_10"], df_t["speaker_3"])
        if ct_speaker.size > 0 and (ct_speaker < 5).sum().sum() < ct_speaker.size * 0.5:
            chi2_s, p_s, dof_s, _ = stats.chi2_contingency(ct_speaker)
            tab2["chi2_topic_speaker_p"] = p_s
        else:
            tab2["chi2_topic_speaker_p"] = np.nan
    else:
        tab2["chi2_topic_speaker_p"] = np.nan

    # Chi-square: topic × gender
    ct_gen = pd.crosstab(df["topic_10"], df["gender_inferred"])
    if ct_gen.size > 0:
        chi2_g, p_g, _, _ = stats.chi2_contingency(ct_gen)
        tab2["chi2_topic_gender_p"] = p_g
    else:
        tab2["chi2_topic_gender_p"] = np.nan

    # 2.2 Engagement by topic
    likes = pd.to_numeric(df["like_count"], errors="coerce").fillna(0)
    kw_groups = [likes[df["topic_10"] == t].values for t in df["topic_10"].unique() if (df["topic_10"] == t).sum() > 0]
    if len(kw_groups) >= 2 and all(len(g) > 0 for g in kw_groups):
        h, p_kw = kruskal(*kw_groups)
        tab2["kruskal_likes_p"] = p_kw
    else:
        tab2["kruskal_likes_p"] = np.nan

    # 2.3 Sentiment by topic
    ct_sent = pd.crosstab(df["topic_10"], df["sentiment_class"])
    if ct_sent.size > 0:
        chi2_sent, p_sent, _, _ = stats.chi2_contingency(ct_sent)
        tab2["chi2_sentiment_p"] = p_sent
    else:
        tab2["chi2_sentiment_p"] = np.nan

    tab2.to_csv(OUTPUT_DIR / "table2_topics.csv", index=False)

    # Table 4a — engagement by topic (median IQR)
    eng_rows = []
    for topic in df["topic_10"].unique():
        sub = df[df["topic_10"] == topic]
        l = pd.to_numeric(sub["like_count"], errors="coerce").fillna(0)
        eng_rows.append({
            "Topic": topic,
            "median_likes": l.median(),
            "IQR": f"{l.quantile(0.25):.0f}–{l.quantile(0.75):.0f}",
        })
    pd.DataFrame(eng_rows).to_csv(OUTPUT_DIR / "table4a_engagement_by_topic.csv", index=False)
    return tab2


# ===========================================================================
# BLOCK 3 — Unmet needs prevalence
# ===========================================================================
def block3_unmet_needs(df):
    prev = []
    for u in UNMET_NEEDS:
        if u not in df.columns:
            continue
        n = int(df[u].sum())
        pct = 100 * n / len(df)
        prev.append({"Unmet need": UNMET_NEED_LABELS.get(u, u), "N": n, "Percent": f"{pct:.1f}%"})
    tab3 = pd.DataFrame(prev)
    tab3.to_csv(OUTPUT_DIR / "table3_unmet_needs_prevalence.csv", index=False)
    return tab3


# ===========================================================================
# BLOCK 4 — Engagement + Unmet needs
# ===========================================================================
def block4_engagement_unmet(df):
    likes = pd.to_numeric(df["like_count"], errors="coerce").fillna(0)
    high_eng = df["high_engagement"].fillna(0).astype(int)

    rows = []
    pvals_mw = []
    for u in UNMET_NEEDS:
        if u not in df.columns:
            continue
        g1 = likes[df[u] == 1].values
        g0 = likes[df[u] == 0].values
        if len(g1) > 0 and len(g0) > 0:
            stat, p = mannwhitneyu(g1, g0, alternative="two-sided")
            pvals_mw.append(p)
        else:
            pvals_mw.append(1.0)

    p_adj, _ = fdr_correct(pvals_mw)
    idx = 0
    for u in UNMET_NEEDS:
        if u not in df.columns:
            continue
        g1 = likes[df[u] == 1].values
        g0 = likes[df[u] == 0].values
        delta = cliffs_delta(g1, g0) if len(g1) > 0 and len(g0) > 0 else np.nan
        p_adj_i = p_adj[idx] if idx < len(p_adj) else np.nan
        # OR for high_engagement
        ct = pd.crosstab(df[u], high_eng)
        if ct.shape == (2, 2):
            o = (ct.loc[1, 1] * ct.loc[0, 0]) / (ct.loc[1, 0] * ct.loc[0, 1] + 1e-9)
        else:
            o = np.nan
        rows.append({
            "Unmet need": UNMET_NEED_LABELS.get(u, u),
            "median_likes_flag1": np.median(g1) if len(g1) > 0 else np.nan,
            "median_likes_flag0": np.median(g0) if len(g0) > 0 else np.nan,
            "Mann_Whitney_p_FDR": p_adj_i,
            "Cliffs_delta": delta,
            "OR_high_engagement": o,
        })
        idx += 1

    tab4b = pd.DataFrame(rows)
    tab4b.to_csv(OUTPUT_DIR / "table4b_engagement_by_unmet_need.csv", index=False)
    return tab4b


# ===========================================================================
# BLOCK 5 — Sentiment + Unmet needs
# ===========================================================================
def block5_sentiment_unmet(df):
    compound = df["sentiment_compound"].fillna(0)
    rows = []
    pvals_mw = []
    pvals_chi = []
    for u in UNMET_NEEDS:
        if u not in df.columns:
            continue
        g1 = compound[df[u] == 1].values
        g0 = compound[df[u] == 0].values
        if len(g1) > 0 and len(g0) > 0:
            _, p = mannwhitneyu(g1, g0, alternative="two-sided")
            pvals_mw.append(p)
        else:
            pvals_mw.append(1.0)
        ct = pd.crosstab(df[u], df["sentiment_class"])
        if ct.size > 0:
            _, p_chi, _, _ = stats.chi2_contingency(ct)
            pvals_chi.append(p_chi)
        else:
            pvals_chi.append(1.0)

    p_adj_mw, _ = fdr_correct(pvals_mw)
    p_adj_chi, _ = fdr_correct(pvals_chi)
    idx = 0
    for u in UNMET_NEEDS:
        if u not in df.columns:
            continue
        g1 = compound[df[u] == 1].values
        g0 = compound[df[u] == 0].values
        rows.append({
            "Unmet need": UNMET_NEED_LABELS.get(u, u),
            "sentiment_compound_flag1": np.median(g1) if len(g1) > 0 else np.nan,
            "sentiment_compound_flag0": np.median(g0) if len(g0) > 0 else np.nan,
            "Mann_Whitney_p_FDR": p_adj_mw[idx] if idx < len(p_adj_mw) else np.nan,
            "Chi2_sentiment_p_FDR": p_adj_chi[idx] if idx < len(p_adj_chi) else np.nan,
        })
        idx += 1
    tab5 = pd.DataFrame(rows)
    tab5.to_csv(OUTPUT_DIR / "table5_sentiment_by_unmet_need.csv", index=False)
    return tab5


# ===========================================================================
# BLOCK 6 — Misinformation
# ===========================================================================
def block6_misinformation(df):
    misinfo = df["misinformation_any"].fillna(0).astype(int)
    likes = pd.to_numeric(df["like_count"], errors="coerce").fillna(0)
    high_eng = df["high_engagement"].fillna(0).astype(int)

    # Prevalence
    n_mis = int(misinfo.sum())
    pct_mis = 100 * n_mis / len(df)

    # Misinfo types if available
    misinfo_types = [c for c in df.columns if c.startswith("misinfo_") and c != "misinformation_any" and np.issubdtype(df[c].dtype, np.number)]
    type_breakdown = {}
    for t in misinfo_types:
        type_breakdown[t] = int(df[t].sum())

    # Engagement: misinfo=1 vs 0
    g1 = likes[misinfo == 1].values
    g0 = likes[misinfo == 0].values
    _, p_mw = mannwhitneyu(g1, g0, alternative="two-sided") if len(g1) > 0 and len(g0) > 0 else (np.nan, np.nan)

    # OR high_engagement
    ct = pd.crosstab(misinfo, high_eng)
    or_high = (ct.loc[1, 1] * ct.loc[0, 0]) / (ct.loc[1, 0] * ct.loc[0, 1] + 1e-9) if ct.shape == (2, 2) else np.nan

    # Misinfo by topic
    ct_topic = pd.crosstab(df["topic_10"], misinfo)
    _, p_topic, _, _ = stats.chi2_contingency(ct_topic)

    # Misinfo by each unmet need
    pvals_unmet = []
    for u in UNMET_NEEDS:
        if u not in df.columns:
            continue
        ct = pd.crosstab(df[u], misinfo)
        if ct.size > 0:
            _, p, _, _ = stats.chi2_contingency(ct)
            pvals_unmet.append(p)
        else:
            pvals_unmet.append(1.0)
    p_adj_unmet, _ = fdr_correct(pvals_unmet)

    rows6 = [
        {"Metric": "misinformation_any N", "Value": n_mis},
        {"Metric": "misinformation_any %", "Value": f"{pct_mis:.1f}%"},
    ]
    for t, n in type_breakdown.items():
        rows6.append({"Metric": f"misinfo_type_{t}", "Value": n})
    rows6.extend([
        {"Metric": "median_likes_misinfo1", "Value": np.median(g1) if len(g1) > 0 else np.nan},
        {"Metric": "median_likes_misinfo0", "Value": np.median(g0) if len(g0) > 0 else np.nan},
        {"Metric": "Mann_Whitney_p", "Value": p_mw},
        {"Metric": "OR_high_engagement", "Value": or_high},
        {"Metric": "chi2_misinfo_by_topic_p", "Value": p_topic},
    ])
    tab6 = pd.DataFrame(rows6)
    tab6.to_csv(OUTPUT_DIR / "table6_misinformation.csv", index=False)

    # Multivariable logistic regression
    if HAS_LOGIT and misinfo.sum() >= 10:
        pred_cols = ["misinformation_vulnerability", "trigger_uncertainty", "treatment_failure", "airway_fear", "emotional_distress"]
        pred_cols = [c for c in pred_cols if c in df.columns][:5]
        X = df[pred_cols + ["comment_length_tokens", "year"]].fillna(0)
        X = sm.add_constant(X)
        y = misinfo
        try:
            model = Logit(y, X).fit(disp=0)
            ors = pd.DataFrame({
                "predictor": model.params.index,
                "OR": np.exp(model.params),
                "CI_lower": np.exp(model.conf_int()[0]),
                "CI_upper": np.exp(model.conf_int()[1]),
                "p_value": model.pvalues,
            })
            ors.to_csv(OUTPUT_DIR / "table6b_misinfo_logistic_OR.csv", index=False)
        except Exception:
            pass
    return tab6


# ===========================================================================
# BLOCK 7 — Treatments
# ===========================================================================
def block7_treatments(df):
    treat_counts = {}
    for c in TREATMENT_COLS:
        if c in df.columns:
            treat_counts[TREATMENT_LABELS.get(c, c)] = int(df[c].sum())
    n_total = len(df)
    rows = []
    for name, n in sorted(treat_counts.items(), key=lambda x: -x[1]):
        per_1k = 1000 * n / n_total
        rows.append({"Treatment": name, "N_mentions": n, "Per_1000_comments": f"{per_1k:.1f}"})
    tab7 = pd.DataFrame(rows)
    tab7.to_csv(OUTPUT_DIR / "table7_treatments.csv", index=False)
    return tab7


# ===========================================================================
# BLOCK 8 — Time series
# ===========================================================================
def block8_timeseries(df):
    df = df.copy()
    df["year_month"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01", errors="coerce")
    df = df.dropna(subset=["year_month"])

    monthly = df.groupby("year_month").agg({
        "comment_id": "count",
        "airway_fear": "mean",
        "treatment_failure": "mean",
        "emotional_distress": "mean",
        "misinformation_any": "mean",
    }).reset_index()
    monthly.columns = ["year_month", "n_comments", "airway_fear", "treatment_failure", "emotional_distress", "misinformation_any"]
    monthly.to_csv(OUTPUT_DIR / "timeseries_monthly.csv", index=False)

    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax1, ax2 = axes
        ax1.bar(monthly["year_month"], monthly["n_comments"], color="steelblue", alpha=0.8, width=20)
        ax1.set_ylabel("Monthly comment count")
        ax1.set_title("Panel A: Monthly Comment Volume")

        for col, label in [
            ("airway_fear", "Airway fear"),
            ("treatment_failure", "Treatment failure"),
            ("emotional_distress", "Emotional distress"),
            ("misinformation_any", "Misinformation"),
        ]:
            ax2.plot(monthly["year_month"], monthly[col] * 100, label=label, linewidth=2)
        ax2.set_ylabel("Prevalence (%)")
        ax2.set_xlabel("Date")
        ax2.set_title("Panel B: Monthly Prevalence of Key Unmet Needs and Misinformation")
        ax2.legend(loc="upper right", fontsize=9)
        ax2.set_ylim(0, None)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "figure2_time_dynamics.png", dpi=150, bbox_inches="tight")
        plt.close()
    return monthly


# ===========================================================================
# BLOCK 9 — Co-occurrence network
# ===========================================================================
def block9_cooccurrence(df):
    domains = [u for u in UNMET_NEEDS if u in df.columns]
    n_d = len(domains)
    cooccur = pd.DataFrame(0.0, index=domains, columns=domains)
    for i, d1 in enumerate(domains):
        for j, d2 in enumerate(domains):
            if i == j:
                cooccur.loc[d1, d2] = int(df[d1].sum())
            elif i < j:
                both = int(((df[d1] == 1) & (df[d2] == 1)).sum())
                cooccur.loc[d1, d2] = both
                cooccur.loc[d2, d1] = both

    cooccur.to_csv(OUTPUT_DIR / "unmet_need_cooccurrence_matrix_full.csv")

    # Top 5 pairs
    pairs = []
    for i in range(n_d):
        for j in range(i + 1, n_d):
            pairs.append((domains[i], domains[j], int(cooccur.loc[domains[i], domains[j]])))
    pairs.sort(key=lambda x: -x[2])
    top5 = pd.DataFrame(pairs[:5], columns=["Unmet_need_1", "Unmet_need_2", "Co_occurrence"])
    top5.to_csv(OUTPUT_DIR / "top5_cooccurrence_pairs.csv", index=False)

    G = None
    if HAS_NETWORKX:
        G = nx.Graph()
        for d in domains:
            G.add_node(d, size=int(df[d].sum()))
        for i, d1 in enumerate(domains):
            for j, d2 in enumerate(domains):
                if i < j:
                    both = int(((df[d1] == 1) & (df[d2] == 1)).sum())
                    if both >= 5:
                        G.add_edge(d1, d2, weight=both)
        if HAS_MATPLOTLIB:
            pos = nx.spring_layout(G, seed=42, k=1.5, iterations=50)
            sizes = [max(300, G.nodes[d].get("size", 0) * 0.5) for d in G.nodes()]
            widths = [max(0.5, G[u][v].get("weight", 0) / 8) for u, v in G.edges()]
            plt.figure(figsize=(14, 10))
            nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color="steelblue", alpha=0.85)
            nx.draw_networkx_edges(G, pos, width=widths, alpha=0.5, edge_color="gray")
            nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")
            plt.title("Co-occurrence Network of Unmet Needs\n(node size = prevalence; edge = co-occurrence ≥ 5)")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "figure5_cooccurrence_network.png", dpi=150, bbox_inches="tight")
            plt.close()

    # Degree centrality
    if G is not None and G.number_of_nodes() > 0:
        deg = nx.degree_centrality(G) if G.number_of_edges() > 0 else {d: 0 for d in domains}
        deg_df = pd.DataFrame(list(deg.items()), columns=["Unmet_need", "Degree_centrality"]).sort_values("Degree_centrality", ascending=False)
        deg_df.to_csv(OUTPUT_DIR / "unmet_need_degree_centrality.csv", index=False)
    return cooccur


# ===========================================================================
# FIGURES
# ===========================================================================
def make_figure1_prisma():
    """PRISMA-like flow — placeholder; user fills numbers from pipeline."""
    if not HAS_MATPLOTLIB:
        return
    # Placeholder values; user should replace with actual pipeline counts
    stages = [
        ("Comments extracted via YouTube API", 15000),
        ("Duplicates removed", 12000),
        ("Non-English removed (stage 1+2)", 8000),
        ("Spam/short text removed", 6000),
        ("Final analytic corpus", 5298),
    ]
    fig, ax = plt.subplots(figsize=(8, 6))
    y = len(stages) - 1
    for label, n in stages:
        ax.text(0.1, y, f"{label}: N={n}", fontsize=11, va="center")
        if y > 0:
            ax.annotate("", xy=(0.05, y - 0.3), xytext=(0.05, y + 0.3),
                        arrowprops=dict(arrowstyle="->", color="gray"))
        y -= 1
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(stages))
    ax.axis("off")
    ax.set_title("Figure 1: Data Pipeline (PRISMA-like Flow)\n(Replace N with actual pipeline counts)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure1_prisma_flow.png", dpi=150, bbox_inches="tight")
    plt.close()


def make_figure3_topic_prevalence(df):
    if not HAS_MATPLOTLIB:
        return
    topic_pct = df["topic_10"].value_counts(normalize=True).mul(100)
    fig, ax = plt.subplots(figsize=(10, 6))
    topics = topic_pct.index.tolist()
    ax.barh(range(len(topics)), topic_pct.values, color="steelblue", alpha=0.8)
    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels([t[:50] + "…" if len(str(t)) > 50 else t for t in topics], fontsize=9)
    ax.set_xlabel("Percent of comments")
    ax.set_title("Figure 3: Topic Prevalence (Top 10)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure3_topic_prevalence.png", dpi=150, bbox_inches="tight")
    plt.close()


def make_figure4_topic_evolution(df):
    if not HAS_MATPLOTLIB:
        return
    df = df.copy()
    df["year_month"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01", errors="coerce")
    df = df.dropna(subset=["year_month"])
    pivot = df.pivot_table(index="topic_10", columns="year_month", aggfunc="size", fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=0), axis=1).fillna(0) * 100
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(pivot_pct.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=30)
    ax.set_xticks(range(len(pivot_pct.columns)))
    ax.set_xticklabels([str(c)[:7] for c in pivot_pct.columns], rotation=45)
    ax.set_yticks(range(len(pivot_pct.index)))
    ax.set_yticklabels([str(i)[:40] for i in pivot_pct.index], fontsize=8)
    ax.set_xlabel("Time (year-month)")
    ax.set_ylabel("Topic")
    ax.set_title("Figure 4: Topic Evolution Heatmap (% per period)")
    plt.colorbar(im, ax=ax, label="%")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure4_topic_evolution.png", dpi=150, bbox_inches="tight")
    plt.close()


def make_supp_treatment_bars(df):
    """Supplementary: Treatment mentions bar chart."""
    if not HAS_MATPLOTLIB:
        return
    counts = []
    labels = []
    for c in TREATMENT_COLS:
        if c in df.columns:
            labels.append(TREATMENT_LABELS.get(c, c))
            counts.append(int(df[c].sum()))
    n = len(df)
    per_1k = [1000 * c / n for c in counts]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, per_1k, color="steelblue", alpha=0.8)
    ax.set_ylabel("Mentions per 1,000 comments")
    ax.set_title("Supplementary: Treatment Mentions")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "supp_figure_treatment_mentions.png", dpi=150, bbox_inches="tight")
    plt.close()


def make_supp_sentiment_by_topic(df):
    """Supplementary: Sentiment stacked bars by topic."""
    if not HAS_MATPLOTLIB:
        return
    ct = pd.crosstab(df["topic_10"], df["sentiment_class"])
    for col in ["negative", "neutral", "positive"]:
        if col not in ct.columns:
            ct[col] = 0
    ct = ct[["negative", "neutral", "positive"]]
    ct_pct = ct.div(ct.sum(axis=1), axis=0).mul(100)
    fig, ax = plt.subplots(figsize=(12, 6))
    ct_pct.plot(kind="barh", stacked=True, ax=ax, color=["#d62728", "#7f7f7f", "#2ca02c"], width=0.8)
    ax.set_xlabel("Percent")
    ax.set_ylabel("Topic")
    ax.set_title("Supplementary: Sentiment Distribution by Topic")
    ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "supp_figure_sentiment_by_topic.png", dpi=150, bbox_inches="tight")
    plt.close()


def make_figure6_engagement_misinfo(df):
    if not HAS_MATPLOTLIB:
        return
    likes = pd.to_numeric(df["like_count"], errors="coerce").fillna(0)
    misinfo = df["misinformation_any"].fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(6, 5))
    data = [likes[misinfo == 0].values, likes[misinfo == 1].values]
    bp = ax.boxplot(data, labels=["No misinformation", "Misinformation"], patch_artist=True)
    bp["boxes"][0].set_facecolor("lightblue")
    bp["boxes"][1].set_facecolor("lightcoral")
    ax.set_ylabel("Like count")
    ax.set_title("Figure 6: Engagement (Likes) by Misinformation Status")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure6_engagement_misinfo.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("Loading data...")
    df = load_and_prepare()
    n = len(df)
    print(f"  -> {n} comments loaded")

    print("\nBlock 1 — Corpus characterization")
    block1_corpus(df)

    print("Block 2 — Topics")
    block2_topics(df)

    print("Block 3 — Unmet needs")
    block3_unmet_needs(df)

    print("Block 4 — Engagement + Unmet needs")
    block4_engagement_unmet(df)

    print("Block 5 — Sentiment + Unmet needs")
    block5_sentiment_unmet(df)

    print("Block 6 — Misinformation")
    block6_misinformation(df)

    print("Block 7 — Treatments")
    block7_treatments(df)

    print("Block 8 — Time series")
    block8_timeseries(df)

    print("Block 9 — Co-occurrence network")
    block9_cooccurrence(df)

    print("\nGenerating figures...")
    make_figure1_prisma()
    make_figure3_topic_prevalence(df)
    make_figure4_topic_evolution(df)
    make_figure6_engagement_misinfo(df)
    make_supp_treatment_bars(df)
    make_supp_sentiment_by_topic(df)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
