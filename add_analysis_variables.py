"""
Add analysis variables to comments_with_unmet_needs.csv:
- time_series (year, month)
- sentiment (VADER + RoBERTa compound + label)
- treatment mentions (dictionary NER)
- misinformation (taxonomy: alt_remedy_mention, misinfo_type_*, misinformation_any, causal_certainty)
- engagement (high_engagement = like_count >= 90th percentile)
"""

import re
import pandas as pd
from pathlib import Path

from misinfo_detection import apply_rule_based_pipeline, export_labeling_sample, MISINFO_TYPES

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_CSV = "topic_outputs/comments_with_unmet_needs.csv"
OUTPUT_CSV = "topic_outputs/comments_with_unmet_needs.csv"
TEXT_COL = "clean_text"
DATE_COL = "comment_date"
LIKE_COL = "like_count"

# Sentiment thresholds
SENT_NEG = -0.05
SENT_POS = 0.05

# Treatment groups: drug/brand -> class (for dictionary NER)
# Antihistamines
ANTIHISTAMINES = [
    "cetirizine", "zyrtec", "loratadine", "claritin", "claritan", "fexofenadine",
    "allegra", "hydroxyzine", "diphenhydramine", "benadryl", "chlorpheniramine",
    "levocetirizine", "desloratadine", "allerzet", "allerzet", "allerstat",
    "levocetirizine", "xyzal", "bilastine", "rupatadine", "acrivastine",
    "bendryl", "phenergan", "promethazine", "avil", "piriton",
]
# Steroids
STEROIDS = [
    "prednisone", "prednisolone", "hydrocortisone", "methylprednisolone",
    "dexamethasone", "dex", "dexa", "cortisone", "betamethasone",
]
# H2 blockers
H2_BLOCKERS = ["famotidine", "pepcid", "ranitidine", "cimetidine", "zantac"]
# Biologic
BIOLOGIC = ["omalizumab", "xolair", "zolair", "zolaire"]
# Montelukast
MONTELUKAST = ["montelukast", "singulair"]
# Other common urticaria treatments
PROTON_PUMP = ["omeprazole", "ppi"]
QUERECTIN = ["querectin", "quercetin"]
# Combine into treatment dict for matching
TREATMENT_DICT = {
    "antihistamine": ANTIHISTAMINES,
    "steroid": STEROIDS,
    "h2_blocker": H2_BLOCKERS,
    "biologic": BIOLOGIC,
    "montelukast": MONTELUKAST,
}

# Misinformation: see misinfo_detection.py for full taxonomy and rules


def extract_year_month(ser: pd.Series) -> tuple:
    """Extract year and month from ISO date string."""
    years, months = [], []
    for val in ser:
        try:
            s = str(val)
            if "T" in s:
                part = s.split("T")[0]
            else:
                part = s[:10]
            parts = part.split("-")
            y = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else None
            m = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            years.append(y)
            months.append(m)
        except (ValueError, IndexError, TypeError):
            years.append(None)
            months.append(None)
    return years, months


def match_treatment(text: str) -> dict:
    """Return dict of treatment_group: 1 if mentioned, else 0."""
    if pd.isna(text) or not str(text).strip():
        return {k: 0 for k in TREATMENT_DICT}
    t = " " + str(text).lower() + " "
    out = {}
    for group, terms in TREATMENT_DICT.items():
        found = any(re.search(r"\b" + re.escape(term) + r"\b", t) for term in terms)
        out[group] = 1 if found else 0
    return out


def main():
    print("Loading...")
    df = pd.read_csv(INPUT_CSV)
    n = len(df)
    print(f"  -> {n} rows from {INPUT_CSV}")
    drop_cols = ["sentiment_compound", "sentiment_label", "sentiment_vader_compound",
                 "sentiment_vader_label", "sentiment_roberta_compound", "sentiment_roberta_label"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    # Drop old misinfo columns to replace with taxonomy
    for col in ["misinformation", "alt_remedy_mention", "misinformation_any", "causal_certainty"] + list(MISINFO_TYPES):
        if col in df.columns:
            df = df.drop(columns=[col])

    # 1. TIME SERIES
    print("Adding year, month...")
    years, months = extract_year_month(df[DATE_COL])
    df["year"] = years
    df["month"] = months

    # 2. SENTIMENT (VADER + RoBERTa for comparison)
    texts_sent = df[TEXT_COL].fillna("").astype(str).tolist()
    if HAS_VADER:
        print("Computing sentiment (VADER)...")
        analyzer = SentimentIntensityAnalyzer()
        compounds = []
        labels = []
        for t in texts_sent:
            s = analyzer.polarity_scores(t)
            c = s["compound"]
            compounds.append(round(c, 4))
            if c <= SENT_NEG:
                labels.append("negative")
            elif c >= SENT_POS:
                labels.append("positive")
            else:
                labels.append("neutral")
        df["sentiment_vader_compound"] = compounds
        df["sentiment_vader_label"] = labels
    else:
        print("  VADER not installed (pip install vaderSentiment) — skipping")
        df["sentiment_vader_compound"] = None
        df["sentiment_vader_label"] = None

    if HAS_TRANSFORMERS:
        print("Computing sentiment (RoBERTa)...")
        try:
            pipe = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                top_k=None,
                truncation=True,
                max_length=512,
            )
            compounds_roberta = []
            labels_roberta = []
            batch_size = 32
            for i in range(0, len(texts_sent), batch_size):
                batch = texts_sent[i : i + batch_size]
                batch = [t[:512] if len(t) > 512 else t for t in batch]
                out = pipe(batch)
                for row in out:
                    scores = {x["label"].lower(): x["score"] for x in row}
                    pos = scores.get("positive", 0)
                    neg = scores.get("negative", 0)
                    neu = scores.get("neutral", 0)
                    if neg >= pos and neg >= neu:
                        label = "negative"
                        compound = -neg
                    elif pos >= neg and pos >= neu:
                        label = "positive"
                        compound = pos
                    else:
                        label = "neutral"
                        compound = 0.0
                    compounds_roberta.append(round(compound, 4))
                    labels_roberta.append(label)
            df["sentiment_roberta_compound"] = compounds_roberta
            df["sentiment_roberta_label"] = labels_roberta
            print(f"  RoBERTa label counts: {pd.Series(labels_roberta).value_counts().to_dict()}")
        except Exception as e:
            print(f"  RoBERTa failed: {e}")
            df["sentiment_roberta_compound"] = None
            df["sentiment_roberta_label"] = None
    else:
        print("  transformers not installed (pip install transformers torch) — skipping RoBERTa")
        df["sentiment_roberta_compound"] = None
        df["sentiment_roberta_label"] = None

    # 3. TREATMENT MENTIONS
    print("Extracting treatment mentions...")
    texts = df[TEXT_COL].fillna("").astype(str)
    for group in TREATMENT_DICT:
        col = f"mentioned_{group}"
        df[col] = [match_treatment(t)[group] for t in texts]

    # 4. MISINFORMATION (taxonomy: alt_remedy, misinfo_types, misinformation_any, causal_certainty)
    print("Tagging misinformation (taxonomy)...")
    misinfo_df = apply_rule_based_pipeline(texts)
    for c in misinfo_df.columns:
        df[c] = misinfo_df[c].values
    print(f"  alt_remedy_mention: {df['alt_remedy_mention'].sum()}, misinformation_any: {df['misinformation_any'].sum()}")

    # 5. ENGAGEMENT
    print("Adding high_engagement...")
    likes = pd.to_numeric(df[LIKE_COL], errors="coerce").fillna(0)
    p90 = likes.quantile(0.90)
    df["high_engagement"] = (likes >= p90).astype(int)
    print(f"  90th percentile like_count = {p90:.0f}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")

    # Export stratified labeling sample (600 comments) for manual labeling
    try:
        sample_path = export_labeling_sample(df, text_col=TEXT_COL)
        print(f"Labeling sample: {sample_path}")
    except Exception as e:
        print(f"Could not export labeling sample: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("VARIABLE SUMMARY")
    print("=" * 60)
    print(f"  year: {df['year'].nunique()} unique years")
    print(f"  month: 1-12")
    if HAS_VADER:
        print(f"  sentiment_vader_label: {df['sentiment_vader_label'].value_counts().to_dict()}")
    if "sentiment_roberta_label" in df.columns and df["sentiment_roberta_label"].notna().any():
        print(f"  sentiment_roberta_label: {df['sentiment_roberta_label'].value_counts().to_dict()}")
    for g in TREATMENT_DICT:
        c = df[f"mentioned_{g}"].sum()
        print(f"  mentioned_{g}: {c} ({100*c/n:.1f}%)")
    print(f"  alt_remedy_mention: {df['alt_remedy_mention'].sum()}, misinformation_any: {df['misinformation_any'].sum()} ({100*df['misinformation_any'].mean():.1f}%)")
    print(f"  high_engagement: {df['high_engagement'].sum()} ({100*df['high_engagement'].mean():.1f}%)")


if __name__ == "__main__":
    main()
