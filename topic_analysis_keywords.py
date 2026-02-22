"""
Urticaria YouTube Comments — Topic analysis with 10 predefined themes (keyword-guided).
Uses comments_with_speaker_and_gender.csv.
- Keyword-based theme assignment: each comment scored against 10 topic keyword lists.
- Optional: BERTopic clustering (when available) with same 10 themes as seeds.
Hindi–English mixed: hai, se, ho, mujhe, hu, tulsi, ki, bhi, tha are stop words
so only English terms drive topic representation.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction import text as sklearn_text
    from sklearn.feature_extraction.text import CountVectorizer
    HAS_BERTOPIC = True
except ImportError:
    HAS_BERTOPIC = False

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_CSV = "comments_with_speaker_and_gender.csv"
OUTPUT_DIR = Path("topic_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
TEXT_COL = "clean_text"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MIN_TOPIC_SIZE = 15
N_TOP_WORDS = 10
N_REPR_DOCS = 5

# Hindi/common words to exclude from topic vocabulary (treat as stop words;
# in Hindi–English mixed comments, only English words drive topics)
HINDI_COMMON_STOP = [
    "hai", "se", "ho", "mujhe", "hu", "tulsi", "ki", "bhi", "tha",
]

# 10 predefined topic themes — seed keywords for BERTopic guided modeling
# (and for keyword-based theme assignment)
TOPIC_THEMES = [
    {  # Topic 1 — Chronic urticaria lived experience and persistence
        "id": 1,
        "name": "Chronic lived experience & persistence",
        "keywords": [
            "chronic", "years", "months", "daily", "every day", "flare", "relapse", "recurring",
            "suffering", "miserable", "ruined", "hopeless", "no cure", "nothing works",
            "for 5 years", "for years", "for months",
        ],
    },
    {  # Topic 2 — Symptom phenotype: itch, wheals, rash, burning
        "id": 2,
        "name": "Symptom phenotype: itch, wheals, rash, burning",
        "keywords": [
            "itch", "itchy", "itching", "hives", "welts", "wheals", "rash", "bumps", "burning",
            "red", "skin", "bites",
        ],
    },
    {  # Topic 3 — Angioedema episodes and airway/ER fear
        "id": 3,
        "name": "Angioedema & airway/ER fear",
        "keywords": [
            "angioedema", "swollen lip", "swollen face", "tongue swelling", "throat closing",
            "cant breathe", "can't breathe", "breathe", "ER", "emergency", "hospital",
            "epipen", "anaphylaxis", "intubation",
        ],
    },
    {  # Topic 4 — Drug-induced (ACE inhibitors / BP meds)
        "id": 4,
        "name": "Drug-induced angioedema (ACE/BP meds)",
        "keywords": [
            "lisinopril", "ACE inhibitor", "blood pressure", "BP meds", "medication caused",
            "side effect", "reaction", "medication",
        ],
    },
    {  # Topic 5 — Triggers & causal beliefs
        "id": 5,
        "name": "Triggers & causal beliefs (food, cold, stress, infections)",
        "keywords": [
            "food", "dairy", "milk", "chocolate", "diet", "histamine", "probiotics",
            "what triggers", "triggers", "cold urticaria", "heat", "pressure", "stress",
            "infection", "virus", "gut", "keto", "gallbladder", "bile",
        ],
    },
    {  # Topic 6 — Conventional treatments & symptom control
        "id": 6,
        "name": "Conventional treatments & symptom control",
        "keywords": [
            "antihistamine", "cetirizine", "zyrtec", "loratadine", "claritin", "fexofenadine",
            "allegra", "benadryl", "diphenhydramine", "hydroxyzine", "prednisone", "steroid",
            "hydrocortisone", "famotidine", "pepcid", "montelukast",
        ],
    },
    {  # Topic 7 — Advanced therapy and complex biomedical framing
        "id": 7,
        "name": "Advanced therapy & biomedical (MCAS, Xolair, dermatographism)",
        "keywords": [
            "xolair", "omalizumab", "biologic", "injection", "mg", "300 mg",
            "MCAS", "mast cell activation", "dermatographism", "dermatographia",
            "inducible urticaria",
        ],
    },
    {  # Topic 8 — Alternative remedies, cure narratives, supplement (misinfo-prone)
        "id": 8,
        "name": "Alternative remedies & cure narratives (misinfo-prone)",
        "keywords": [
            "herbal", "herbs", "natural cure", "detox", "cleanse", "liver cleanse",
            "ayurveda", "homeopathy", "iherb", "planet ayurveda", "cured", "miracle",
            "guaranteed", "DM me", "WhatsApp", "order", "buy",
        ],
    },
    {  # Topic 9 — Vaccine narratives (COVID shots)
        "id": 9,
        "name": "Vaccine narratives (COVID shots)",
        "keywords": [
            "covid vaccine", "vaccine", "shot", "booster", "pfizer", "after vaccine",
            "triggered my hives", "covid",
        ],
    },
    {  # Topic 10 — Non-clinical / meta / praise / religion (engagement-only)
        "id": 10,
        "name": "Non-clinical / meta / praise / religion (engagement)",
        "keywords": [
            "thanks", "thank you", "great video", "informative", "helpful", "bless you",
            "god", "jesus", "please make more", "share link", "compliments", "awesome",
            "love this", "appreciate",
        ],
    },
]


def build_seed_topic_list():
    """Seed list for BERTopic: list of lists of keywords (lowercase, single tokens preferred)."""
    seed_list = []
    for theme in TOPIC_THEMES:
        words = []
        for kw in theme["keywords"]:
            # Split phrases into tokens; keep multi-word as single entry for embedding
            tokens = kw.lower().replace("'", "").split()
            words.extend(tokens)
        # Dedupe and limit so each seed topic has a manageable set
        seen = set()
        unique = [w for w in words if w not in seen and not seen.add(w)]
        seed_list.append(unique[:25])  # cap per topic
    return seed_list


def keyword_score_for_theme(text: str, theme: dict) -> int:
    """Count how many theme keywords (as whole-word/substring) appear in text. Case-insensitive."""
    if pd.isna(text) or not text:
        return 0
    t = " " + text.lower() + " "
    score = 0
    for kw in theme["keywords"]:
        k = kw.lower().replace("'", "")
        # whole-word or as phrase
        if re.search(r"\b" + re.escape(k) + r"\b", t):
            score += 1
    return score


def assign_theme_by_keywords(text: str) -> tuple:
    """Return (theme_id, theme_name) with highest keyword score; 0 if no match."""
    best_id, best_name, best_score = 0, "No keyword match", 0
    for theme in TOPIC_THEMES:
        s = keyword_score_for_theme(text, theme)
        if s > best_score:
            best_score = s
            best_id = theme["id"]
            best_name = theme["name"]
    return (best_id, best_name)


def main():
    print("Loading comments...")
    df = pd.read_csv(INPUT_CSV)
    docs = df[TEXT_COL].fillna("").astype(str).tolist()
    n = len(docs)
    print(f"  -> {n} comments from {INPUT_CSV}")

    # Keyword-based theme assignment (each doc -> theme 1-10 or 0)
    theme_ids = []
    theme_names = []
    for i, text in enumerate(docs):
        tid, tname = assign_theme_by_keywords(text)
        theme_ids.append(tid)
        theme_names.append(tname)

    df["theme_id"] = theme_ids
    df["theme_name"] = theme_names

    # Optional: BERTopic (when available)
    if HAS_BERTOPIC:
        domain_stop = [
            "hives", "urticaria", "get", "got", "getting", "really", "one", "like",
            "also", "know", "think", "people", "would", "could", "can", "please",
            "thank", "thanks", "help", "need", "want", "im", "ive", "dont", "doesnt",
        ]
        all_stops = list(sklearn_text.ENGLISH_STOP_WORDS) + domain_stop + HINDI_COMMON_STOP
        vectorizer = CountVectorizer(stop_words=all_stops, max_features=5000)
        seed_topic_list = build_seed_topic_list()
        print("Initializing BERTopic (guided with 10 seed topics)...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        try:
            topic_model = BERTopic(
                embedding_model=embedding_model,
                vectorizer_model=vectorizer,
                min_topic_size=MIN_TOPIC_SIZE,
                verbose=True,
                calculate_probabilities=False,
                nr_topics="auto",
                seed_topic_list=seed_topic_list,
            )
        except TypeError:
            topic_model = BERTopic(
                embedding_model=embedding_model,
                vectorizer_model=vectorizer,
                min_topic_size=MIN_TOPIC_SIZE,
                verbose=True,
                calculate_probabilities=False,
                nr_topics="auto",
            )
        print("Fitting BERTopic (this may take several minutes)...")
        topics_bertopic, probs = topic_model.fit_transform(docs)
        df["topic_id_bertopic"] = [t if t != -1 else "Outlier" for t in topics_bertopic]
        info = topic_model.get_topic_info()
        table_rows = []
        for _, row in info.iterrows():
            tid = row["Topic"]
            count = int(row["Count"])
            pct = 100 * count / n
            topic_words = topic_model.get_topic(tid)
            top_words = ", ".join([w for w, _ in topic_words[:N_TOP_WORDS]]) if topic_words else ("(outliers)" if tid == -1 else "")
            repr_docs = topic_model.get_representative_docs(tid) or []
            repr_str = " | ".join([d[:80] + "..." if len(d) > 80 else d for d in repr_docs[:N_REPR_DOCS]])
            table_rows.append({"Topic ID": tid if tid != -1 else "Outlier", "Size": count, "Size (%)": f"{pct:.1f}%", "Top Words": top_words, "Representative Comments": repr_str})
        table_df = pd.DataFrame(table_rows)
        table_df.to_csv(OUTPUT_DIR / "table2_topic_summary.csv", index=False)
        print(f"  Table 2 (BERTopic) saved: {OUTPUT_DIR / 'table2_topic_summary.csv'}")
    else:
        print("BERTopic not installed (optional). Using keyword-based themes only.")
        df["topic_id_bertopic"] = ""

    # Summary by theme (keyword-based)
    theme_counts = pd.Series(theme_ids).value_counts().sort_index()
    theme_rows = []
    for theme in TOPIC_THEMES:
        tid = theme["id"]
        count = theme_counts.get(tid, 0)
        pct = 100 * count / n
        theme_rows.append({
            "Theme ID": tid,
            "Theme Name": theme["name"],
            "Count": count,
            "Percent": f"{pct:.1f}%",
        })
    theme_rows.append({
        "Theme ID": 0,
        "Theme Name": "No keyword match",
        "Count": theme_counts.get(0, 0),
        "Percent": f"{100 * theme_counts.get(0, 0) / n:.1f}%",
    })
    theme_df = pd.DataFrame(theme_rows)
    theme_path = OUTPUT_DIR / "table_theme_by_keywords.csv"
    theme_df.to_csv(theme_path, index=False)
    print(f"  Theme (keyword) summary saved: {theme_path}")

    # Figure: theme prevalence (keyword-based)
    plot_df = theme_df[theme_df["Theme ID"] > 0].copy()
    if len(plot_df) > 0 and HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(12, max(6, len(plot_df) * 0.4)))
        y_pos = np.arange(len(plot_df))
        labels = [f"T{r['Theme ID']}: " + (r["Theme Name"][:50] + "..." if len(r["Theme Name"]) > 50 else r["Theme Name"]) for _, r in plot_df.iterrows()]
        counts = plot_df["Count"].values
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(plot_df)))
        ax.barh(y_pos, counts, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Number of comments")
        ax.set_title("Topic themes (keyword-based assignment)")
        ax.invert_yaxis()
        plt.tight_layout()
        fig_path = OUTPUT_DIR / "figure_theme_prevalence.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Figure saved: {fig_path}")
    elif len(plot_df) > 0 and not HAS_MATPLOTLIB:
        print("  Skipping figure (matplotlib not installed)")

    # Save comments with both BERTopic topic and theme_id/theme_name
    out_csv = OUTPUT_DIR / "comments_with_speaker_gender_and_topics.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Comments with topics saved: {out_csv}")

    # Console summary
    print("\n" + "=" * 80)
    print("THEME PREVALENCE (keyword-based)")
    print("=" * 80)
    print(theme_df.to_string(index=False))
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if HAS_BERTOPIC and "topic_id_bertopic" in df.columns and (df["topic_id_bertopic"] != "").any():
        # BERTopic was run: topic_id is int or "Outlier"
        bt = df["topic_id_bertopic"]
        n_out = (bt == "Outlier").sum()
        n_bertopic = n - n_out
        print(f"  BERTopic topics (excluding outliers): {n_bertopic}")
        print(f"  BERTopic outliers: {n_out} ({100*n_out/n:.1f}%)")
    print(f"  Theme assignment: keyword-based (10 predefined themes)")
    print(f"  Outputs in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
