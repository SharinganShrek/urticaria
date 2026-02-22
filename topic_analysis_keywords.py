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
    HAS_BERTOPIC = True
except ImportError:
    HAS_BERTOPIC = False

from sklearn.feature_extraction import text as sklearn_text
from sklearn.feature_extraction.text import CountVectorizer

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
# Primary keywords: strong signal. Fallback: weaker, used only when no primary match.
# Keywords derived from unmatched comment analysis to minimize theme_id=0.
TOPIC_THEMES = [
    {  # Topic 1 — Chronic urticaria lived experience and persistence
        "id": 1,
        "name": "Chronic lived experience & persistence",
        "keywords": [
            "chronic", "years", "months", "daily", "every day", "flare", "relapse", "recurring",
            "suffering", "miserable", "ruined", "hopeless", "no cure", "nothing works",
            "for years", "for months", "yrs", "weeks", "problem", "incurable", "continuous",
            "persist", "lasting", "gone", "life", "months", "weeks",
        ],
        "fallback": ["year", "days", "month", "week", "bad", "worse", "tired"],
    },
    {  # Topic 2 — Symptom phenotype: itch, wheals, rash, burning
        "id": 2,
        "name": "Symptom phenotype: itch, wheals, rash, burning",
        "keywords": [
            "itch", "itchy", "itching", "hives", "welts", "wheals", "rash", "bumps", "burning",
            "red", "skin", "bites", "body", "arms", "legs", "painful", "rashes", "lumps",
            "all over", "irritating", "annoying", "scratch",
        ],
        "fallback": ["arms", "legs", "back", "chest", "hard", "woke", "morning", "night"],
    },
    {  # Topic 3 — Angioedema episodes and airway/ER fear
        "id": 3,
        "name": "Angioedema & airway/ER fear",
        "keywords": [
            "angioedema", "swollen lip", "swollen face", "swollen lips", "lip swollen",
            "tongue swelling", "throat closing", "throat", "cant breathe", "can't breathe",
            "breathe", "ER", "emergency", "hospital", "epipen", "anaphylaxis", "intubation",
            "lips", "lip", "swollen", "swelling", "swell", "swelled", "face", "eyes", "mouth",
            "puffy", "puffiness", "upper lip", "lower lip",
        ],
        "fallback": ["tongue", "lips", "lip", "throat", "pregnant"],
    },
    {  # Topic 4 — Drug-induced (ACE inhibitors / BP meds)
        "id": 4,
        "name": "Drug-induced angioedema (ACE/BP meds)",
        "keywords": [
            "lisinopril", "ACE inhibitor", "blood pressure", "BP meds", "medication caused",
            "side effect", "medication", "drug", "prescribed", "pills", "frusemide",
            "fruzolidone", "minoxidil", "finasteride",
        ],
        "fallback": ["medication", "medicine", "drug", "prescribed"],
    },
    {  # Topic 5 — Triggers & causal beliefs
        "id": 5,
        "name": "Triggers & causal beliefs (food, cold, stress, infections)",
        "keywords": [
            "food", "foods", "dairy", "milk", "chocolate", "diet", "histamine", "probiotics",
            "triggers", "triggered", "cold urticaria", "cold air", "cold water", "heat",
            "pressure", "stress", "infection", "virus", "gut", "keto", "gallbladder", "bile",
            "allergy", "allergic", "allergies", "cause", "caused", "causes", "eating", "eat",
            "ate", "exercise", "water", "contact", "air conditioning", "peanut", "salmon",
            "dehydration", "weed", "smoking", "tick", "tickbite", "alphagal", "celiac",
            "autoimmune", "thyroid", "hormone", "mosquito", "strep", "parasite",
        ],
        "fallback": ["cause", "reason", "because", "trigger", "happen", "happened", "happens"],
    },
    {  # Topic 6 — Conventional treatments & symptom control
        "id": 6,
        "name": "Conventional treatments & symptom control",
        "keywords": [
            "antihistamine", "cetirizine", "zyrtec", "loratadine", "claritin", "claritan",
            "fexofenadine", "allegra", "benadryl", "diphenhydramine", "hydroxyzine",
            "prednisone", "steroid", "hydrocortisone", "famotidine", "pepcid", "montelukast",
            "medicine", "treatment", "cream", "vitamin", "vinegar", "apple cider",
            "calamine", "avil", "dex", "dexa", "allergy panel", "test", "blood work",
            "allergist", "doctor", "dr", "sir", "mam", "treat", "treating", "cure",
            "solution", "rid", "get rid", "home remedies", "advice", "suggest",
        ],
        "fallback": ["doctor", "dr", "treatment", "medicine", "test", "help", "tell", "suggest"],
    },
    {  # Topic 7 — Advanced therapy and complex biomedical framing
        "id": 7,
        "name": "Advanced therapy & biomedical (MCAS, Xolair, dermatographism)",
        "keywords": [
            "xolair", "zolair", "zolaire", "omalizumab", "biologic", "injection", "mg",
            "300 mg", "MCAS", "mast cell activation", "dermatographism", "dermatographia",
            "inducible urticaria", "colonoscopy", "querectin",
        ],
        "fallback": ["biologic", "injection", "shots"],
    },
    {  # Topic 8 — Alternative remedies, cure narratives (misinfo-prone)
        "id": 8,
        "name": "Alternative remedies & cure narratives (misinfo-prone)",
        "keywords": [
            "herbal", "herbs", "natural cure", "natural", "detox", "cleanse", "liver cleanse",
            "parasite cleanse", "ayurveda", "ayurvedic", "homeopathy", "iherb", "planet ayurveda",
            "cured", "miracle", "guaranteed", "DM me", "WhatsApp", "order", "buy",
            "apple cider vinegar", "neem oil", "ghee", "black pepper", "patanjali",
            "acapulco plant", "dr berg", "berg",
        ],
        "fallback": ["cured", "cure", "remedy", "natural", "herbal", "detox", "cleanse"],
    },
    {  # Topic 9 — Vaccine narratives (COVID shots)
        "id": 9,
        "name": "Vaccine narratives (COVID shots)",
        "keywords": [
            "covid vaccine", "vaccine", "vaccines", "shot", "shots", "booster", "boosters",
            "pfizer", "moderna", "after vaccine", "triggered my hives", "covid",
        ],
        "fallback": ["vaccine", "booster", "covid"],
    },
    {  # Topic 10 — Non-clinical / meta / praise / religion (engagement-only)
        "id": 10,
        "name": "Non-clinical / meta / praise / religion (engagement)",
        "keywords": [
            "thanks", "thank you", "ty", "tysm", "great video", "informative", "helpful",
            "bless you", "god", "jesus", "please make more", "share link", "compliments",
            "awesome", "love this", "appreciate", "wow", "nice", "good", "hope", "peace",
            "best", "great", "love", "subscribed", "video", "videos",
        ],
        "fallback": ["thanks", "thank", "great", "good", "nice", "love", "hope", "wow", "appreciate"],
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


def keyword_score_for_theme(text: str, theme: dict, use_fallback: bool = False) -> float:
    """Count how many theme keywords (whole-word) appear in text. Case-insensitive.
    If use_fallback, include fallback keywords (lower weight)."""
    if pd.isna(text) or not text:
        return 0.0
    t = " " + text.lower() + " "
    score = 0.0
    for kw in theme["keywords"]:
        k = kw.lower().replace("'", "")
        if re.search(r"\b" + re.escape(k) + r"\b", t):
            score += 1.0
    if use_fallback and "fallback" in theme:
        for kw in theme["fallback"]:
            k = kw.lower().replace("'", "")
            if re.search(r"\b" + re.escape(k) + r"\b", t):
                score += 0.5  # weaker weight for fallback
    return score


def assign_theme_by_keywords(text: str) -> tuple:
    """Return (theme_id, theme_name) with highest keyword score; 0 if no match.
    Two-pass: primary keywords first; if no match, use fallback keywords."""
    best_id, best_name, best_score = 0, "No keyword match", 0.0
    # Pass 1: primary keywords only
    for theme in TOPIC_THEMES:
        s = keyword_score_for_theme(text, theme, use_fallback=False)
        if s > best_score:
            best_score = s
            best_id = theme["id"]
            best_name = theme["name"]
    # Pass 2: if no match, try with fallback keywords
    if best_score == 0:
        for theme in TOPIC_THEMES:
            s = keyword_score_for_theme(text, theme, use_fallback=True)
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
    n_unmatched = (np.array(theme_ids) == 0).sum()
    print(f"  After keyword assignment: {n_unmatched} unmatched ({100*n_unmatched/n:.1f}%)")

    # Embedding fallback: assign remaining unmatched to nearest theme centroid
    if HAS_BERTOPIC and n_unmatched > 0:
        print("  Embedding fallback for unmatched comments...")
        try:
            from sentence_transformers import SentenceTransformer
            emb_model = SentenceTransformer(EMBEDDING_MODEL)
            unmatched_idx = [i for i in range(n) if theme_ids[i] == 0]
            matched_mask = np.array(theme_ids) != 0
            # Centroids per theme (from matched comments)
            theme_embeddings = {tid: [] for tid in range(1, 11)}
            matched_docs = [docs[i] for i in range(n) if matched_mask[i]]
            matched_tids = [theme_ids[i] for i in range(n) if matched_mask[i]]
            if matched_docs:
                all_embs = emb_model.encode(matched_docs, show_progress_bar=False)
                for i, tid in enumerate(matched_tids):
                    theme_embeddings[tid].append(all_embs[i])
                centroids = {}
                for tid in range(1, 11):
                    arr = np.array(theme_embeddings[tid])
                    centroids[tid] = arr.mean(axis=0) if len(arr) > 0 else None
                # Encode unmatched and assign to nearest centroid
                unmatched_docs = [docs[i] for i in unmatched_idx]
                un_embs = emb_model.encode(unmatched_docs, show_progress_bar=False)
                tid_to_name = {t["id"]: t["name"] for t in TOPIC_THEMES}
                for k, i in enumerate(unmatched_idx):
                    vec = un_embs[k]
                    best_tid, best_dist = 0, float("inf")
                    for tid, c in centroids.items():
                        if c is not None:
                            d = float(np.linalg.norm(vec - c))
                            if d < best_dist:
                                best_dist, best_tid = d, tid
                    if best_tid > 0:
                        theme_ids[i] = best_tid
                        theme_names[i] = tid_to_name[best_tid]
                df["theme_id"] = theme_ids
                df["theme_name"] = theme_names
                n_after = (np.array(theme_ids) == 0).sum()
                print(f"  After embedding fallback: {n_after} unmatched ({100*n_after/n:.1f}%)")
        except Exception as e:
            print(f"  Embedding fallback skipped: {e}")

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

    # Table: 10-theme summary with top words and representative comments (final version)
    domain_stop = [
        "hives", "urticaria", "get", "got", "getting", "really", "one", "like",
        "also", "know", "think", "people", "would", "could", "can", "please",
        "thank", "thanks", "help", "need", "want", "im", "ive", "dont", "doesnt",
    ]
    all_stops = list(sklearn_text.ENGLISH_STOP_WORDS) + domain_stop + HINDI_COMMON_STOP
    cv = CountVectorizer(stop_words=all_stops, max_features=5000)
    summary_rows = []
    for theme in TOPIC_THEMES:
        tid = theme["id"]
        subset = df[df["theme_id"] == tid]
        count = len(subset)
        pct = 100 * count / n
        docs_t = subset[TEXT_COL].fillna("").astype(str).tolist()
        if docs_t:
            try:
                X = cv.fit_transform(docs_t)
                sums = np.asarray(X.sum(axis=0)).flatten()
                idx = np.argsort(-sums)[:N_TOP_WORDS]
                vocab = cv.get_feature_names_out()
                top_words = ", ".join([vocab[i] for i in idx if sums[i] > 0][:N_TOP_WORDS])
            except Exception:
                top_words = ""
            repr_docs = list(subset[TEXT_COL].dropna().astype(str))
            np.random.seed(42)
            if len(repr_docs) > N_REPR_DOCS:
                repr_docs = list(np.random.choice(repr_docs, N_REPR_DOCS, replace=False))
            repr_str = " | ".join([d[:80] + "..." if len(d) > 80 else d for d in repr_docs[:N_REPR_DOCS]])
        else:
            top_words = ""
            repr_str = ""
        summary_rows.append({
            "Theme ID": tid,
            "Theme Name": theme["name"],
            "Size": count,
            "Size (%)": f"{pct:.1f}%",
            "Top Words": top_words,
            "Representative Comments": repr_str,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "table_theme_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Theme summary (final) saved: {summary_path}")

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

    # Save comments with theme_id/theme_name (topic_id_bertopic not used)
    df_out = df.drop(columns=["topic_id_bertopic"], errors="ignore")
    out_csv = OUTPUT_DIR / "comments_with_speaker_gender_and_topics.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"  Comments with topics saved: {out_csv}")

    # Console summary
    print("\n" + "=" * 80)
    print("THEME PREVALENCE (keyword-based)")
    print("=" * 80)
    print(theme_df.to_string(index=False))
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Theme assignment: keyword-based (10 predefined themes) + embedding fallback")
    print(f"  Outputs in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
