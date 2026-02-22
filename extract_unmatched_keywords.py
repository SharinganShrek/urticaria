"""
Extract frequent words from theme_id=0 (unmatched) comments.
Uses same TOPIC_THEMES keyword logic as topic_analysis_keywords.py.
Outputs: top 100-150 topic keyword candidates + sample of unmatched comments for manual review.
"""
import re
import pandas as pd
from pathlib import Path

try:
    from sklearn.feature_extraction import text as sklearn_text
except ImportError:
    sklearn_text = None

# Copy of config and TOPIC_THEMES from topic_analysis_keywords (avoid heavy imports)
INPUT_CSV = "comments_with_speaker_and_gender.csv"
TEXT_COL = "clean_text"
HINDI_COMMON_STOP = ["hai", "se", "ho", "mujhe", "hu", "tulsi", "ki", "bhi", "tha"]

TOPIC_THEMES = [
    {"id": 1, "name": "Chronic lived experience & persistence",
     "keywords": ["chronic", "years", "months", "daily", "every day", "flare", "relapse", "recurring", "suffering", "miserable", "ruined", "hopeless", "no cure", "nothing works", "for 5 years", "for years", "for months"]},
    {"id": 2, "name": "Symptom phenotype: itch, wheals, rash, burning",
     "keywords": ["itch", "itchy", "itching", "hives", "welts", "wheals", "rash", "bumps", "burning", "red", "skin", "bites"]},
    {"id": 3, "name": "Angioedema & airway/ER fear",
     "keywords": ["angioedema", "swollen lip", "swollen face", "tongue swelling", "throat closing", "cant breathe", "can't breathe", "breathe", "ER", "emergency", "hospital", "epipen", "anaphylaxis", "intubation"]},
    {"id": 4, "name": "Drug-induced angioedema (ACE/BP meds)",
     "keywords": ["lisinopril", "ACE inhibitor", "blood pressure", "BP meds", "medication caused", "side effect", "reaction", "medication"]},
    {"id": 5, "name": "Triggers & causal beliefs",
     "keywords": ["food", "dairy", "milk", "chocolate", "diet", "histamine", "probiotics", "what triggers", "triggers", "cold urticaria", "heat", "pressure", "stress", "infection", "virus", "gut", "keto", "gallbladder", "bile"]},
    {"id": 6, "name": "Conventional treatments",
     "keywords": ["antihistamine", "cetirizine", "zyrtec", "loratadine", "claritin", "fexofenadine", "allegra", "benadryl", "diphenhydramine", "hydroxyzine", "prednisone", "steroid", "hydrocortisone", "famotidine", "pepcid", "montelukast"]},
    {"id": 7, "name": "Advanced therapy & biomedical",
     "keywords": ["xolair", "omalizumab", "biologic", "injection", "mg", "300 mg", "MCAS", "mast cell activation", "dermatographism", "dermatographia", "inducible urticaria"]},
    {"id": 8, "name": "Alternative remedies & cure narratives",
     "keywords": ["herbal", "herbs", "natural cure", "detox", "cleanse", "liver cleanse", "ayurveda", "homeopathy", "iherb", "planet ayurveda", "cured", "miracle", "guaranteed", "DM me", "WhatsApp", "order", "buy"]},
    {"id": 9, "name": "Vaccine narratives",
     "keywords": ["covid vaccine", "vaccine", "shot", "booster", "pfizer", "after vaccine", "triggered my hives", "covid"]},
    {"id": 10, "name": "Non-clinical / meta / praise",
     "keywords": ["thanks", "thank you", "great video", "informative", "helpful", "bless you", "god", "jesus", "please make more", "share link", "compliments", "awesome", "love this", "appreciate"]},
]


def keyword_score_for_theme(text: str, theme: dict) -> int:
    if pd.isna(text) or not text:
        return 0
    t = " " + text.lower() + " "
    score = 0
    for kw in theme["keywords"]:
        k = kw.lower().replace("'", "")
        if re.search(r"\b" + re.escape(k) + r"\b", t):
            score += 1
    return score


def assign_theme_by_keywords(text: str) -> tuple:
    best_id, best_name, best_score = 0, "No keyword match", 0
    for theme in TOPIC_THEMES:
        s = keyword_score_for_theme(text, theme)
        if s > best_score:
            best_score, best_id, best_name = s, theme["id"], theme["name"]
    return (best_id, best_name)

OUTPUT_DIR = Path("topic_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Fallback input if speaker file missing
FALLBACK_CSV = "comments_english_only_with_usernames.csv"

# Stop words: common English + domain (hives, urticaria) + contractions + filler
EXTRA_STOP = [
    "hives", "urticaria", "hive", "urticarial",
    "got", "just", "im", "like", "know", "dont", "video", "time", "people", "right",
    "think", "use", "really", "does", "need", "make", "don", "ive", "going", "thing",
    "having", "hi", "feel", "want", "ve", "bit", "long", "say", "yes", "try", "great",
    "did", "plz", "pls", "sure", "way", "lot", "look", "come", "literally", "thats",
    "whats", "guys", "times", "scared", "mom", "sleep", "said", "looks", "ago", "big",
    "doesnt", "didnt", "doing", "person", "trying", "watching", "worst", "rn",
]
STOP_WORDS = set(EXTRA_STOP + HINDI_COMMON_STOP)
if sklearn_text:
    STOP_WORDS |= set(sklearn_text.ENGLISH_STOP_WORDS)


def load_comments():
    """Load comments CSV; prefer speaker file, fallback to english_only."""
    p = Path(INPUT_CSV)
    fallback = Path(FALLBACK_CSV)
    if p.exists():
        df = pd.read_csv(p)
        print(f"  Loaded {INPUT_CSV}")
    elif fallback.exists():
        df = pd.read_csv(fallback)
        print(f"  Loaded {FALLBACK_CSV} (speaker file missing)")
    else:
        raise FileNotFoundError(f"Neither {INPUT_CSV} nor {FALLBACK_CSV} found")
    return df


def get_unmatched_comments(df):
    """Return comments that would have theme_id=0 (no keyword match)."""
    texts = df[TEXT_COL].fillna("").astype(str).tolist()
    unmatched_mask = []
    for text in texts:
        tid, _ = assign_theme_by_keywords(text)
        unmatched_mask.append(tid == 0)
    return df[unmatched_mask].copy(), sum(unmatched_mask)


def tokenize(text: str):
    """Lowercase tokenization; keep only letters (alphanumeric)."""
    if pd.isna(text) or not text:
        return []
    text = text.lower().replace("'", "")
    # Split on non-alpha; keep tokens with letters
    tokens = re.findall(r"[a-z]+", text)
    return [t for t in tokens if len(t) >= 2]


def extract_frequent_words(unmatched_df, min_len=2):
    """Count words in unmatched comments; exclude stop words and short tokens."""
    word_counts = {}
    for text in unmatched_df[TEXT_COL].fillna("").astype(str):
        for w in tokenize(text):
            if len(w) < min_len:
                continue
            if w in STOP_WORDS:
                continue
            word_counts[w] = word_counts.get(w, 0) + 1
    return word_counts


def main():
    print("=" * 60)
    print("Unmatched comments (theme_id=0) — keyword extraction")
    print("=" * 60)

    df = load_comments()
    n_total = len(df)

    unmatched_df, n_unmatched = get_unmatched_comments(df)
    print(f"  Total comments: {n_total}")
    print(f"  Unmatched (theme_id=0): {n_unmatched} ({100 * n_unmatched / n_total:.1f}%)")

    word_counts = extract_frequent_words(unmatched_df)
    sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
    top_terms = sorted_words[:150]  # Top 100–150 as requested

    # Save top 100–150 terms
    keywords_path = OUTPUT_DIR / "unmatched_topic_keywords.csv"
    kw_df = pd.DataFrame(top_terms, columns=["term", "count"])
    kw_df.to_csv(keywords_path, index=False)
    print(f"\n  Top {len(top_terms)} keyword candidates saved: {keywords_path}")

    # Save sample of 30–50 unmatched comment texts for manual review
    sample_size = min(50, max(30, len(unmatched_df)))
    sample_df = unmatched_df.sample(n=sample_size, random_state=42) if len(unmatched_df) >= sample_size else unmatched_df
    sample_path = OUTPUT_DIR / "unmatched_comments_sample.csv"
    sample_df[[TEXT_COL]].to_csv(sample_path, index=False)
    print(f"  Sample of {len(sample_df)} unmatched comments saved: {sample_path}")

    # Console: top 100–150 terms
    print("\n" + "=" * 60)
    print("TOP 100–150 FREQUENT TERMS (potential topic keywords)")
    print("=" * 60)
    for i, (term, cnt) in enumerate(top_terms, 1):
        print(f"  {i:3d}. {term}: {cnt}")


if __name__ == "__main__":
    main()
