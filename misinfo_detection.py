"""
Publishable misinformation detection pipeline for urticaria/angioedema YouTube comments.

Two distinct concepts:
  A) alt_remedy_mention (0/1): herbs/ayurveda/detox/homeopathy/supplements — NOT automatically misinformation
  B) misinformation_any (0/1): only guideline-contradicting or unsafe claims

Taxonomy (multi-label): misinfo_false_cure, misinfo_unsafe_advice, misinfo_commercial_scam,
  misinfo_strong_causality, misinfo_conspiracy, misinfo_biomedical_falsehood

Includes causal_certainty scoring and hybrid rule-based + supervised approach.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# ALT REMEDY MENTION (herbs, ayurveda, detox, etc.) — DO NOT label as misinfo automatically
# ---------------------------------------------------------------------------
ALT_REMEDY_PATTERNS = [
    r"\bayurved(a|ic)\b", r"\bherbal\b", r"\bherbs?\b", r"\bplant[-\s]?based\b",
    r"\bdetox(ify|ification)?\b", r"\bdetoxifying\b", r"\bhomeopath(y|ic)\b",
    r"\bnatural\s+remed(y|ies)\b", r"\bessential\s+oil\b", r"\bsupplement\b",
    r"\bturmeric\b", r"\bneem\b", r"\bquercetin\b", r"\bvitamin\s+d\b", r"\bomega\s*3\b",
    r"\bprobiotic\b", r"\bchlorella\b", r"\bhumic\s*acid\b", r"\bacapulco\s*plant\b",
    r"\bplanet\s*ayurveda\b", r"\biherb\b", r"\bbermuda\s*grass\b", r"\barugampul\b",
    r"\bturkey\s*berry\b", r"\bsundakkai\b", r"\bkeerai\b",
]

# ---------------------------------------------------------------------------
# MISINFORMATION TAXONOMY — regex rules per type
# ---------------------------------------------------------------------------

# misinfo_false_cure: guaranteed/100% cure claims
MISINFO_FALSE_CURE = [
    r"\bmiracle\s*cure\b", r"\bguaranteed\s*(cure|to\s*cure)\b", r"\b100\s*%\s*cure\b",
    r"\bpermanent\s*cure\b", r"\binstant\s*cure\b", r"\bproven\s*cure\b",
    r"\bsecret\s*remedy\b", r"\bcured\s*me\s*completely\b", r"\bcured\s*it\s*all\b",
    r"\bcured\s*completely\b", r"\bcompletely\s*cured\b", r"\bfound\s*a\s*cure\b",
    r"\bfound\s*the\s*cure\b", r"\bcure\s*for\s*(this|hives|urticaria|allergies)\b",
    r"\bcure\s*(hives|urticaria|allergies)\b",  # "Fasting can cure hives"
    r"\b[tT]ry\s+it\s+.*cured\b", r"\bcured\b.*\b[tT]ry\s+it\b",
    r"\bno\s*more\s*medication\b", r"\bno\s*more\s*meds\b", r"\bno\s*more\s*hives\b",
    r"\ball\s*gone\b.*\b(after|in)\s*(few|a\s*few)\b", r"\bdefinitely\s*cures?\b",
    r"\bwill\s*cure\b", r"\bcan\s*cure\b", r"\bcures?\s*(hives|urticaria)\b",
    r"\bremedy\s*(that|which)\s*cures?\b", r"\bget\s*rid\s*of\s*(it|hives)\s*forever\b",
]

# misinfo_unsafe_advice: stop meds, avoid doctors, avoid epipen, dangerous advice
MISINFO_UNSAFE_ADVICE = [
    r"\bstop\s*(all|your)?\s*meds?\b", r"\bstop\s*(all|your)?\s*medication\b",
    r"\bquit\s*(all|your)?\s*meds?\b", r"\bdon'?t\s*take\s*medication\b",
    r"\bdon'?t\s*take\s*meds\b", r"\bavoid\s*(all\s*)?medication\b",
    r"\bavoid\s*doctors?\b", r"\bdon'?t\s*see\s*(a\s*)?doctor\b",
    r"\bdon'?t\s*go\s*to\s*(the\s*)?doctor\b", r"\bnever\s*take\s*(steroids|prednisone)\b",
    r"\bavoid\s*epipen\b", r"\bdon'?t\s*use\s*epipen\b", r"\bno\s*epipen\b",
    r"\bstop\s*(antihistamines|xolair|steroids)\b", r"\bcome\s*off\s*medication\b",
    r"\bthrow\s*away\s*(your\s*)?meds?\b", r"\bdangerous\s*(to\s*take|medication)\b",
    r"\bmedication\s*(is\s*)?(dangerous|harmful|poison)\b",
]

# misinfo_commercial_scam: DM/WhatsApp/URL/affiliate pitch
MISINFO_COMMERCIAL_SCAM = [
    r"\bDM\s*me\b", r"\bD\.M\.\s*me\b", r"\bwhatsapp\b", r"\bWhatsApp\b",
    r"\border\s*now\b", r"\bbuy\s*now\b", r"\bmessage\s*me\b", r"\bcontact\s*me\b",
    r"\breply\s*here\s*(and|to)\s*(get|i\s*will)\b", r"\bi\s*will\s*give\s*you\b",
    r"\bcheck\s*(out\s*)?(my\s*)?(link|website|channel)\b", r"https?://\S+",
    r"\baffiliate\b", r"\bscam\b", r"\bcon\b", r"\bpromo\s*code\b",
    r"\bdiscount\s*code\b", r"\blink\s*in\s*(bio|description)\b",
    r"\bvisit\s*(my\s*)?(site|website|channel)\b", r"\btext\s*me\s*(for|to)\b",
    r"\bcall\s*me\s*(for|to)\b", r"\bemail\s*me\s*(for|to)\b",
]

# misinfo_strong_causality: high-certainty causal claims (vaccine/drug causes urticaria)
MISINFO_STRONG_CAUSALITY = [
    r"\bvaccine\s*(definitely|certainly|100%|for\s*sure)\s*cause", r"\bcause[sd]\s*(by|from)\s*vaccine\b",
    r"\bvaccine\s*cause[sd]\b", r"\bcovid\s*vaccine\s*cause[sd]\b",
    r"\b(steroids|prednisone|xolair|antihistamine)\s*(definitely|certainly)\s*(cause|destroy|harm)\b",
    r"\b(steroids|prednisone)\s*cause[sd]\s*(chronic|permanent)\b",
    r"\bxolair\s*cause[sd]\b", r"\bantihistamine[s]?\s*cause[sd]\b",
    r"\b100%\s*(caused|caused\s*by)\b", r"\bdefinitely\s*caused\b",
    r"\bfor\s*sure\s*cause[sd]\b", r"\bno\s*doubt\s*(it|that)\s*cause[sd]\b",
    r"\bguaranteed\s*(to\s*)?cause\b", r"\balways\s*cause[sd]\b",
    r"\bnever\s*(helps?|work)\b.*\b(cause|trigger)\b",
]

# misinfo_conspiracy: big pharma, doctors lie, etc.
MISINFO_CONSPIRACY = [
    r"\bbig\s*pharma\b", r"\bpharma\s*hides?\b", r"\bdoctors?\s*lie\b",
    r"\bdoctors?\s*hide\b", r"\bmedical\s*establishment\b", r"\bthey\s*hide\s*the\s*cure\b",
    r"\bhide\s*the\s*cure\b", r"\bconspiracy\b", r"\bcover[-\s]?up\b",
    r"\bno\s*one\s*wants?\s*to\s*find\s*the\s*cure\b", r"\bno\s*cure\s*on\s*purpose\b",
    r"\bprofit\s*from\s*(sickness|illness)\b", r"\bkeep\s*you\s*sick\b",
    r"\bdon'?t\s*want\s*you\s*cured\b", r"\bpharmaceutical\s*conspiracy\b",
]

# misinfo_biomedical_falsehood: false claims about Xolair/antihistamines/steroids as definitely harmful
MISINFO_BIOMEDICAL_FALSEHOOD = [
    r"\b(steroids|prednisone)\s*(are|is)\s*(poison|toxic|destroy)\b",
    r"\bantihistamines?\s*(are|is)\s*(dangerous|harmful|poison)\b",
    r"\bxolair\s*(is|are)\s*(dangerous|harmful|poison|toxic)\b",
    r"\bsteroids\s*destroy\s*(your\s*)?(body|immune|adrenals)\b",
    r"\bantihistamine[s]?\s*destroy\b", r"\bxolair\s*destroy\b",
    r"\bprednisone\s*(will|will\s*never)\s*(help|work)\b",  # definite falsehood
    r"\b(steroids|xolair)\s*don'?t\s*work\s*at\s*all\b",
    r"\ball\s*(pharma|conventional)\s*meds?\s*are\s*poison\b",
    r"\bchemotherapy\s*for\s*(hives|allergies)\b",  # absurd comparison
]

# ---------------------------------------------------------------------------
# CAUSAL CERTAINTY — high vs low
# ---------------------------------------------------------------------------
HIGH_CERTAINTY_CUES = [
    r"\bdefinitely\b", r"\bfor\s*sure\b", r"\b100\s*%\b", r"\bno\s*doubt\b",
    r"\bguaranteed\b", r"\bcauses?\b", r"\bcaused\b", r"\bcausing\b",
    r"\balways\b", r"\bnever\b", r"\bcertainly\b", r"\babsolutely\b",
    r"\bproven\b", r"\bfact\b", r"\bwithout\s*doubt\b",
]
LOW_CERTAINTY_CUES = [
    r"\bmaybe\b", r"\bmight\b", r"\bpossibly\b", r"\bI\s*think\b",
    r"\bseems?\b", r"\bcould\s*be\b", r"\bperhaps\b", r"\bmay\b",
    r"\bnot\s*sure\b", r"\bunclear\b", r"\bwondering\b", r"\bmight\s*be\b",
]

MISINFO_TYPES = [
    "misinfo_false_cure",
    "misinfo_unsafe_advice",
    "misinfo_commercial_scam",
    "misinfo_strong_causality",
    "misinfo_conspiracy",
    "misinfo_biomedical_falsehood",
]
MISINFO_PATTERNS = {
    "misinfo_false_cure": MISINFO_FALSE_CURE,
    "misinfo_unsafe_advice": MISINFO_UNSAFE_ADVICE,
    "misinfo_commercial_scam": MISINFO_COMMERCIAL_SCAM,
    "misinfo_strong_causality": MISINFO_STRONG_CAUSALITY,
    "misinfo_conspiracy": MISINFO_CONSPIRACY,
    "misinfo_biomedical_falsehood": MISINFO_BIOMEDICAL_FALSEHOOD,
}


def _any_match(text: str, patterns: list) -> bool:
    """Return True if any regex in patterns matches text."""
    if pd.isna(text) or not str(text).strip():
        return False
    t = str(text)
    for pat in patterns:
        if re.search(pat, t, re.IGNORECASE):
            return True
    return False


def detect_alt_remedy_mention(text: str) -> int:
    """alt_remedy_mention (0/1): herbs/ayurveda/detox/homeopathy/supplements. NOT misinformation per se."""
    return 1 if _any_match(text, ALT_REMEDY_PATTERNS) else 0


def detect_misinfo_types(text: str) -> dict:
    """Return dict of misinfo_type -> 0/1 for each taxonomy category."""
    out = {k: 0 for k in MISINFO_TYPES}
    if pd.isna(text) or not str(text).strip():
        return out
    for mtype, patterns in MISINFO_PATTERNS.items():
        if _any_match(text, patterns):
            out[mtype] = 1
    return out


def misinformation_any_from_types(types: dict) -> int:
    """misinformation_any = 1 if any misinfo_type is 1."""
    return 1 if any(types.values()) else 0


def compute_causal_certainty(text: str) -> float:
    """
    Causal certainty score for trigger/vaccine/drug causality language.
    Returns: high_cues - low_cues (can be negative).
    High: definitely, for sure, 100%, no doubt, guaranteed, causes, always, never
    Low: maybe, might, possibly, I think, seems, could be
    """
    if pd.isna(text) or not str(text).strip():
        return 0.0
    t = str(text)
    high = sum(1 for p in HIGH_CERTAINTY_CUES if re.search(p, t, re.IGNORECASE))
    low = sum(1 for p in LOW_CERTAINTY_CUES if re.search(p, t, re.IGNORECASE))
    return float(high - low)


def apply_rule_based_pipeline(texts: pd.Series) -> pd.DataFrame:
    """
    Apply rule-based detection to a series of texts.
    Returns DataFrame with columns:
      alt_remedy_mention, misinfo_type_*, misinformation_any, causal_certainty
    """
    alt = [detect_alt_remedy_mention(t) for t in texts]
    types_list = [detect_misinfo_types(t) for t in texts]
    misinfo_any = [misinformation_any_from_types(t) for t in types_list]
    certainty = [compute_causal_certainty(t) for t in texts]

    out = pd.DataFrame({"alt_remedy_mention": alt, "misinformation_any": misinfo_any, "causal_certainty": certainty})
    for mtype in MISINFO_TYPES:
        out[mtype] = [t[mtype] for t in types_list]

    return out


def rule_uncertainty_score(df: pd.DataFrame) -> pd.Series:
    """
    Heuristic for "high uncertainty" (rule conflicts / borderline):
    - alt_remedy_mention=1 but misinformation_any=0 (might be borderline)
    - causal_certainty near 0 with causality-related words
    - multiple misinfo types = 0 but alt_remedy=1
    """
    alt = df["alt_remedy_mention"]
    misinfo = df["misinformation_any"]
    cert = df["causal_certainty"]
    # Alt remedy without rule-based misinfo = uncertain
    uncertain = (alt == 1) & (misinfo == 0)
    uncertain = uncertain | (cert == 0)  # low certainty adds to pool
    return uncertain.astype(int)


def export_labeling_sample(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    output_path: str = "topic_outputs/misinfo_labeling_sample.csv",
    n_rule_positive: int = 200,
    n_rule_negative: int = 200,
    n_uncertain: int = 200,
) -> str:
    """
    Export stratified sample of 600 comments for manual labeling.
    Columns: [existing] + human_misinformation_any, human_misinfo_type_*, human_alt_remedy_mention (empty)
    """
    if "misinformation_any" not in df.columns or "alt_remedy_mention" not in df.columns:
        raise ValueError("DataFrame must have misinformation_any and alt_remedy_mention from rule pipeline.")

    unc = rule_uncertainty_score(df)
    rule_pos = df["misinformation_any"] == 1
    rule_neg = df["misinformation_any"] == 0
    uncertain = unc == 1

    # Stratify
    idx_pos = df.index[rule_pos].tolist()
    idx_neg = df.index[rule_neg & ~uncertain].tolist()
    idx_unc = df.index[rule_neg & uncertain].tolist()

    import random
    rng = random.Random(42)
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)
    rng.shuffle(idx_unc)

    sel_pos = idx_pos[:n_rule_positive]
    sel_neg = idx_neg[:n_rule_negative]
    sel_unc = idx_unc[:n_uncertain]

    # If we don't have enough in a stratum, take from others
    needed_pos = n_rule_positive - len(sel_pos)
    needed_neg = n_rule_negative - len(sel_neg)
    needed_unc = n_uncertain - len(sel_unc)

    if needed_pos > 0:
        extra = [i for i in idx_neg + idx_unc if i not in sel_pos][:needed_pos]
        sel_pos = sel_pos + extra
    if needed_neg > 0:
        extra = [i for i in idx_pos + idx_unc if i not in sel_neg][:needed_neg]
        sel_neg = sel_neg + extra
    if needed_unc > 0:
        extra = [i for i in idx_pos + idx_neg if i not in sel_unc][:needed_unc]
        sel_unc = sel_unc + extra

    all_idx = list(dict.fromkeys(sel_pos + sel_neg + sel_unc))[:600]
    sample_df = df.loc[all_idx].copy()

    # Add empty human label columns
    sample_df["human_misinformation_any"] = ""
    sample_df["human_alt_remedy_mention"] = ""
    for mtype in MISINFO_TYPES:
        sample_df[f"human_{mtype}"] = ""

    sample_df.to_csv(output_path, index=False)
    return output_path


def train_and_apply_model(
    labeled_path: str = "topic_outputs/misinfo_labeling_sample.csv",
    full_df_path: str = "topic_outputs/comments_with_unmet_needs.csv",
    text_col: str = "clean_text",
    output_path: Optional[str] = None,
    precision_target: float = 0.80,
) -> dict:
    """
    After manual labeling: load labeled file, train TF-IDF + Logistic Regression.
    Evaluate on held-out split, choose threshold (precision >= precision_target).
    Apply to full dataset and save with _pred columns and model_confidence.

    Returns dict with metrics.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
    except ImportError:
        raise ImportError("sklearn required: pip install scikit-learn")

    lab = pd.read_csv(labeled_path)
    # Require human_misinformation_any filled
    lab = lab[lab["human_misinformation_any"].notna() & (lab["human_misinformation_any"].astype(str).str.strip() != "")]
    lab["y"] = lab["human_misinformation_any"].astype(str).str.lower().isin(("1", "yes", "true", "y")).astype(int)

    if lab["y"].sum() < 5 or (lab["y"] == 0).sum() < 5:
        raise ValueError("Need at least 5 positive and 5 negative labeled examples.")

    X_text = lab[text_col].fillna("").astype(str)
    y = lab["y"].values

    X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.25, random_state=42, stratify=y)

    from scipy.sparse import hstack

    vec_word = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 3),
        analyzer="word",
        min_df=2,
        sublinear_tf=True,
    )
    vec_char = TfidfVectorizer(
        max_features=4000,
        ngram_range=(2, 5),
        analyzer="char_wb",
        min_df=2,
        sublinear_tf=True,
    )
    X_train_word = vec_word.fit_transform(X_train)
    X_train_char = vec_char.fit_transform(X_train)
    X_train_vec = hstack([X_train_word, X_train_char])
    X_test_vec = hstack([vec_word.transform(X_test), vec_char.transform(X_test)])

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_train_vec, y_train)

    probs = clf.predict_proba(X_test_vec)[:, 1]
    # Threshold search for precision >= target
    best_thresh = 0.5
    best_recall = 0
    for t in np.arange(0.3, 0.95, 0.05):
        pred = (probs >= t).astype(int)
        p = precision_score(y_test, pred, zero_division=0)
        if p >= precision_target and recall_score(y_test, pred, zero_division=0) > best_recall:
            best_recall = recall_score(y_test, pred, zero_division=0)
            best_thresh = t

    pred_test = (probs >= best_thresh).astype(int)
    metrics = {
        "precision": precision_score(y_test, pred_test, zero_division=0),
        "recall": recall_score(y_test, pred_test, zero_division=0),
        "f1": f1_score(y_test, pred_test, zero_division=0),
        "threshold": best_thresh,
        "report": classification_report(y_test, pred_test, zero_division=0),
    }

    # Apply to full dataset
    full = pd.read_csv(full_df_path)
    X_full = hstack([
        vec_word.transform(full[text_col].fillna("").astype(str)),
        vec_char.transform(full[text_col].fillna("").astype(str)),
    ])
    probs_full = clf.predict_proba(X_full)[:, 1]
    full["misinformation_any_pred"] = (probs_full >= best_thresh).astype(int)
    full["misinfo_model_confidence"] = probs_full

    out_path = output_path or full_df_path
    full.to_csv(out_path, index=False)

    return metrics


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Misinformation detection pipeline")
    parser.add_argument("action", choices=["export_sample", "train"], help="Export labeling sample or train model after manual labels")
    parser.add_argument("--labeled", default="topic_outputs/misinfo_labeling_sample.csv", help="Path to manually labeled CSV")
    parser.add_argument("--full", default="topic_outputs/comments_with_unmet_needs.csv", help="Path to full comments CSV")
    parser.add_argument("--text-col", default="clean_text", help="Text column name")
    parser.add_argument("--precision-target", type=float, default=0.80, help="Target precision for threshold")
    args = parser.parse_args()

    if args.action == "export_sample":
        df = pd.read_csv(args.full)
        if "misinformation_any" not in df.columns or "alt_remedy_mention" not in df.columns:
            df = pd.read_csv(args.full)
            misinfo_df = apply_rule_based_pipeline(df[args.text_col])
            for c in misinfo_df.columns:
                df[c] = misinfo_df[c].values
        path = export_labeling_sample(df, text_col=args.text_col, output_path=args.labeled.replace(".csv", "_sample.csv") if "sample" not in args.labeled else args.labeled)
        print(f"Exported: {path}")
    elif args.action == "train":
        m = train_and_apply_model(labeled_path=args.labeled, full_df_path=args.full, text_col=args.text_col, precision_target=args.precision_target)
        print("Metrics:", m)
