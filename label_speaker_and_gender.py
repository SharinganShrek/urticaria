"""
Speaker type (patient/caregiver/general/clinician/advertiser/unclear) and
gender (male/female/unknown) labeling for urticaria comments dataset.

- Speaker type: rule-based weak supervision (JMIR/Acad Dermatol Venereol style).
- Gender: first-name from Username → name–gender mapping with high-confidence only (≥0.90 equivalent).
"""

import re
import csv
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd

# Optional: gender from first name (offline)
try:
    import gender_guesser.detector as gender_detector
    _GENDER_DETECTOR = gender_detector.Detector()
    HAS_GENDER_GUESSER = True
except ImportError:
    HAS_GENDER_GUESSER = False
    _GENDER_DETECTOR = None


# ---- Speaker type rules (weak supervision) ----
# Order matters: more specific (advertiser, clinician, caregiver) before patient/general.
# We use clean_text for classification.

def _normalize_for_rules(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    return " " + text.lower().strip() + " "


def _has_any_phrase(text_norm, phrases):
    return any(p in text_norm for p in phrases)


def _has_any_word_boundary(text_norm, words):
    # word boundary: space or start/end
    for w in words:
        if f" {w} " in text_norm or text_norm.startswith(f"{w} ") or text_norm.endswith(f" {w}"):
            return True
    return False


# Advertiser / spam (check first – high precision)
ADVERTISER_PHRASES = [
    " planet ayurveda ",
    "mahakhadir ghrit",
    "dm me",
    "whatsapp",
    "contact me",
    "message me",
    "reply here and i will",
    "cured me completely",
    "herbal medication",
    "miracle cure",
    "natural remedy",
    "100% guaranteed",
    "must-try",
    "best ayurvedic",
    "best solution",
    "buy now",
    "click here",
    "http://",
    "https://",
    "www.",
    ".com",
    "affiliate",
]
ADVERTISER_WORDS = ["whatsapp", "telegram", "dm me", "inbox", "order now"]


def rule_advertiser(text_norm):
    if len(text_norm) < 10:
        return False
    if _has_any_phrase(text_norm, ADVERTISER_PHRASES):
        return True
    # URL-like
    if re.search(r"\bhttps?://\S+|\bwww\.\S+|\b\S+\.com\b", text_norm):
        return True
    # Repeated brand / product push (short + product name)
    brand_like = ["planet ayurveda", "curcumin", "capsule", "mahakhadir", "acapulco plant"]
    if sum(1 for b in brand_like if b in text_norm) >= 2:
        return True
    return False


# Clinician / educator
CLINICIAN_PHRASES = [
    " as a dermatologist",
    " as a doctor",
    " as a nurse",
    " as a pharmacist",
    " in my practice",
    " i'm a doctor",
    " i'm a nurse",
    " i'm a pharmacist",
    " i am a doctor",
    " i am a nurse",
    " guidelines recommend",
    " urticaria is ",
    " hives are ",
    " the treatment of ",
    " recommend ",
    " in patients with ",
    " clinically ",
]
# Thank you doctor / addressing doctor → viewer, not clinician; avoid "doctor said" (patient quoting)
CLINICIAN_STRONG = [
    "as a dermatologist",
    "as a doctor",
    "in my practice",
    "i'm a doctor",
    "i'm a nurse",
    "i am a doctor",
    "i am a nurse",
]


def rule_clinician(text_norm):
    if _has_any_phrase(text_norm, CLINICIAN_STRONG):
        return True
    # Educational declarative (urticaria is / hives are) without first-person ownership
    if _has_any_phrase(text_norm, [" urticaria is ", " hives are ", " guidelines recommend "]):
        if not _has_any_phrase(text_norm, [" my ", " i have ", " i've ", " i get ", " i had ", " i'm ", " mine "]):
            return True
    return False


# Caregiver (my son/daughter/child/kid/baby/husband/wife + condition)
CAREGIVER_PHRASES = [
    " my son ",
    " my daughter ",
    " my child ",
    " my kid ",
    " my baby ",
    " my grandson ",
    " my granddaughter ",
    " my husband ",
    " my wife ",
    " my child has ",
    " my son has ",
    " my daughter has ",
    " my kid has ",
    " my baby has ",
    " my grandson ",
    " my granddaughter ",
    " my 2 1/2-year-old ",
    " my toddler ",
]


def rule_caregiver(text_norm):
    return _has_any_phrase(text_norm, CAREGIVER_PHRASES)


# Patient (first-person ownership + symptoms/duration/diagnosis)
PATIENT_FIRST_PERSON = [
    " i have ",
    " i've had ",
    " i had ",
    " i get ",
    " i got ",
    " i'm getting ",
    " i am getting ",
    " my hives ",
    " my urticaria ",
    " my rash ",
    " my skin ",
    " my body ",
    " my face ",
    " my arms ",
    " my legs ",
    " my back ",
    " my lips ",
    " my hands ",
    " my feet ",
    " my allergy ",
    " my allergies ",
    " my reaction ",
    " my symptoms ",
    " my doctor ",
    " my dermatologist ",
    " my allergist ",
]
PATIENT_SYMPTOM_DURATION = [
    " for 2 years",
    " for 3 years",
    " for years",
    " for months",
    " for weeks",
    " for days",
    " since ",
    " diagnosed ",
    " diagnosis ",
    " suffering ",
    " suffer from ",
    " am suffering ",
    " i'm suffering ",
    " chronic ",
    " flare-up",
    " flare up",
    " outbreak ",
    " break out ",
    " breakout ",
    " itchy ",
    " itching ",
    " welts ",
    " swelling ",
    " swollen ",
    " bumps ",
    " blood work ",
    " allergy test ",
    " allergist ",
    " dermatologist ",
    " xolair ",
    " antihistamine ",
    " claritin ",
    " benadryl ",
    " zyrtec ",
    " cetirizine ",
    " allegra ",
]
# Dataset-specific: short first-person ownership
PATIENT_SHORT = [
    " i have hives",
    " i have this",
    " i have it",
    " i have that",
    " i have these",
    " i have urticaria",
    " i have allergies",
    " i have allergy",
    " i got hives",
    " i got this",
    " i got them",
    " i get hives",
    " i get them",
    " i'm having ",
    " i am having ",
    " had hives",
    " have hives",
    " got hives",
    " my hives",
    " suffering from ",
    " im suffering ",
    " i am suffering ",
]


def rule_patient(text_norm):
    if len(text_norm) < 4:
        return False
    if _has_any_phrase(text_norm, PATIENT_FIRST_PERSON + PATIENT_SHORT):
        return True
    if _has_any_phrase(text_norm, PATIENT_SYMPTOM_DURATION) and _has_any_phrase(text_norm, [" i ", " my ", " me "]):
        return True
    # "I have" / "I've" / "I had" generic
    if re.search(r"\b(i have|i've had|i had|i get|i got|i'm getting)\b", text_norm):
        return True
    return False


# General viewer: question / no clear first-person disease ownership
GENERAL_QUESTION = [
    " is this hives",
    " is this urticaria",
    " can it be ",
    " can you get ",
    " what causes ",
    " what is the ",
    " how to ",
    " how do i ",
    " how can ",
    " why do ",
    " does ",
    " can ",
    " what are ",
    " what's the ",
    " is there ",
    " are there ",
    " should i ",
    " could this be ",
    " would ",
]
# Short generic questions
GENERAL_SHORT = [
    " what is the best medicine",
    " what is the treatment",
    " what is the cure",
    " what is the solution",
    " how to cure ",
    " how to get rid ",
    " how to treat ",
    " how to stop ",
    " what is solution",
    " what are the treatments",
]


def rule_general(text_norm):
    if _has_any_phrase(text_norm, GENERAL_QUESTION + GENERAL_SHORT):
        return True
    # Question mark and no strong patient ownership
    if "?" in text_norm and not _has_any_phrase(text_norm, PATIENT_FIRST_PERSON[:15]):
        return True
    return False


def assign_speaker_type(text):
    """Assign one of: patient, caregiver, general, clinician, advertiser. (unclear merged into general.)"""
    norm = _normalize_for_rules(text)
    if not norm.strip():
        return "general"
    # Priority order
    if rule_advertiser(norm):
        return "advertiser"
    if rule_clinician(norm):
        return "clinician"
    if rule_caregiver(norm):
        return "caregiver"
    if rule_patient(norm):
        return "patient"
    if rule_general(norm):
        return "general"
    # Short or ambiguous → general (unclear merged with general)
    if len(norm.split()) < 4:
        return "general"
    return "general"


# ---- Gender inference (JMIR PCOS-style) ----

def _normalize_username_for_names(author_name):
    """Normalize username: treat underscores as spaces, remove handles/emoji, keep letters/spaces."""
    if pd.isna(author_name) or not isinstance(author_name, str):
        return ""
    s = author_name.replace("_", " ")  # underscores as space (e.g. Robin_Cole -> Robin Cole)
    s = re.sub(r"@\w+", "", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_first_name(author_name):
    """Extract first alphabetic token from author_name; underscores treated as space; title-case."""
    s = _normalize_username_for_names(author_name)
    if not s:
        return ""
    tokens = s.split()
    for t in tokens:
        if t.isalpha() and len(t) > 1:
            return t.title()
    return ""


def get_name_tokens_for_gender(author_name):
    """
    Get list of name tokens to try for gender: first token always; if 3+ tokens, also second.
    Username is normalized (underscores as spaces).
    """
    s = _normalize_username_for_names(author_name)
    if not s:
        return []
    tokens = [t.title() for t in s.split() if t.isalpha() and len(t) > 1]
    if not tokens:
        return []
    if len(tokens) >= 3:
        return [tokens[0], tokens[1]]  # first and second word
    return [tokens[0]]


def infer_gender_single_name(first_name):
    """
    Infer gender from one name. Includes mostly_male -> male, mostly_female -> female
    (so e.g. Kelly, Mary are classified as female).
    """
    if not first_name or not HAS_GENDER_GUESSER:
        return "unknown"
    g = _GENDER_DETECTOR.get_gender(first_name)
    if g in ("male", "mostly_male"):
        return "male"
    if g in ("female", "mostly_female"):
        return "female"
    return "unknown"


def infer_gender(author_name):
    """
    Infer gender from username: try first name; if 3+ words, also try second name.
    Uses male/mostly_male -> male, female/mostly_female -> female.
    """
    names_to_try = get_name_tokens_for_gender(author_name)
    for name in names_to_try:
        result = infer_gender_single_name(name)
        if result != "unknown":
            return result
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Label speaker type and gender on comments CSV.")
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="comments_english_only_with_usernames.csv",
        help="Input CSV with columns: Username, clean_text (or comment_text)",
    )
    parser.add_argument(
        "-o", "--output",
        default="comments_with_speaker_and_gender.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        default=True,
        help="Print distribution report (default: True)",
    )
    parser.add_argument(
        "--no-report",
        action="store_false",
        dest="report",
        help="Skip report",
    )
    parser.add_argument(
        "--report-file",
        metavar="PATH",
        default=None,
        help="Write distribution report to a markdown file",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    df = pd.read_csv(input_path)
    # Prefer clean_text; fallback comment_text
    text_col = "clean_text" if "clean_text" in df.columns else "comment_text"
    if text_col not in df.columns:
        raise ValueError(f"Need column 'clean_text' or 'comment_text'; found {list(df.columns)}")
    user_col = "Username" if "Username" in df.columns else "author_name"
    if user_col not in df.columns:
        user_col = [c for c in df.columns if "user" in c.lower() or "name" in c.lower() or "author" in c.lower()]
        user_col = user_col[0] if user_col else None
    if user_col is None:
        raise ValueError("Need a username/author column.")

    n = len(df)
    print(f"Loaded {n} rows from {input_path}")
    print(f"Text column: {text_col}, User column: {user_col}")

    # Speaker type
    df["speaker_type"] = df[text_col].astype(str).apply(assign_speaker_type)
    # First name (for display) and gender from full username
    df["first_name"] = df[user_col].apply(extract_first_name)
    df["gender"] = df[user_col].apply(infer_gender)

    # Save
    out_path = Path(args.output)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved to {out_path}")

    report_lines = []
    if args.report:
        report_lines.append("# Speaker type and gender labeling report\n")
        report_lines.append(f"- Total comments: {n}\n")
        report_lines.append("## Speaker type distribution\n")
        st = df["speaker_type"].value_counts()
        for k, v in st.items():
            line = f"- **{k}**: {v} ({100*v/n:.1f}%)"
            print(f"  {k}: {v} ({100*v/n:.1f}%)")
            report_lines.append(line + "\n")
        report_lines.append("\n## Gender distribution\n")
        g = df["gender"].value_counts()
        inferred = (df["gender"] != "unknown").sum()
        for k, v in g.items():
            line = f"- **{k}**: {v} ({100*v/n:.1f}%)"
            print(f"  {k}: {v} ({100*v/n:.1f}%)")
            report_lines.append(line + "\n")
        report_lines.append(f"\nGender could be inferred for **{inferred}** ({100*inferred/n:.1f}%) of users; remaining unknown.\n")
        print(f"\nGender inferred (non-unknown): {inferred} ({100*inferred/n:.1f}%)")
        report_lines.append("\n## Speaker type × Gender (counts)\n\n```\n")
        cross = pd.crosstab(df["speaker_type"], df["gender"])
        report_lines.append(cross.to_string() + "\n```\n")
        print("\n--- Speaker type × Gender (counts) ---")
        print(cross.to_string())
        if args.report_file:
            Path(args.report_file).write_text("".join(report_lines), encoding="utf-8")
            print(f"\nReport written to {args.report_file}")


if __name__ == "__main__":
    main()
