# pyright: reportMissingImports=false
import pandas as pd
import re
import emoji
from langdetect import detect_langs, DetectorFactory

# Deterministik sonuçlar için seed
DetectorFactory.seed = 0

# CSV yükle
df = pd.read_csv("comments_deduplicated.csv")

# --------------------
# 1️⃣ TEXT CLEANING
# --------------------

def clean_text(text):
    text = str(text)
    
    # URL kaldır
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Emoji kaldır
    text = emoji.replace_emoji(text, replace='')
    
    # Fazla boşluk temizle
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

df["clean_text"] = df["comment_text"].apply(clean_text)

# --------------------
# 2️⃣ Çok kısa yorumları çıkar (<3 kelime)
# --------------------

df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))
df = df[df["word_count"] >= 3]

# --------------------
# 3️⃣ Dil tespiti
# --------------------

def detect_language(text):
    try:
        if not text or not text.strip():
            return "unknown", 0.0
        langs = detect_langs(text)
        if not langs:
            return "unknown", 0.0
        top = langs[0]
        return top.lang, top.prob
    except Exception:
        return "unknown", 0.0

df[["language", "confidence"]] = df["clean_text"].apply(
    lambda x: pd.Series(detect_language(x))
)

# --------------------
# 4️⃣ Confidence threshold koy
# --------------------
# %80 güven altını çıkar

df = df[df["confidence"] > 0.80]

# --------------------
# 5️⃣ English filtre
# --------------------

english_df = df[df["language"] == "en"]
non_english_df = df[df["language"] != "en"]

# --------------------
# 6️⃣ Kaydet
# --------------------

english_df.to_csv("comments_english_only.csv", index=False)
non_english_df.to_csv("comments_non_english.csv", index=False)

print("English comments:", len(english_df))
print("Non-English comments:", len(non_english_df))