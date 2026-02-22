"""
Urticaria/Angioedema YouTube Comments — BERTopic Topic Modeling
Step 2 of analysis roadmap: Topic discovery
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# BERTopic
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction import text as sklearn_text
from sklearn.feature_extraction.text import CountVectorizer

# Optional: reduce verbosity
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_CSV = "comments_english_only.csv"
OUTPUT_DIR = Path("topic_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Use clean_text (pre-cleaned) or comment_text
TEXT_COL = "clean_text"

# Lightweight, fast embedding model (alternatives: all-mpnet-base-v2 for quality)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Min topic size (HDBSCAN min_cluster_size) — smaller = more topics
MIN_TOPIC_SIZE = 15

# Number of top words per topic
N_TOP_WORDS = 10
N_REPR_DOCS = 5

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading comments...")
df = pd.read_csv(INPUT_CSV)
docs = df[TEXT_COL].fillna("").astype(str).tolist()
print(f"  → {len(docs)} comments loaded")

# ---------------------------------------------------------------------------
# BERTopic setup
# ---------------------------------------------------------------------------
# Vectorizer: exclude very common domain terms to get clinically meaningful topics
domain_stop = [
    "hives", "urticaria", "get", "got", "getting", "really", "one", "like",
    "also", "know", "think", "people", "would", "could", "can", "please",
    "thank", "thanks", "help", "need", "want", "im", "ive", "dont", "doesnt",
]
all_stops = list(sklearn_text.ENGLISH_STOP_WORDS) + domain_stop
vectorizer = CountVectorizer(stop_words=all_stops, max_features=5000)

print("Initializing BERTopic (embedding model + vectorizer)...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
topic_model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=vectorizer,
    min_topic_size=MIN_TOPIC_SIZE,
    verbose=True,
    calculate_probabilities=False,
    nr_topics="auto",  # merge similar topics
)

# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
print("Fitting BERTopic (this may take several minutes)...")
topics, probs = topic_model.fit_transform(docs)

# ---------------------------------------------------------------------------
# Add topic_id to dataframe
# ---------------------------------------------------------------------------
df["topic_id"] = topics
df["topic_id"] = df["topic_id"].replace(-1, "Outlier")  # -1 = outlier in HDBSCAN

# ---------------------------------------------------------------------------
# Table 2: Topic summary
# ---------------------------------------------------------------------------
info = topic_model.get_topic_info()

table_rows = []
for _, row in info.iterrows():
    tid = row["Topic"]
    count = int(row["Count"])
    pct = 100 * count / len(df)
    
    # Get top words
    topic_words = topic_model.get_topic(tid)
    if topic_words:
        top_words = ", ".join([w for w, _ in topic_words[:N_TOP_WORDS]])
    else:
        top_words = "(outliers)" if tid == -1 else ""
    
    # Representative comments
    repr_docs = topic_model.get_representative_docs(tid)
    if repr_docs:
        repr_short = repr_docs[:N_REPR_DOCS]
        repr_str = " | ".join([d[:80] + "..." if len(d) > 80 else d for d in repr_short])
    else:
        repr_str = ""
    
    table_rows.append({
        "Topic ID": tid if tid != -1 else "Outlier",
        "Size": count,
        "Size (%)": f"{pct:.1f}%",
        "Top Words": top_words,
        "Representative Comments": repr_str,
        "Clinical Interpretation": "",  # Manual fill
    })

table_df = pd.DataFrame(table_rows)

# Save Table 2
table_path = OUTPUT_DIR / "table2_topic_summary.csv"
table_df.to_csv(table_path, index=False)
print(f"\n→ Table 2 saved: {table_path}")

# Print to console
print("\n" + "=" * 80)
print("TABLE 2: Topic summary")
print("=" * 80)
print(table_df[["Topic ID", "Size", "Size (%)", "Top Words"]].to_string(index=False))

# ---------------------------------------------------------------------------
# Figure 2: Bar chart of topic prevalence
# ---------------------------------------------------------------------------
# Exclude outliers for cleaner viz
plot_df = table_df[table_df["Topic ID"] != "Outlier"].copy()
if len(plot_df) == 0:
    plot_df = table_df

fig, ax = plt.subplots(figsize=(12, max(6, len(plot_df) * 0.35)))
y_pos = np.arange(len(plot_df))
labels = [f"T{r['Topic ID']}" for _, r in plot_df.iterrows()]
counts = plot_df["Size"].values
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(plot_df)))

bars = ax.barh(y_pos, counts, color=colors)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("Number of comments", fontsize=11)
ax.set_title("Figure 2: Topic prevalence (BERTopic)", fontsize=12)
ax.invert_yaxis()
plt.tight_layout()
fig_path = OUTPUT_DIR / "figure2_topic_prevalence.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"→ Figure 2 saved: {fig_path}")

# ---------------------------------------------------------------------------
# Save comments with topic labels
# ---------------------------------------------------------------------------
comments_labeled_path = OUTPUT_DIR / "comments_with_topics.csv"
df.to_csv(comments_labeled_path, index=False)
print(f"→ Comments with topic_id saved: {comments_labeled_path}")

# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------
n_topics = len([t for t in topics if t != -1])
n_outliers = sum(1 for t in topics if t == -1)
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"  Topics discovered: {n_topics}")
print(f"  Outliers (-1): {n_outliers} ({100*n_outliers/len(df):.1f}%)")
print(f"  Outputs in: {OUTPUT_DIR}/")
