"""
comments_english_only.csv içindeki her yorum için author'ın display name (kullanıcı adı) alır
ve Username sütununu ekleyerek yeni CSV kaydeder.
YouTube Data API v3 (config.py veya YOUTUBE_API_KEY env) kullanır.
"""

import os
import sys
import time
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_CSV = "comments_english_only.csv"
OUTPUT_CSV = "comments_english_only_with_usernames.csv"
BATCH_SIZE = 50  # YouTube API channels.list max 50 ID per request
DELAY_SEC = 0.1  # Rate limiting


def get_api_key():
    """Load API key from environment or config (same as datacollection.py)."""
    key = os.environ.get("YOUTUBE_API_KEY")
    if key:
        return key
    try:
        from config import YOUTUBE_API_KEY
        return YOUTUBE_API_KEY
    except ImportError:
        pass
    print("Error: YOUTUBE_API_KEY not found.")
    print("Set environment variable or create config.py with YOUTUBE_API_KEY.")
    sys.exit(1)


def fetch_channel_display_names(youtube, channel_ids):
    """
    YouTube channels.list ile channel_id -> display name (snippet.title) eşlemesi döner.
    channel_ids: list of author_channel_id
    Returns: dict {channel_id: display_name}
    """
    result = {}
    unique_ids = list(dict.fromkeys(c for c in channel_ids if pd.notna(c) and str(c).strip()))

    for i in tqdm(range(0, len(unique_ids), BATCH_SIZE), desc="Fetching usernames"):
        batch = unique_ids[i : i + BATCH_SIZE]
        try:
            response = youtube.channels().list(
                part="snippet",
                id=",".join(batch),
            ).execute()

            for item in response.get("items", []):
                cid = item["id"]
                title = item.get("snippet", {}).get("title", "") or ""
                result[cid] = title.strip() or "(no name)"

            time.sleep(DELAY_SEC)
        except HttpError as e:
            if e.resp.status == 403 and "quotaExceeded" in str(e):
                print("\nQuota exceeded. Try again tomorrow.")
                raise
            print(f"\nAPI error for batch: {e}")
            for cid in batch:
                if cid not in result:
                    result[cid] = "(unknown)"

    return result


def main():
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    if "author_channel_id" not in df.columns:
        print("Error: author_channel_id column not found.")
        sys.exit(1)

    youtube = build("youtube", "v3", developerKey=get_api_key())

    print("Fetching display names from YouTube API...")
    id_to_name = fetch_channel_display_names(youtube, df["author_channel_id"].astype(str).tolist())

    # Username sütununu author_channel_id'den sonra ekle
    usernames = df["author_channel_id"].astype(str).map(lambda x: id_to_name.get(x, "(unknown)"))
    # author_channel_id sütununun hemen sonrasına Username ekle
    cols = list(df.columns)
    idx = cols.index("author_channel_id") + 1
    df.insert(idx, "Username", usernames)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUTPUT_CSV} ({len(df)} rows)")
    print("Sample (first 3):")
    print(df[["author_channel_id", "Username", "comment_text"]].head(3).to_string())


if __name__ == "__main__":
    main()
