"""
Urticaria/Angioedema YouTube Infodemiology — Data Collection
Methodology: JDV dermatoses infodemiology + JMIR YouTube big-data

Steps:
  1. videos  → Search videos by keywords → videos_raw.csv
  2. filter  → Rule-based auto-filter → videos_prefiltered.csv, videos_shortlist.csv
  3. comments → Extract comments from videos_clean.csv → comments_raw.csv, comments_deduplicated.csv
"""

import argparse
import os
import re
import sys
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# B1) Video inclusion query — core terms (methodology B1)
SEARCH_KEYWORDS = [
    "urticaria",
    "hives",
    "chronic hives",
    "angioedema",
    "lip swelling",
    "face swelling",
    "eyelid swelling",
    "tongue swelling",
]

# C2) Bias control (JMIR logic)
MAX_COMMENTS_PER_VIDEO = 350
FIRST_COMMENT_PER_USER_ONLY = True

# Output filenames
VIDEOS_RAW = "videos_raw.csv"
VIDEOS_PREFILTERED = "videos_prefiltered.csv"
VIDEOS_SHORTLIST = "videos_shortlist.csv"
VIDEOS_CLEAN = "videos_clean.csv"
COMMENTS_RAW = "comments_raw.csv"
COMMENTS_DEDUPED = "comments_deduplicated.csv"

# B2) Rule-based video filter (methodology B2 + ChatGPT precision filter)
# Include: title OR description must contain ≥1 core term
INCLUDE_TERMS = r"\b(urticaria|hives|angioedema)\b"
# Exclude: confounders (pattern -> exclusion_reason)
EXCLUDE_RULES = [
    (r"\b(eczema|atopic\s*dermatitis)\b", "eczema/atopic_dermatitis"),
    (r"\b(contact\s*dermatitis)\b", "contact_dermatitis"),
    (r"\b(drug\s*rash|SJS|stevens[\s\-]johnson|TEN|toxic\s*epidermal)\b", "drug_rash_SJS_TEN"),
    (r"\b(psoriasis|psoriatic)\b", "psoriasis"),
    (r"\b(rosacea)\b", "rosacea"),
    (r"\b(viral\s*exanthem|measles|chickenpox|varicella)\b", "viral_exanthem"),
    (r"\b(bedbug|bed\s*bug|insect\s*bite|mosquito\s*bite)\b", "insect_bedbug"),
    (r"\b(veterinary|pet\s*dog|cat\s*skin)\b", "veterinary"),
    (r"\b(detox|liver\s*cleanse|parasite\s*cure)\b", "alternative_marketing"),
]


def get_api_key():
    """Load API key from environment or config."""
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


def get_youtube_client():
    api_key = get_api_key()
    return build("youtube", "v3", developerKey=api_key)


# ---------------------------------------------------------------------------
# Step 1: Video search
# ---------------------------------------------------------------------------


def search_videos(youtube):
    """
    Search YouTube for each keyword. Returns list of dicts with:
    video_id, title, description, publish_date, search_term
    """
    all_videos = {}  # video_id -> {video data, search_terms list}

    for keyword in tqdm(SEARCH_KEYWORDS, desc="Searching keywords"):
        next_page = None
        while True:
            try:
                request = youtube.search().list(
                    part="snippet",
                    q=keyword,
                    type="video",
                    maxResults=50,
                    pageToken=next_page,
                    order="relevance",
                )
                response = request.execute()
            except HttpError as e:
                if e.resp.status == 403 and "quotaExceeded" in str(e):
                    print("\nQuota exceeded. Wait for next day or request increase.")
                raise

            for item in response.get("items", []):
                vid = item["id"].get("videoId")
                if not vid:
                    continue
                snippet = item["snippet"]
                if vid not in all_videos:
                    all_videos[vid] = {
                        "video_id": vid,
                        "title": snippet.get("title", ""),
                        "description": snippet.get("description", ""),
                        "publish_date": snippet.get("publishedAt", ""),
                        "search_term": keyword,
                    }
                else:
                    all_videos[vid]["search_term"] += "; " + keyword

            next_page = response.get("nextPageToken")
            if not next_page:
                break

    return list(all_videos.values())


def fetch_full_descriptions(youtube, video_ids):
    """Fetch full descriptions via videos.list (batch of 50)."""
    results = {}
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i : i + 50]
        try:
            response = youtube.videos().list(
                part="snippet",
                id=",".join(batch),
            ).execute()
            for item in response.get("items", []):
                vid = item["id"]
                results[vid] = item["snippet"].get("description", "")
        except HttpError:
            pass
    return results


def run_step_videos():
    """Step 1: Search videos and save videos_raw.csv."""
    print("Step 1: Video search")
    print("-" * 50)
    youtube = get_youtube_client()

    videos = search_videos(youtube)
    if not videos:
        print("No videos found.")
        return

    video_ids = [v["video_id"] for v in videos]
    print(f"Fetching full descriptions for {len(video_ids)} unique videos...")
    descriptions = fetch_full_descriptions(youtube, video_ids)

    for v in videos:
        v["description"] = descriptions.get(v["video_id"], v.get("description", ""))
        v["eligible_video"] = ""
        v["exclusion_reason"] = ""

    df = pd.DataFrame(videos)
    cols = ["video_id", "title", "description", "publish_date", "search_term", "eligible_video", "exclusion_reason"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(VIDEOS_RAW, index=False, encoding="utf-8-sig")
    print(f"Saved {VIDEOS_RAW} ({len(df)} videos)")
    print("\nNext: python datacollection.py --step filter")


# ---------------------------------------------------------------------------
# Step 2: Rule-based video filter
# ---------------------------------------------------------------------------


def run_step_filter(args):
    """
    Rule-based filter to cut ~70-80% junk. Outputs:
    - videos_prefiltered.csv: all videos + prefilter_eligible, auto_exclusion_reason
    - videos_shortlist.csv: only prefilter_eligible=1, for manual screening (~400-500)
    """
    print("Step 2: Rule-based video filter")
    print("-" * 50)

    if not os.path.exists(VIDEOS_RAW):
        print(f"Error: {VIDEOS_RAW} not found. Run --step videos first.")
        sys.exit(1)

    df = pd.read_csv(VIDEOS_RAW, encoding="utf-8-sig")

    include_re = re.compile(INCLUDE_TERMS, re.IGNORECASE)
    exclude_res = [(re.compile(p, re.IGNORECASE), r) for p, r in EXCLUDE_RULES]

    prefilter_eligible = []
    auto_exclusion_reason = []
    exclude_counts = {}

    for _, row in df.iterrows():
        title = str(row.get("title", "") or "")
        desc = str(row.get("description", "") or "")
        combined = f" {title} {desc} "

        # Must contain ≥1 core term
        if not include_re.search(combined):
            prefilter_eligible.append(0)
            auto_exclusion_reason.append("no_core_term")
            exclude_counts["no_core_term"] = exclude_counts.get("no_core_term", 0) + 1
            continue

        # Check exclusions
        excluded = None
        for pattern, reason in exclude_res:
            if pattern.search(combined):
                excluded = reason
                break
        if excluded:
            prefilter_eligible.append(0)
            auto_exclusion_reason.append(excluded)
            exclude_counts[excluded] = exclude_counts.get(excluded, 0) + 1
        else:
            prefilter_eligible.append(1)
            auto_exclusion_reason.append("")

    df["prefilter_eligible"] = prefilter_eligible
    df["auto_exclusion_reason"] = auto_exclusion_reason
    df.to_csv(VIDEOS_PREFILTERED, index=False, encoding="utf-8-sig")

    shortlist = df[df["prefilter_eligible"] == 1].copy()
    shortlist["eligible_video"] = ""
    shortlist["exclusion_reason"] = ""
    shortlist.to_csv(VIDEOS_SHORTLIST, index=False, encoding="utf-8-sig")

    n_total = len(df)
    n_shortlist = len(shortlist)
    n_excluded = n_total - n_shortlist

    print(f"Total videos:        {n_total}")
    print(f"Excluded (auto):     {n_excluded} ({100 * n_excluded / n_total:.1f}%)")
    print(f"Shortlist (manual):  {n_shortlist}")
    print("\nExclusion breakdown:")
    for reason, count in sorted(exclude_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    print(f"\nSaved {VIDEOS_PREFILTERED}")
    print(f"Saved {VIDEOS_SHORTLIST} ({n_shortlist} videos for manual screening)")
    print("\nNext: Manually screen videos_shortlist.csv -> set eligible_video (1/0), exclusion_reason -> save as videos_clean.csv")


# ---------------------------------------------------------------------------
# Step 3: Comment extraction
# ---------------------------------------------------------------------------


def extract_comments_for_video(youtube, video_id, max_comments=MAX_COMMENTS_PER_VIDEO):
    """
    Extract top-level comments for one video.
    Returns list of dicts: comment_id, video_id, author_channel_id, comment_text, comment_date, like_count
    """
    comments = []
    next_page = None

    while len(comments) < max_comments:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                pageToken=next_page,
                order="time",
                textFormat="plainText",
            )
            response = request.execute()
        except HttpError as e:
            if "commentsDisabled" in str(e) or "disabledComments" in str(e):
                return []
            raise

        for item in response.get("items", []):
            top = item["snippet"]["topLevelComment"]
            sid = top["snippet"]
            c = {
                "comment_id": top["id"],
                "video_id": video_id,
                "author_channel_id": sid.get("authorChannelId", {}).get("value", ""),
                "comment_text": sid.get("textDisplay", ""),
                "comment_date": sid.get("publishedAt", ""),
                "like_count": sid.get("likeCount", 0),
            }
            comments.append(c)
            if len(comments) >= max_comments:
                break

        next_page = response.get("nextPageToken")
        if not next_page:
            break

    return comments


def deduplicate_by_user(comments):
    """Keep first (most recent) comment per author_channel_id."""
    seen = set()
    out = []
    for c in comments:
        aid = c.get("author_channel_id") or ""
        if aid and aid in seen:
            continue
        if aid:
            seen.add(aid)
        out.append(c)
    return out


def run_step_comments():
    """Step 3: Extract comments from videos_clean.csv."""
    print("Step 3: Comment extraction")
    print("-" * 50)

    if not os.path.exists(VIDEOS_CLEAN):
        print(f"Error: {VIDEOS_CLEAN} not found.")
        print("Create it by: 1) --step videos, 2) --step filter, 3) manual screen shortlist, 4) save as videos_clean.csv")
        sys.exit(1)

    df_videos = pd.read_csv(VIDEOS_CLEAN, encoding="utf-8-sig")
    eligible = df_videos[df_videos["eligible_video"] == 1]
    if "eligible_video" in df_videos.columns and eligible.empty:
        eligible = df_videos[df_videos["eligible_video"] == "1"]
    if eligible.empty:
        print("No eligible videos (eligible_video=1) in videos_clean.csv")
        return

    video_ids = eligible["video_id"].astype(str).unique().tolist()
    print(f"Extracting comments for {len(video_ids)} eligible videos...")

    youtube = get_youtube_client()
    all_comments = []
    skipped = 0

    for vid in tqdm(video_ids, desc="Videos"):
        try:
            comments = extract_comments_for_video(youtube, vid, MAX_COMMENTS_PER_VIDEO)
            for c in comments:
                all_comments.append(c)
        except HttpError as e:
            if "quotaExceeded" in str(e):
                print(f"\nQuota exceeded after {len(all_comments)} comments. Save progress and retry later.")
                break
            skipped += 1
            continue

    if not all_comments:
        print("No comments extracted.")
        return

    df_raw = pd.DataFrame(all_comments)
    df_raw.to_csv(COMMENTS_RAW, index=False, encoding="utf-8-sig")
    print(f"Saved {COMMENTS_RAW} ({len(df_raw)} comments)")

    if FIRST_COMMENT_PER_USER_ONLY:
        rows = [c for _, c in df_raw.iterrows()]
        deduped = deduplicate_by_user(rows)
        df_dedup = pd.DataFrame(deduped)
    else:
        df_dedup = df_raw.copy()

    df_dedup.to_csv(COMMENTS_DEDUPED, index=False, encoding="utf-8-sig")
    print(f"Saved {COMMENTS_DEDUPED} ({len(df_dedup)} comments, per-user deduped)")
    if skipped:
        print(f"Skipped {skipped} videos due to API errors.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Urticaria YouTube Infodemiology Data Collection")
    parser.add_argument(
        "--step",
        choices=["videos", "filter", "comments"],
        required=True,
        help="videos: search → videos_raw.csv | filter: rule-based → shortlist | comments: extract from videos_clean.csv",
    )
    args = parser.parse_args()

    if args.step == "videos":
        run_step_videos()
    elif args.step == "filter":
        run_step_filter(args)
    else:
        run_step_comments()


if __name__ == "__main__":
    main()
