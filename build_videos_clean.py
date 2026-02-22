"""
Merge final screened video list with shortlist to produce videos_clean.csv.
Uses video_id to match and pulls description from shortlist.
"""
import pandas as pd

FINAL_PATH = r"c:\Users\emrea\Downloads\english titles - videos_final_no_HAE_english_titles.csv"
SHORTLIST_PATH = "videos_shortlist.csv"
OUTPUT_PATH = "videos_clean.csv"

def main():
    df_final = pd.read_csv(FINAL_PATH, encoding="utf-8-sig")
    df_short = pd.read_csv(SHORTLIST_PATH, encoding="utf-8-sig")

    # Final file = eligible videos only (all have eligible_video=1)
    final_ids = set(df_final["video_id"].astype(str).unique())

    # Keep only shortlist rows that are in final
    short_match = df_short[df_short["video_id"].astype(str).isin(final_ids)].copy()

    # Build videos_clean: video_id, title, description, publish_date, search_term, eligible_video, exclusion_reason
    short_match["eligible_video"] = 1
    short_match["exclusion_reason"] = ""

    cols = ["video_id", "title", "description", "publish_date", "search_term", "eligible_video", "exclusion_reason"]
    df_clean = short_match[[c for c in cols if c in short_match.columns]]

    df_clean.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Created {OUTPUT_PATH} with {len(df_clean)} videos")

    # Check for any final IDs not found in shortlist
    short_ids = set(short_match["video_id"].astype(str))
    missing = final_ids - short_ids
    if missing:
        print(f"Warning: {len(missing)} video_ids from final file not found in shortlist: {list(missing)[:5]}...")

if __name__ == "__main__":
    main()
