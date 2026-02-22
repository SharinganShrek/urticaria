"""
Urticaria YouTube Comments — Unmet Needs Analysis
10 binary unmet need domains + co-occurrence matrix + network graph.

Unmet need = clinically meaningful problem (burden, fear, uncertainty, failure, gap).
A comment can have multiple unmet needs.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_CSV = "topic_outputs/comments_with_speaker_gender_and_topics.csv"
OUTPUT_DIR = Path("topic_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
TEXT_COL = "clean_text"
MIN_COOCCURRENCE = 5  # Only draw edges with co-occurrence >= this

# ---------------------------------------------------------------------------
# 10 Unmet Need Domains — Keyword dictionaries (regex, case-insensitive)
# User-provided + dataset-derived expansions
# ---------------------------------------------------------------------------
UNMET_NEED_KEYWORDS = {
    "uncontrolled_symptoms": [
        r"\bsevere\b", r"\bunbearable\b", r"\bnonstop\b", r"\bcan'?t stop\b",
        r"\bcan't stop\b", r"\bworse\b", r"\bflare\b", r"\bflaring\b",
        r"\bdriving me crazy\b", r"\bterrible\b", r"\bawful\b", r"\bhorrible\b",
        r"\bpainful\b", r"\bbad\b", r"\breally bad\b", r"\bso bad\b",
        r"\bnever stops\b", r"\bwon't stop\b", r"\bdoesn't go away\b",
        r"\bextreme\b", r"\bintense\b",
    ],
    "airway_fear": [
        r"\bthroat\b", r"\bthroat closing\b", r"\btongue swelling\b",
        r"\bswollen lip\b", r"\bswollen lips\b", r"\bswollen face\b",
        r"\bcan'?t breathe\b", r"\bcant breathe\b", r"\bbreathing\b",
        r"\bER\b", r"\bemergency\b", r"\bepipen\b", r"\banaphylaxis\b",
        r"\bpanic\b", r"\bscared\b", r"\bintubation\b",
        r"\blip swollen\b", r"\blips swollen\b", r"\bswelling\b", r"\bswell\b",
        r"\bpuffy\b", r"\bpuffiness\b",
    ],
    "treatment_failure": [
        r"\bnothing works\b", r"\btried everything\b", r"\bstill have\b",
        r"\bno relief\b", r"\bfailed\b", r"\brefractory\b", r"\bno cure\b",
        r"\bincurable\b", r"\bdoesn'?t work\b", r"\bdont work\b",
        r"\bdoesn'?t help\b", r"\bno help\b", r"\bnot working\b",
        r"\bnever goes away\b", r"\bwon'?t go away\b", r"\bstill getting\b",
        r"\bno improvement\b", r"\bstill suffering\b",
    ],
    "medication_side_effect_fear": [
        r"\bprednisone\b", r"\bsteroid\b", r"\bsteroids\b", r"\bside effect",
        r"\bside effects\b", r"\bsleepy\b", r"\bdrowsy\b", r"\bdrowsiness\b",
        r"\bweight\b", r"\bafraid to take\b", r"\bscared to take\b",
        r"\bbenadryl\b", r"\bdiphenhydramine\b", r"\bout of it\b",
        r"\bfast heartbeat\b", r"\bheart palpitations\b", r"\banxious\b",
        r"\bmedication.?fear\b", r"\bdon'?t want to take\b",
    ],
    "trigger_uncertainty": [
        r"\bdon'?t know.*trigger\b", r"\bwhat causes\b", r"\bcause unknown\b",
        r"\bno idea\b", r"\bconfused\b", r"\bconflicting\b", r"\bdon'?t know why\b",
        r"\bdon'?t know what\b", r"\btrigger\b", r"\btriggers\b",
        r"\bhistamine confusion\b", r"\bfood confusion\b", r"\bwhy.*hives\b",
        r"\bwhat.?causing\b", r"\bcause.?unknown\b", r"\bidiopathic\b",
        r"\bno.?cause\b", r"\bnever found.?cause\b",
    ],
    "emotional_distress": [
        r"\banxiety\b", r"\banxious\b", r"\bdepressed\b", r"\bdepression\b",
        r"\bstressed\b", r"\bstress\b", r"\bhopeless\b", r"\bhopelessness\b",
        r"\bscared\b", r"\bfear\b", r"\bcrying\b", r"\bcry\b", r"\bmiserable\b",
        r"\bdevastating\b", r"\boverwhelming\b", r"\btraumatic\b",
        r"\bmental\b", r"\bpsychologically\b", r"\bemotionally\b",
        r"\bdying\b", r"\bdesperate\b", r"\bfrustrated\b", r"\bfrustrating\b",
    ],
    "sleep_daily_impairment": [
        r"\bcan'?t sleep\b", r"\bcant sleep\b", r"\binsomnia\b", r"\bnight\b",
        r"\bwoke\b", r"\bwaking\b", r"\bmiddle of the night\b", r"\b3 am\b",
        r"\b3am\b", r"\bwork\b", r"\bschool\b", r"\bjob\b", r"\baffecting\b",
        r"\baffected\b", r"\bdisrupt\b", r"\bdisruption\b", r"\bdaily\b",
        r"\bquality of life\b", r"\bnormal life\b", r"\broutine\b",
    ],
    "access_cost_barrier": [
        r"\bexpensive\b", r"\bcost\b", r"\bcosts\b", r"\binsurance\b",
        r"\bcan'?t afford\b", r"\bcant afford\b", r"\bwaiting\b",
        r"\bappointment\b", r"\bappointments\b", r"\breferral\b",
        r"\bxolair.?cost\b", r"\bafford\b", r"\bunaffordable\b",
        r"\bwait.?list\b", r"\bwaitlist\b", r"\bdoctor.?doesn'?t\b",
    ],
    "diagnostic_confusion": [
        r"\bis this hives\b", r"\bmosquito bites\b", r"\ballergy vs\b",
        r"\bdoctor doesn'?t know\b", r"\bdoctor dont know\b", r"\bdiagnosed\b",
        r"\bdiagnosis\b", r"\bwhat is this\b", r"\bis it hives\b",
        r"\bbites or hives\b", r"\brash or hives\b", r"\bconfused.*diagnosis\b",
        r"\bunclear.*cause\b", r"\bblood work\b", r"\ballergy test\b",
        r"\btested negative\b", r"\bno.?diagnosis\b",
    ],
    "misinformation_vulnerability": [
        r"\bmiracle\b", r"\bmiracle cure\b", r"\bguaranteed\b", r"\bguarantee\b",
        r"\bdetox\b", r"\bcleanse\b", r"\bherbal cure\b", r"\bnatural cure\b",
        r"\bstop meds\b", r"\bstop medication\b", r"\bscam\b", r"\bfake\b",
        r"\bDM me\b", r"\bWhatsApp\b", r"\border now\b", r"\bbuy\b", r"\bplanet ayurveda\b",
        r"\bcured me\b", r"\b100 percent\b", r"\binstant.?cure\b",
        r"\bproven.?cure\b", r"\bsecret.?remedy\b", r"\biherb\b",
    ],
}


def match_unmet_need(text: str, keywords: list) -> int:
    """Return 1 if any keyword (regex) matches text, else 0. Case-insensitive."""
    if pd.isna(text) or not str(text).strip():
        return 0
    t = str(text)
    for pat in keywords:
        if re.search(pat, t, re.IGNORECASE):
            return 1
    return 0


def main():
    print("Loading comments...")
    df = pd.read_csv(INPUT_CSV)
    texts = df[TEXT_COL].fillna("").astype(str).tolist()
    n = len(texts)
    print(f"  -> {n} comments from {INPUT_CSV}")

    # STEP 1 — Create 10 binary unmet need columns
    print("\nTagging unmet needs (regex, case-insensitive)...")
    for domain, keywords in UNMET_NEED_KEYWORDS.items():
        df[domain] = [match_unmet_need(t, keywords) for t in texts]
        count = df[domain].sum()
        print(f"  {domain}: {count} ({100*count/n:.1f}%)")

    # STEP 2 — Prevalence table
    prev_rows = []
    for domain in UNMET_NEED_KEYWORDS:
        freq = int(df[domain].sum())
        pct = 100 * freq / n
        prev_rows.append({"unmet_need": domain, "frequency": freq, "percent": f"{pct:.1f}%"})
    prev_df = pd.DataFrame(prev_rows)
    prev_path = OUTPUT_DIR / "unmet_need_prevalence.csv"
    prev_df.to_csv(prev_path, index=False)
    print(f"\nPrevalence table saved: {prev_path}")

    # STEP 3 — Co-occurrence matrix
    domains = list(UNMET_NEED_KEYWORDS.keys())
    cooccur = pd.DataFrame(0, index=domains, columns=domains)
    for i, d1 in enumerate(domains):
        for j, d2 in enumerate(domains):
            if i == j:
                cooccur.loc[d1, d2] = int(df[d1].sum())
            elif i < j:
                both = int(((df[d1] == 1) & (df[d2] == 1)).sum())
                cooccur.loc[d1, d2] = both
                cooccur.loc[d2, d1] = both
    cooccur_path = OUTPUT_DIR / "unmet_need_cooccurrence_matrix.csv"
    cooccur.to_csv(cooccur_path)
    print(f"Co-occurrence matrix saved: {cooccur_path}")

    # STEP 4 — Network graph
    if HAS_NETWORKX:
        G = nx.Graph()
        for d in domains:
            G.add_node(d, size=int(df[d].sum()))
        for i, d1 in enumerate(domains):
            for j, d2 in enumerate(domains):
                if i < j:
                    both = int(((df[d1] == 1) & (df[d2] == 1)).sum())
                    if both >= MIN_COOCCURRENCE:
                        G.add_edge(d1, d2, weight=both)
        pos = nx.spring_layout(G, seed=42, k=1.5, iterations=50)
        sizes = [G.nodes[d].get("size", 0) for d in G.nodes()]
        sizes_norm = [max(200, s * 0.5) for s in sizes]
        widths = [G[u][v].get("weight", 0) / 10 for u, v in G.edges()]
        widths = [max(0.5, w) for w in widths]
        plt.figure(figsize=(12, 10))
        nx.draw_networkx_nodes(G, pos, node_size=sizes_norm, node_color="steelblue", alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=widths, alpha=0.5, edge_color="gray")
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")
        plt.title("Unmet Needs Co-occurrence Network\n(node size = prevalence; edge = co-occurrence >= 5)")
        plt.axis("off")
        plt.tight_layout()
        fig_path = OUTPUT_DIR / "unmet_need_network.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Network graph saved: {fig_path}")
    else:
        print("  (networkx/matplotlib not installed — skipping network graph)")

    # Save comments with unmet need columns
    out_csv = OUTPUT_DIR / "comments_with_unmet_needs.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nComments with unmet needs saved: {out_csv}")

    # Console summary
    print("\n" + "=" * 80)
    print("UNMET NEED PREVALENCE")
    print("=" * 80)
    print(prev_df.to_string(index=False))
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    n_multi = (df[domains].sum(axis=1) > 1).sum()
    print(f"  Comments with >=1 unmet need: {(df[domains].sum(axis=1) >= 1).sum()}")
    print(f"  Comments with multiple unmet needs: {n_multi}")
    print(f"  Outputs in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
