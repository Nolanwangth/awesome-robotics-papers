#!/usr/bin/env python3
"""
Crawl CoRL 2025 and RSS 2025 for embodied AI papers (oral / best paper / highlight).
Filters by relevance keywords (VLA, RL, manipulation, imitation learning, world model, etc.)
Finds arxiv links. Supports incremental updates.
"""

import json
import os
import re
import hashlib
import time
import html as html_mod
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from urllib.parse import quote

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "papers.json"

REQUEST_DELAY = 1.0  # seconds between webpage requests
ARXIV_DELAY = 3.5    # seconds between arxiv API calls (rate limit is strict)

# ── Relevance Keywords ──────────────────────────────────────────────
# Papers matching any of these phrases (case-insensitive) are included.
# These cover embodied AI topics: VLA, RL, manipulation, imitation learning,
# behavior cloning, world models, and closely related areas.
KEYWORDS = [
    # ── VLA / Vision-Language-Action ──
    r"\bvla\b",                         # VLA (standalone abbreviation)
    r"vision.?language.?action",        # vision-language-action (any separator)
    r"visuomotor", r"visuo.?motor",     # visuomotor / visuo-motor
    r"visual.*policy",                  # visual policy, visual-motor policy
    r"generalist.*(?:robot|policy)",    # generalist robot/policy
    r"robot.*foundation",              # robot foundation model
    r"openvla",                         # common specific VLA model
    r"rt[-_]?[12]",                     # RT-1, RT-2
    r"\bflow.?matching\b",             # flow matching (used in modern VLA)
    r"diffusion.*policy",              # diffusion policy
    r"\bpolicy.*(?:learning|model)\b", # policy learning / policy model
    r"cross.?embodiment",              # cross-embodiment learning

    # ── Reinforcement Learning ──
    r"\brl\b",                          # RL abbreviation
    r"reinforcement learning",          # full term
    r"reward\s+(?:function|learning|shaping|model)",
    r"inverse reinforcement",           # IRL
    r"\b(?:model.based|model.?free)\b.*\brl\b",  # model-based/free RL

    # ── Manipulation ──
    r"\bmanipulation\b",                # robot/manipulation
    r"\bgrasp(?:ing)?\b",              # grasp, grasping
    r"\bdexterous\b",                   # dexterous manipulation
    r"pick.?and.?place",               # pick-and-place
    r"in.?hand.*manipulation",         # in-hand manipulation
    r"mobile.*manipulation",           # mobile manipulation
    r"loco.?manipulation",             # loco-manipulation
    r"\b(?:re)?grasp\w*ing\b",        # regrasping etc.

    # ── Imitation Learning / Behavior Cloning ──
    r"imitation learning",              # full term
    r"learning from demonstration",     # LfD
    r"\blfd\b",                         # abbreviation
    r"behavior.?clon",                  # behavior cloning / behavioral cloning
    r"behaviour.?clon",                 # British spelling
    r"apprenticeship learning",         # synonym
    r"\bdemonstration",                # demos / demonstration
    r"few.?shot.*(?:imit|demonstrat)",  # few-shot imitation
    r"\bdagger\b",                      # DAgger algorithm
    r"\bgail\b",                        # GAIL (generative adversarial IL)
    r"\bmimic\w*\b",                     # mimic / mimicking / mimicry

    # ── World Models ──
    r"world model",                     # world model
    r"dynamics model",                  # learned dynamics
    r"model.based.*\b(?:rl|learn|control)\b",  # model-based approaches
    r"(?:video|visual).*(?:predict|world)",    # video prediction as world model
    r"latent.*dynamic",                 # latent dynamics

    # ── Broader Embodied / Robot Learning ──
    r"\bembodied\b",                    # embodied AI / embodied intelligence
    r"\bhumanoid\b",                    # humanoid robots
    r"sim2real\b", r"sim.?to.?real\b", # sim-to-real
    r"robot\s*(?:learn|policy|control|skill|foundation)",
    r"policy.*(?:optimization|gradient|search)",
]

# ── Helpers ─────────────────────────────────────────────────────────

def _fetch(url, headers=None):
    """Fetch URL with a simple delay to avoid getting rate-limited."""
    time.sleep(REQUEST_DELAY)
    req = Request(url, headers=headers or {"User-Agent": "awesome-robotics-papers/1.0"})
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def _paper_id(conference, title):
    """Deterministic unique ID for a paper."""
    raw = f"{conference}::{title.strip().lower()}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _normalize_arxiv(url):
    """Normalize arxiv URL to canonical abs format."""
    if not url:
        return None
    m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)", url)
    return f"https://arxiv.org/abs/{m.group(1)}" if m else url


def _load_existing():
    if DATA_FILE.exists():
        with open(DATA_FILE) as f:
            return json.load(f)
    return {"papers": [], "metadata": {"last_updated": None}}


def _save(data):
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    data["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()
    # Normalize all arxiv URLs
    for p in data["papers"]:
        if p.get("arxiv_url"):
            p["arxiv_url"] = _normalize_arxiv(p["arxiv_url"])
    # Deduplicate by id
    seen = set()
    unique = []
    for p in data["papers"]:
        if p["id"] not in seen:
            seen.add(p["id"])
            unique.append(p)
    data["papers"] = unique
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved {len(data['papers'])} papers to papers.json")


def _keyword_match(text):
    """Check if text matches any relevance keyword."""
    if not text:
        return False
    text_lower = text.lower()
    for pattern in KEYWORDS:
        if re.search(pattern, text_lower):
            return True
    return False


def _search_arxiv(title):
    """Search arxiv by paper title, return URL or None."""
    query = re.sub(r"[^\x00-\x7F]+", "", title.strip())[:200]
    if not query.strip():
        return None
    encoded = quote(query)
    url = f"https://export.arxiv.org/api/query?search_query=ti:{encoded}&max_results=3"
    for attempt in range(2):
        try:
            time.sleep(ARXIV_DELAY)
            req = Request(url, headers={"User-Agent": "awesome-robotics-papers/1.0"})
            with urlopen(req, timeout=20) as resp:
                xml = resp.read().decode("utf-8", errors="replace")
            entries = re.findall(
                r"<entry>.*?<id>(.*?)</id>.*?<title>(.*?)</title>",
                xml, re.DOTALL
            )
            for link, entry_title in entries:
                link = link.strip()
                entry_title = re.sub(r"\s+", " ", entry_title.strip())
                if "arxiv.org/abs/" in link and _title_similar(query, entry_title):
                    clean = re.sub(r"(v\d+)$", "", link.rstrip("/"))
                    return clean.split("?")[0]
            return None
        except HTTPError as e:
            if e.code == 429 and attempt == 0:
                time.sleep(10)
                continue
            return None
        except Exception:
            return None
    return None


def _title_similar(a, b):
    """Check if two titles are similar enough (simple character overlap)."""
    a = re.sub(r"[^a-z0-9\s]", "", a.lower())
    b = re.sub(r"[^a-z0-9\s]", "", b.lower())
    a_words = set(a.split())
    b_words = set(b.split())
    if len(a_words) < 3 or len(b_words) < 3:
        return a == b
    overlap = len(a_words & b_words)
    return overlap / min(len(a_words), len(b_words)) >= 0.6


# ── CoRL 2025 Scraper ──────────────────────────────────────────────

CORL_README_URL = "https://raw.githubusercontent.com/smallfryy/corl-2025-papers/main/README.md"

def crawl_corl_2025():
    """Parse the smallfryy/corl-2025-papers README for oral papers."""
    print("\n=== CoRL 2025 ===")
    papers = []
    try:
        md = _fetch(CORL_README_URL)
    except Exception as e:
        print(f"  ✗ Failed to fetch CoRL README: {e}")
        return papers

    # Parse the markdown tables for the Orals section
    # The README has ## Orals then a table, then ## Posters then a table
    # Table rows: | # | Title | TLDR | Project Page | Paper | Code |

    # Find orals section
    orals_section = re.search(r"## Orals\n(.*?)(?=## Posters|\Z)", md, re.DOTALL)
    if not orals_section:
        print("  ✗ Could not find Orals section in README")
        return papers

    table_text = orals_section.group(1)
    # Parse HTML table rows
    rows = re.findall(r"<tr>\s*<td>(.*?)</td>\s*<td>(.*?)</td>\s*<td>(.*?)</td>\s*<td>(.*?)</td>\s*<td>(.*?)</td>\s*<td>(.*?)</td>\s*</tr>", table_text, re.DOTALL)

    print(f"  Found {len(rows)} oral paper entries")
    for row in rows:
        num, title_html, tldr_html, proj_html, paper_html, code_html = row
        title = re.sub(r"<[^>]+>", "", title_html).strip()
        tldr = re.sub(r"<[^>]+>", "", tldr_html).strip()
        paper_link = re.search(r'href="([^"]+)"', paper_html)
        paper_url = paper_link.group(1) if paper_link else ""
        arxiv_url = _normalize_arxiv(paper_url) if "arxiv" in paper_url.lower() else None
        openreview_url = paper_url if "openreview" in paper_url.lower() else None

        pid = _paper_id("CoRL 2025", title)
        papers.append({
            "id": pid,
            "title": title,
            "conference": "CoRL 2025",
            "category": "Oral",
            "authors": [],
            "abstract": tldr,
            "arxiv_url": arxiv_url or None,
            "openreview_url": openreview_url or None,
            "project_url": _extract_link(proj_html),
            "code_url": _extract_link(code_html),
            "crawled_at": datetime.now(timezone.utc).isoformat(),
        })

    print(f"  → {len(papers)} oral papers from CoRL 2025")
    return papers


def _extract_link(html_fragment):
    """Extract href from an HTML anchor fragment, returning None for N/A."""
    m = re.search(r'href="([^"]+)"', html_fragment)
    return m.group(1) if m else None


# ── RSS 2025 Scraper ───────────────────────────────────────────────

RSS_PROCEEDINGS_URL = "https://www.roboticsproceedings.org/rss21/index.html"
RSS_AWARDS_URL = "https://roboticsconference.org/2025/program/awards/"

# Sessions most relevant to embodied AI from RSS schedule
RSS_EMBODIED_SESSIONS = {
    "2. VLA Models",
    "6. Manipulation I",
    "7. Humanoids",
    "8. Imitation Learning I",
    "11. Manipulation II",
    "13. Mobile Manipulation and Locomotion",
    "16. Manipulation III",
    "17. Imitation Learning II",
    "3. Scaling Robot Learning",
}


def crawl_rss_2025():
    """Crawl RSS 2025 proceedings for papers + session tags from conf website."""
    print("\n=== RSS 2025 ===")
    papers = []

    # Step 1: Grab the paper list from proceedings (title + authors)
    try:
        html = _fetch(RSS_PROCEEDINGS_URL)
    except Exception as e:
        print(f"  ✗ Failed to fetch RSS proceedings: {e}")
        return papers

    # Extract paper entries
    entries = re.findall(
        r'<a href="(p\d+\.html)">(.*?)</a><br>\s*<i>(.*?)</i>',
        html, re.DOTALL
    )
    print(f"  Found {len(entries)} proceedings entries")

    # Step 2: Get session assignments from the conference website
    try:
        conf_html = _fetch("https://roboticsconference.org/2025/program/papers/")
    except Exception as e:
        print(f"  ✗ Failed to fetch RSS conf site: {e}")
        conf_html = ""

    # Build session map: paper number -> session
    session_map = {}
    if conf_html:
        for m in re.finditer(
            r'<tr[^>]*session="([^"]*)"\s*>.*?<td[^>]*>\s*(\d+)\s*</td>',
            conf_html, re.DOTALL
        ):
            session_map[m.group(2)] = m.group(1)

    # Step 3: Build category based on session and awards
    award_winners = fetch_rss_awards()

    processed = 0
    for pid, title, authors in entries:
        title = html_mod.unescape(title.strip())
        authors = html_mod.unescape(authors.strip())
        num = re.search(r"p(\d+)", pid).group(1)

        # Determine category
        session = session_map.get(num, "")
        category = "Accepted"
        if title in award_winners:
            category = "Best Paper"
        elif session in RSS_EMBODIED_SESSIONS:
            category = "Oral"

        # Relevance filter (title only for initial pass)
        if not _keyword_match(title) and category not in ("Best Paper",):
            continue

        # Build authors list
        author_list = [a.strip() for a in authors.split(",")]
        pid_unique = _paper_id("RSS 2025", title)
        pdf_url = f"https://www.roboticsproceedings.org/rss21/p{num}.pdf"
        papers.append({
            "id": pid_unique,
            "title": title,
            "authors": author_list,
            "conference": "RSS 2025",
            "category": category,
            "session": session,
            "abstract": "",
            "arxiv_url": None,
            "openreview_url": None,
            "project_url": None,
            "code_url": None,
            "pdf_url": pdf_url,
            "crawled_at": datetime.now(timezone.utc).isoformat(),
        })
        processed += 1

    print(f"  → {processed} relevant papers after filtering")
    return papers


def fetch_rss_awards():
    """Fetch RSS 2025 award winners."""
    print("  Fetching RSS awards page...")
    award_titles = set()
    try:
        html = _fetch(RSS_AWARDS_URL)
        # Looking for Best Paper Award mentions
        # The awards page typically lists award-winning papers
        # Extract from any heading -> content patterns
        # Remove HTML tags and find paper-like content
        text = re.sub(r"<[^>]+>", "\n", html)
        lines = [l.strip() for l in text.split("\n") if l.strip() and len(l.strip()) > 30]
        # For now, return empty set - can be augmented manually
        # RSS typically announces awards at the conference
    except Exception as e:
        print(f"  ⚠ Could not fetch awards: {e}")
    return award_titles


# ── Merge & Update ─────────────────────────────────────────────────

def merge_papers(existing_data, new_papers):
    """Merge newly crawled papers with existing data (incremental update)."""
    existing = {p["id"]: p for p in existing_data["papers"]}
    updated = 0
    added = 0

    for np in new_papers:
        pid = np["id"]
        if pid in existing:
            ep = existing[pid]
            # Update arxiv_url if newly found
            if not ep.get("arxiv_url") and np.get("arxiv_url"):
                ep["arxiv_url"] = np["arxiv_url"]
                ep["crawled_at"] = np["crawled_at"]
                updated += 1
            # Update abstract/tldr if missing
            if not ep.get("abstract") and np.get("abstract"):
                ep["abstract"] = np["abstract"]
                updated += 1
            # Update category if better info available (e.g., Best Paper)
            if np.get("category") in ("Best Paper", "Oral") and ep.get("category") in ("Accepted",):
                ep["category"] = np["category"]
                updated += 1
        else:
            existing[pid] = np
            added += 1

    result = list(existing.values())
    result.sort(key=lambda p: (p["conference"], p["title"]))
    print(f"\n  Incremental update: +{added} new, {updated} updated, {len(result)} total")
    return {"papers": result, "metadata": existing_data["metadata"]}


# ── Arxiv Link Enrichment ──────────────────────────────────────────

def enrich_arxiv_links(data):
    """Search arxiv for papers missing arxiv links."""
    to_check = [p for p in data["papers"] if not p.get("arxiv_url")]
    if not to_check:
        print("  All papers already have arxiv links.")
        return data

    print(f"\n  Searching arxiv for {len(to_check)} papers without links...")
    found = 0
    for i, p in enumerate(to_check):
        if i > 0 and i % 10 == 0:
            print(f"    ... {i}/{len(to_check)} checked ({found} found)")
        url = _search_arxiv(p["title"])
        if url:
            p["arxiv_url"] = url
            found += 1
        # Find in existing data
        for ep in data["papers"]:
            if ep["id"] == p["id"]:
                ep["arxiv_url"] = url
                break

    print(f"  Found arxiv links for {found}/{len(to_check)} papers")
    return data


# ── README Generator ───────────────────────────────────────────────

def generate_readme(data):
    """Generate README.md from papers data."""
    papers = data["papers"]
    last_updated = data["metadata"].get("last_updated", "N/A")[:10]

    lines = [
        "# Awesome Robotics Papers",
        "",
        "A curated list of **embodied AI / robot learning** papers from top robotics conferences.",
        "",
        f"Last updated: {last_updated}  |  Papers: {len(papers)}",
        "",
        "## Contents",
        "- [CoRL 2025](#corl-2025)",
        "- [RSS 2025](#rss-2025)",
        "",
    ]

    for conf in ["CoRL 2025", "RSS 2025"]:
        conf_papers = [p for p in papers if p["conference"] == conf]
        if not conf_papers:
            continue

        lines.append(f"## {conf}\n")

        # Group by category
        for cat in ["Best Paper", "Oral", "Highlight", "Accepted"]:
            cat_papers = [p for p in conf_papers if p["category"] == cat]
            if not cat_papers:
                continue

            lines.append(f"### {cat}\n")
            lines.append("| # | Title | Links |")
            lines.append("|---|-------|-------|")

            for i, p in enumerate(cat_papers, 1):
                title = p["title"]
                links = []

                # Build links column
                if p.get("arxiv_url"):
                    links.append(f"[arXiv]({p['arxiv_url']})")
                if p.get("project_url"):
                    links.append(f"[Project]({p['project_url']})")
                if p.get("code_url"):
                    links.append(f"[Code]({p['code_url']})")
                if p.get("openreview_url"):
                    links.append(f"[OpenReview]({p['openreview_url']})")
                if p.get("pdf_url"):
                    links.append(f"[PDF]({p['pdf_url']})")

                links_str = " · ".join(links) if links else "—"
                lines.append(f"| {i} | {title} | {links_str} |")

            lines.append("")

    lines.append("---")
    lines.append("Auto-generated by [crawl_papers.py](scripts/crawl_papers.py)")
    lines.append("")

    readme = "\n".join(lines)
    readme_path = BASE_DIR / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme)
    print(f"\n  ✓ Generated README.md ({len(papers)} papers)")


# ── Main ───────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("Awesome Robotics Papers Crawler")
    print("=" * 50)

    # Load existing data
    data = _load_existing()
    print(f"Existing papers: {len(data['papers'])}")

    # Crawl CoRL 2025
    corl_papers = crawl_corl_2025()

    # Crawl RSS 2025
    rss_papers = crawl_rss_2025()

    # Merge
    all_new = corl_papers + rss_papers
    data = merge_papers(data, all_new)

    # Enrich with arxiv links
    data = enrich_arxiv_links(data)

    # Save
    _save(data)

    # Generate README
    generate_readme(data)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
