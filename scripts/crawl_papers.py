#!/usr/bin/env python3
"""Parse curated paper lists from GitHub and generate papers.json + README + Obsidian vault.
Sources:
  - Songwxuan/Embodied-AI-Paper-TopConf  (curated embodied AI papers)
  - SarahRastegar/Best-Papers-Top-Venues (Best Paper awards)
  - DoongLi/ICRA2025-Paper-List          (ICRA 2025)
  - DoongLi/IROS2025-Paper-List          (IROS 2025)
"""

import json, os, re, hashlib, time
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import urlopen, Request

BASE = Path(__file__).resolve().parent.parent
DATA_FILE = BASE / "papers.json"
VAULT_DIR = BASE / "papers"

# ── Helpers ────────────────────────────────────────────────────────

def pid(conf, title):
    return hashlib.md5(f"{conf}::{title.strip().lower()}".encode()).hexdigest()[:12]

def slug(title):
    s = re.sub(r"[^a-z0-9\s-]", "", title.lower())
    return re.sub(r"\s+", "-", s.strip(" -"))[:80]

def fetch(url):
    time.sleep(0.5)
    return urlopen(Request(url, headers={"User-Agent": "awesome-robotics/1"})).read().decode("utf-8")

def arxiv_extract(text):
    m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)", text)
    return f"https://arxiv.org/abs/{m.group(1)}" if m else None

# ── Source 1: Embodied-AI-Paper-TopConf ─────────────────────────────

def parse_embodied_ai():
    """Parse Songwxuan/Embodied-AI-Paper-TopConf README."""
    md = fetch("https://raw.githubusercontent.com/Songwxuan/Embodied-AI-Paper-TopConf/main/README.md")
    papers = []

    # Conference name mapping from README headers
    conf_map = {
        "ICML2026": "ICML 2026", "CVPR2026": "CVPR 2026", "ICLR2026": "ICLR 2026",
        "NeuIPS2025": "NeurIPS 2025", "CORL2025": "CoRL 2025", "ICCV2025": "ICCV 2025",
        "ICML2025": "ICML 2025", "RSS2025": "RSS 2025", "CVPR2025": "CVPR 2025",
        "ICLR2025": "ICLR 2025", "ICRA2025": "ICRA 2025",
    }

    current_conf = None
    for line in md.split("\n"):
        # Match conference header: # CVPR2026 or # ICML2026
        m = re.match(r"^#\s+(ICML202[56]|CVPR202[56]|ICLR202[56]|NeuIPS2025|CORL2025|ICCV2025|ICML2025|RSS2025|CVPR2025|ICLR2025|ICRA2025)\s*$", line)
        if m:
            current_conf = conf_map.get(m.group(1))
        if not current_conf:
            continue
        # Match paper entry: - Title [Paper](url) or - Title [Paper](url) | [Code](url)
        pm = re.match(r"^-\s+(.+?)\s+\[Paper\]\(([^)]+)\)", line)
        if not pm: continue
        title = pm.group(1).strip()
        url = pm.group(2)
        arxiv = arxiv_extract(url)
        papers.append({
            "id": pid(current_conf, title), "title": title,
            "conference": current_conf, "category": "Oral",
            "abstract": "", "arxiv_url": arxiv or url,
            "crawled_at": datetime.now(timezone.utc).isoformat(),
        })

    print(f"  [Embodied-AI] {len(papers)} papers from {len(set(p['conference'] for p in papers))} conferences")
    return papers

# ── Source 2: Best-Papers-Top-Venues ────────────────────────────────

def parse_best_papers():
    """Parse SarahRastegar/Best-Papers-Top-Venues. Multi-line format."""
    md = fetch("https://raw.githubusercontent.com/SarahRastegar/Best-Papers-Top-Venues/main/README.md")
    papers = []

    conf_map = {"CVPR": "CVPR", "ICLR": "ICLR", "NeurIPS": "NeurIPS",
                "ICCV": "ICCV", "ICML": "ICML", "ECCV": "ECCV"}

    lines = md.split("\n")
    current_conf = None
    current_year = None
    current_award = "Best Paper"

    for i, line in enumerate(lines):
        # Conference section: ## CVPR
        cm = re.match(r"^##\s+(CVPR|ICLR|NeurIPS|ICCV|ICML|ECCV)\s*$", line)
        if cm: current_conf = conf_map[cm.group(1)]; continue
        # Year: ### Best Papers 2025 or ### 2026
        ym = re.match(r"^###\s+(?:Best Papers\s+)?(\d{4})", line)
        if ym: current_year = ym.group(1); current_award = "Best Paper"; continue
        # Award heading: #### Best Paper Award: etc
        am = re.match(r"^####\s+(.+)", line)
        if am:
            a = am.group(1).rstrip(":").lower()
            if "student" in a and "honorable" not in a: current_award = "Best Student Paper"
            elif "honorable" in a: current_award = "Best Paper Honorable Mention"
            elif "test" in a: current_award = None
            else: current_award = "Best Paper"
            continue
        # Paper line: * Title (CONF YEAR) -- followed by [[Paper](url)] on same or next line
        pm = re.match(r"^\*\s+(.+?)\s+\(([A-Z]+)\s+(\d{4})\)\s*$", line)
        if not pm: continue
        title = pm.group(1).strip()
        conf_tag = conf_map.get(pm.group(2), pm.group(2))
        year = pm.group(3)
        if not current_award: continue
        # Look for arxiv link on this line or next few lines
        arxiv = None
        for j in range(i, min(i + 3, len(lines))):
            am2 = re.search(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)", lines[j])
            if am2: arxiv = f"https://arxiv.org/abs/{am2.group(1)}"; break
        papers.append({
            "id": pid(f"{conf_tag} {year}", title), "title": title,
            "conference": f"{conf_tag} {year}", "category": current_award,
            "abstract": "", "arxiv_url": arxiv,
            "crawled_at": datetime.now(timezone.utc).isoformat(),
        })

    # Filter only 2025-2026
    papers = [p for p in papers if "2025" in p["conference"] or "2026" in p["conference"]]
    print(f"  [Best-Papers] {len(papers)} awards (2025-26)")
    return papers

# ── Source 3: ICRA 2025 ─────────────────────────────────────────────

def parse_icra_2025():
    md = fetch("https://raw.githubusercontent.com/DoongLi/ICRA2025-Paper-List/main/README.md")
    papers = []
    for pm in re.finditer(r"^\|\s*(.+?)\s*\|\s*.*?\s*\|\s*(.+?)\s*\|", md, re.MULTILINE):
        title = pm.group(1).strip()
        session = pm.group(2).strip()
        if title == "Title" or "Award Finalist" not in session: continue  # only oral
        papers.append({
            "id": pid("ICRA 2025", title), "title": title,
            "conference": "ICRA 2025", "category": "Best Paper Finalist",
            "abstract": "", "arxiv_url": None,
            "crawled_at": datetime.now(timezone.utc).isoformat(),
        })
    print(f"  [ICRA 2025] {len(papers)} award finalists (oral)")
    return papers

# ── Source 4: IROS 2025 ─────────────────────────────────────────────

def parse_iros_2025():
    md = fetch("https://raw.githubusercontent.com/DoongLi/IROS2025-Paper-List/main/README.md")
    papers = []
    for pm in re.finditer(r"^\|\s*(.+?)\s*\|\s*.*?\s*\|\s*(.+?)\s*\|", md, re.MULTILINE):
        title = pm.group(1).strip()
        session = pm.group(2).strip()
        if title == "Title" or "Award Finalist" not in session: continue  # only oral
        papers.append({
            "id": pid("IROS 2025", title), "title": title,
            "conference": "IROS 2025", "category": "Best Paper Finalist",
            "abstract": "", "arxiv_url": None,
            "crawled_at": datetime.now(timezone.utc).isoformat(),
        })
    print(f"  [IROS 2025] {len(papers)} award finalists (oral)")
    return papers

# ── Merge ──────────────────────────────────────────────────────────

PR = {"Best Paper": 0, "Best Student Paper": 1, "Best Paper Honorable Mention": 2,
      "Best Paper Finalist": 3, "Oral": 4, "Accepted": 99}

def merge(existing, new_papers):
    by_id = {p["id"]: p for p in existing["papers"]}
    added = updated = 0
    for np in new_papers:
        if np["id"] in by_id:
            ep = by_id[np["id"]]
            if not ep.get("arxiv_url") and np.get("arxiv_url"): ep["arxiv_url"] = np["arxiv_url"]; updated += 1
            if PR.get(np["category"], 99) < PR.get(ep["category"], 99): ep["category"] = np["category"]; updated += 1
        else:
            by_id[np["id"]] = np; added += 1
    result = sorted(by_id.values(), key=lambda p: (p["conference"], p["title"]))
    print(f"  Merge: +{added} new, {updated} updated, {len(result)} total")
    return {"papers": result, "metadata": existing["metadata"]}

# ── Arxiv Enrich (for papers without arxiv link) ───────────────────

def arxiv_search_title(title):
    import threading
    lock = threading.Lock()
    q = re.sub(r"[^\x00-\x7F]+", "", title.strip())[:200]
    if not q: return None, None
    from urllib.parse import quote
    url = f"https://export.arxiv.org/api/query?search_query=ti:{quote(q)}&max_results=3"
    for _ in range(2):
        try:
            with lock: time.sleep(3.5)
            xml = urlopen(Request(url, headers={"User-Agent": "a/1"})).read().decode("utf-8", errors="replace")
            for link, etitle, summary in re.findall(r"<entry>.*?<id>(.*?)</id>.*?<title>(.*?)</title>.*?<summary>(.*?)</summary>", xml, re.DOTALL):
                link = link.strip(); etitle = re.sub(r"\s+", " ", etitle.strip())
                if "arxiv.org/abs/" in link:
                    a = re.sub(r"[^a-z0-9\s]", "", q.lower()); b = re.sub(r"[^a-z0-9\s]", "", etitle.lower())
                    aw, bw = set(a.split()), set(b.split())
                    if aw and bw and len(aw & bw) / min(len(aw), len(bw)) >= 0.6:
                        return re.sub(r"(v\d+)$", "", link.rstrip("/")).split("?")[0], summary.strip()
            return None, None
        except Exception:
            time.sleep(5)
    return None, None

def enrich(data):
    todo = [p for p in data["papers"] if not p.get("arxiv_url")]
    if not todo: return
    print(f"\n  Enriching {len(todo)} papers (arxiv link lookup)...")
    f = 0
    for i, p in enumerate(todo):
        if i % 20 == 0 and i > 0: print(f"    {i}/{len(todo)} ({f} found)")
        url, abstract = arxiv_search_title(p["title"])
        if url: p["arxiv_url"] = url; f += 1
        if abstract and not p.get("abstract"): p["abstract"] = abstract
    print(f"  Enrich done ({f} links found)")

# ── README ─────────────────────────────────────────────────────────

CAT = {"Best Paper": "Best Paper", "Best Student Paper": "Best Paper",
       "Best Paper Honorable Mention": "Best Paper", "Best Paper Finalist": "Best Paper"}

def gen_readme(data):
    papers = data["papers"]
    last = (data["metadata"].get("last_updated") or "")[:10]

    groups = {}
    for p in papers:
        cat = CAT.get(p["category"], p["category"])
        groups.setdefault((p["conference"], cat), []).append(p)

    cp = {"Best Paper": 0, "Oral": 1, "Accepted": 2}
    keys = sorted(groups.keys(), key=lambda k: (k[0], cp.get(k[1], 99)))

    total = len(papers)
    best = sum(1 for p in papers if "best" in p["category"].lower())

    lines = [
        "# Awesome Robotics Papers\n",
        f"**{total}** papers from top AI/ML/CV/Robotics conferences.\n",
        f"Updated: {last}  |  Best Paper awards: {best}  |  Oral: {total - best}\n",
        "## Contents\n",
    ]
    seen = set()
    for conf, _ in keys:
        if conf not in seen:
            a = conf.lower().replace(" ", "-")
            lines.append(f"- [{conf}](#{a})")
            seen.add(conf)
    lines.append("")

    cur = None
    for conf, cat in keys:
        ps = sorted(groups[(conf, cat)], key=lambda x: x["title"])
        if conf != cur:
            lines.append(f"## {conf}\n"); cur = conf
        lines.append(f"### {cat}\n")
        lines.append("| # | Title | Links |")
        lines.append("|---|-------|-------|")
        for i, p in enumerate(ps, 1):
            l = f"[arXiv]({p['arxiv_url']})" if p.get("arxiv_url") else "—"
            lines.append(f"| {i} | {p['title']} | {l} |")
        lines.append("")

    lines.append("---\nAuto-generated by [crawl_papers.py](scripts/crawl_papers.py)\n")
    (BASE / "README.md").write_text("\n".join(lines))
    print(f"\n  ✓ README.md ({len(papers)} papers)")

# ── Obsidian Vault ─────────────────────────────────────────────────

def parse_fm(text):
    if not text.startswith("---\n"): return {}, text
    end = text.find("\n---\n", 4)
    if end == -1: return {}, text
    fm = {}
    for line in text[4:end].split("\n"):
        if ":" in line: k, _, v = line.partition(":"); fm[k.strip()] = v.strip().strip('"')
    return fm, text[end + 5:]

def sync_vault(data):
    VAULT_DIR.mkdir(exist_ok=True)
    seen, created, preserved, skipped = set(), 0, 0, 0

    for p in data["papers"]:
        seen.add(p["id"])
        cat = CAT.get(p["category"], p["category"])
        conf = p["conference"].replace(" ", "-")
        folder = VAULT_DIR / conf / cat
        folder.mkdir(parents=True, exist_ok=True)
        fp = folder / f"{slug(p['title'])}.md"

        year = p["conference"].split()[1] if p["conference"].split() else "—"
        arxiv_url = p.get("arxiv_url")
        # Build new frontmatter
        fm_new = (
            f'title: "{p["title"]}"\nconference: "{p["conference"]}"\n'
            f'year: {year}\ncategory: "{cat}"\nid: "{p["id"]}"'
            + (f'\narxiv: "{arxiv_url}"' if arxiv_url else "")
        )

        if fp.exists():
            old_fm, body = parse_fm(fp.read_text())
            # Only rewrite if frontmatter changed
            old_fm_str = "\n".join(f"{k}: {v}" for k, v in sorted(old_fm.items()))
            if old_fm_str == fm_new:
                skipped += 1
                continue
            preserved += 1
        else:
            body = "## Summary\n\n\n## Notes"
            created += 1

        fml = ["---", fm_new, "---\n", body.strip() + "\n"]
        fp.write_text("\n".join(fml))

    for root, dirs, files in os.walk(VAULT_DIR, topdown=False):
        for fn in files:
            if not fn.endswith(".md"): continue
            fp = Path(root) / fn
            m = re.search(r'id:\s*"?([a-f0-9]{12})', fp.read_text())
            if m and m.group(1) not in seen: fp.unlink()
        if not os.listdir(root): os.rmdir(root)

    print(f"  ✓ Vault ({created} new, {preserved} updated, {skipped} unchanged)")

# ── Main ───────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("Awesome Robotics Papers Crawler")
    print("=" * 50)

    # Load existing
    if DATA_FILE.exists():
        with open(DATA_FILE) as f: data = json.load(f)
    else:
        data = {"papers": [], "metadata": {"last_updated": None}}
    print(f"Existing: {len(data['papers'])} papers")

    # Collect
    all_new = []
    for name, fn in [
        ("Embodied-AI-TopConf", parse_embodied_ai),
        ("Best-Papers", parse_best_papers),
        ("ICRA-2025", parse_icra_2025),
        ("IROS-2025", parse_iros_2025),
    ]:
        try:
            all_new.extend(fn())
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    # Merge + enrich + save
    data = merge(data, all_new)
    enrich(data)
    data["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()
    for p in data["papers"]:
        if p.get("arxiv_url"):
            m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)", p["arxiv_url"])
            if m: p["arxiv_url"] = f"https://arxiv.org/abs/{m.group(1)}"
    with open(DATA_FILE, "w") as f: json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved {len(data['papers'])} papers")

    # Generate
    gen_readme(data)
    sync_vault(data)
    print(f"\n✓ Done! {len(data['papers'])} papers total")

if __name__ == "__main__":
    main()
