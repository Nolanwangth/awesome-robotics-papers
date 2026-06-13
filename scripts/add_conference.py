#!/usr/bin/env python3
"""Add a single conference+year to the database.
Usage: python scripts/add_conference.py ICML 2025
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from crawl_papers import (load, save, merge, enrich, generate_readme, sync_vault, crawl_conf)

if len(sys.argv) < 3:
    print("Usage: python scripts/add_conference.py <Name> <Year>")
    print("Example: python scripts/add_conference.py ICML 2025")
    sys.exit(1)

name = f"{sys.argv[1]} {sys.argv[2]}"
query = f'all:"{sys.argv[1]} {sys.argv[2]}"'

print(f"Adding: {name}")
data = load()
print(f"Existing: {len(data['papers'])} papers")

new = crawl_conf(name, query)
print(f"Crawled: {len(new)} papers")

data = merge(data, new)
data = enrich(data)
save(data)
generate_readme(data)
sync_vault(data)

print(f"\nDone! {len(data['papers'])} papers total")
