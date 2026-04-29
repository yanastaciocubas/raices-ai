"""
source_wikimedia.py  Pull candidate images from Wikimedia Commons by category.

For each motif in the taxonomy, looks up its associated Commons categories
(via data/commons_categories.json), filters by license and file type, and
downloads up to N images per motif with full provenance metadata.

Usage:
    python source_wikimedia.py
    python source_wikimedia.py --dry-run
    python source_wikimedia.py --motifs papel_picado ofrenda
    python source_wikimedia.py --per-motif 5
    python source_wikimedia.py --categories data/commons_categories.json --out data/eval_set

Output:
    <out>/images/<motif_id>/<filename>     downloaded image files
    <out>/candidates.csv                   provenance log (one row per image)
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import unquote

import requests


API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "Raices-Eval-Sourcing/0.1 (research; contact via project repo)"

# Substring matches against the LicenseShortName field.
ACCEPTED_LICENSE_KEYWORDS = ["CC0", "CC BY", "Public domain", "PD"]
REJECTED_LICENSE_KEYWORDS = ["Fair use", "Non-commercial", "ND", "No derivatives"]

ACCEPTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

CSV_FIELDS = [
    "image_path",
    "motif_id_candidate",
    "wikimedia_title",
    "source_page_url",
    "thumb_url",
    "license",
    "author",
    "description",
    "credit",
]


# ----------------------------------------------------------------------------
# API helpers
# ----------------------------------------------------------------------------

def list_category_files(category: str, limit: int = 50) -> List[str]:
    """Return file titles in a Commons category (direct members only, no recursion)."""
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmtype": "file",
        "cmlimit": str(min(limit, 500)),
    }
    try:
        r = requests.get(API, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  ! API error listing '{category}': {e}", file=sys.stderr)
        return []
    if "error" in data:
        print(f"  ! API error for '{category}': {data['error'].get('info', '')}", file=sys.stderr)
        return []
    return [m["title"] for m in data.get("query", {}).get("categorymembers", [])]


def get_image_metadata(file_titles: List[str]) -> List[Dict]:
    """Fetch URL, license, author, and description for a batch of file titles."""
    if not file_titles:
        return []
    out = []
    for i in range(0, len(file_titles), 50):
        batch = file_titles[i:i + 50]
        params = {
            "action": "query",
            "format": "json",
            "titles": "|".join(batch),
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|mime",
            "iiurlwidth": "1024",
        }
        try:
            r = requests.get(API, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  ! API error fetching imageinfo: {e}", file=sys.stderr)
            continue
        for page in data.get("query", {}).get("pages", {}).values():
            ii_list = page.get("imageinfo", [])
            if not ii_list:
                continue
            ii = ii_list[0]
            ext = ii.get("extmetadata", {})
            title = page.get("title", "")
            out.append({
                "title": title,
                "url": ii.get("thumburl") or ii.get("url"),
                "original_url": ii.get("url"),
                "mime": ii.get("mime", ""),
                "license": ext.get("LicenseShortName", {}).get("value", ""),
                "author": _strip_html(ext.get("Artist", {}).get("value", "")),
                "description": _strip_html(ext.get("ImageDescription", {}).get("value", "")),
                "credit": _strip_html(ext.get("Credit", {}).get("value", "")),
                "page_url": "https://commons.wikimedia.org/wiki/" + title.replace(" ", "_"),
            })
        time.sleep(0.2)  # polite pacing
    return out


def _strip_html(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ----------------------------------------------------------------------------
# Filtering & download
# ----------------------------------------------------------------------------

def rejection_reason(meta: Dict) -> Optional[str]:
    """Return a reason string if the image should be skipped, else None."""
    if not meta.get("url"):
        return "no URL"
    mime = meta.get("mime", "")
    if not mime.startswith("image/"):
        return f"non-image MIME ({mime})"
    title = meta.get("title", "")
    ext = os.path.splitext(title)[1].lower()
    if ext not in ACCEPTED_EXTENSIONS:
        return f"extension {ext} not accepted"
    license_text = meta.get("license", "")
    if any(rej.lower() in license_text.lower() for rej in REJECTED_LICENSE_KEYWORDS):
        return f"rejected license: {license_text}"
    if not any(acc.lower() in license_text.lower() for acc in ACCEPTED_LICENSE_KEYWORDS):
        return f"unrecognized license: {license_text}"
    return None


def safe_filename(title: str) -> str:
    if title.startswith("File:"):
        title = title[5:]
    title = unquote(title)
    for ch in r'\/:*?"<>|':
        title = title.replace(ch, "_")
    return title


def download_image(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60, stream=True)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  ! download failed for {url}: {e}", file=sys.stderr)
        return False


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default="data/commons_categories.json",
                    help="JSON mapping motif_id to a list of Commons categories.")
    ap.add_argument("--out", default="data/eval_set",
                    help="Output directory for images and candidates.csv.")
    ap.add_argument("--per-motif", type=int, default=10,
                    help="Target number of images to keep per motif.")
    ap.add_argument("--motifs", nargs="+", default=None,
                    help="Subset of motif IDs to process (default: all in mapping).")
    ap.add_argument("--dry-run", action="store_true",
                    help="List candidates without downloading.")
    args = ap.parse_args()

    cats_path = Path(args.categories)
    if not cats_path.exists():
        print(f"Categories file not found: {cats_path}", file=sys.stderr)
        sys.exit(1)
    with open(cats_path) as f:
        motif_categories = json.load(f)

    # Drop the README key
    motif_categories = {k: v for k, v in motif_categories.items() if not k.startswith("_")}

    out_dir = Path(args.out)
    candidates_csv = out_dir / "candidates.csv"

    # Resume support: load any previously seen titles so we do not re-download.
    seen_titles = set()
    if candidates_csv.exists() and not args.dry_run:
        with open(candidates_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen_titles.add(row["wikimedia_title"])
        print(f"Resuming: {len(seen_titles)} titles already in {candidates_csv}")

    targets = args.motifs or list(motif_categories.keys())

    csv_file = None
    writer = None
    if not args.dry_run:
        is_new = not candidates_csv.exists()
        candidates_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(candidates_csv, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()

    totals = {"kept": 0, "rejected": 0, "skipped_seen": 0}

    for motif_id in targets:
        cats = motif_categories.get(motif_id, [])
        if not cats:
            print(f"[{motif_id}] no categories configured, skipping")
            continue
        print(f"\n[{motif_id}] categories: {cats}")
        kept = 0

        for cat in cats:
            if kept >= args.per_motif:
                break
            titles = list_category_files(cat, limit=50)
            print(f"  category '{cat}': {len(titles)} files listed")
            metas = get_image_metadata(titles)

            for meta in metas:
                if kept >= args.per_motif:
                    break
                if meta["title"] in seen_titles:
                    totals["skipped_seen"] += 1
                    continue
                reason = rejection_reason(meta)
                if reason:
                    totals["rejected"] += 1
                    continue
                seen_titles.add(meta["title"])
                fname = safe_filename(meta["title"])
                rel_path = f"images/{motif_id}/{fname}"
                full_path = out_dir / rel_path

                if args.dry_run:
                    print(f"    DRY: {meta['title']} ({meta['license']})")
                    kept += 1
                    continue

                if not download_image(meta["url"], full_path):
                    continue
                writer.writerow({
                    "image_path": rel_path,
                    "motif_id_candidate": motif_id,
                    "wikimedia_title": meta["title"],
                    "source_page_url": meta["page_url"],
                    "thumb_url": meta["url"],
                    "license": meta["license"],
                    "author": meta["author"],
                    "description": meta["description"],
                    "credit": meta["credit"],
                })
                csv_file.flush()
                kept += 1
                totals["kept"] += 1
                print(f"    ok  {fname} ({meta['license']})")

        print(f"[{motif_id}] kept {kept}/{args.per_motif}")

    if csv_file:
        csv_file.close()
    print(f"\nDone. kept={totals['kept']}, rejected={totals['rejected']}, "
          f"skipped_seen={totals['skipped_seen']}")
    print(f"Provenance log: {candidates_csv}")


if __name__ == "__main__":
    main()