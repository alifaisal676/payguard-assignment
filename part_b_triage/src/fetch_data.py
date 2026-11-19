"""Fetch a fixed, reproducible slice of CFPB complaints into data/complaints_raw.json.

Why not the CFPB API (as originally planned)? On 14 Aug 2026 the CFPB stopped publishing
complaint narratives; the public search API now returns metadata only (product, issue, dates)
and rejects `complaint_what_happened` as a field. Narratives published before that date are
archived (CFPB FOIA reading room) and mirrored on Hugging Face.

Source used: Hugging Face dataset `Mouwiya/cfpb-consumer-complaints`, a Parquet conversion
of the CFPB public download, read page by page through the HF dataset-server `/rows` API
(no 370 MB bulk download). Provenance is checked by looking up a sample of complaint IDs
in the official CFPB API and comparing product, issue and date.

Sampling (fixed, so the slice is reproducible):
- read MAX_PAGES pages of 100 rows at seeded random offsets (SEED)
- keep rows with a narrative, received 2024-01-01 or later, de-duplicated by complaint_id
- stratify by product with the quotas in PRODUCT_QUOTAS (total 500)

Usage (from the repo root):
    python -m part_b_triage.src.fetch_data            # writes part_b_triage/data/complaints_raw.json
    python -m part_b_triage.src.fetch_data --verify 40
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_PATH = DATA_DIR / "complaints_raw.json"

HF_DATASET = "Mouwiya/cfpb-consumer-complaints"
HF_ROWS = "https://datasets-server.huggingface.co/rows"
HF_INFO = f"https://huggingface.co/api/datasets/{HF_DATASET}"
CFPB_API = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"

SEED = 2026
PAGE = 100
MAX_PAGES = 400
MIN_DATE = "2024-01-01"
OTHER = "Other products"
PRODUCT_QUOTAS = {
    "Credit card": 90,
    "Checking or savings account": 90,
    "Money transfer, virtual currency, or money service": 70,
    "Credit reporting or other personal consumer reports": 90,
    "Debt collection": 80,
    OTHER: 80,   # loans, mortgages, prepaid cards, etc.
}
KEEP = ["complaint_id", "date_received", "product", "sub_product", "issue", "sub_issue",
        "company", "state", "tags", "submitted_via", "consumer_complaint_narrative"]


def _get(url, params, tries=6):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=120)
            if r.ok:
                return r.json()
            print(f"  HTTP {r.status_code}: {r.text[:120]}")
        except requests.RequestException as e:
            print(f"  network error: {e.__class__.__name__}")
        time.sleep(min(60, 5 * 2 ** i))
    raise RuntimeError(f"giving up on {url} {params}")


def dataset_info() -> dict:
    info = _get(HF_INFO, {})
    return {"dataset": HF_DATASET, "revision": info.get("sha"), "last_modified": info.get("lastModified")}


def bucket(product: str) -> str:
    return product if product in PRODUCT_QUOTAS else OTHER


def fetch(max_pages: int = MAX_PAGES) -> list[dict]:
    first = _get(HF_ROWS, {"dataset": HF_DATASET, "config": "default", "split": "train", "offset": 0, "length": 1})
    n_total = first["num_rows_total"]
    offsets = np.random.default_rng(SEED).integers(0, n_total - PAGE, size=max_pages)
    picked, seen, counts = [], set(), Counter()
    for p, off in enumerate(offsets, 1):
        page = _get(HF_ROWS, {"dataset": HF_DATASET, "config": "default", "split": "train", "offset": int(off), "length": PAGE})
        for item in page["rows"]:
            row = item["row"]
            text, cid = (row.get("consumer_complaint_narrative") or "").strip(), row.get("complaint_id")
            b = bucket(row.get("product") or "")
            if (len(text) < 100 or cid in seen or (row.get("date_received") or "") < MIN_DATE
                    or counts[b] >= PRODUCT_QUOTAS[b]):
                continue
            seen.add(cid); counts[b] += 1
            picked.append({k: row.get(k) for k in KEEP} | {"product_group": b})
        if p % 20 == 0:
            print(f"page {p}/{max_pages}: {len(picked)} kept {dict(counts)}")
        if all(counts[b] >= q for b, q in PRODUCT_QUOTAS.items()):
            break
    print(f"Read {p} pages; kept {len(picked)}: {dict(counts)}")
    return sorted(picked, key=lambda r: r["complaint_id"])


def verify_against_cfpb(rows: list[dict], n: int = 40, seed: int = SEED) -> dict:
    """Look up a random sample of complaint IDs in the official CFPB API; compare metadata."""
    sample = np.random.default_rng(seed).choice(len(rows), size=min(n, len(rows)), replace=False)
    result = Counter()
    for i in sample:
        r = rows[i]
        try:
            hits = requests.get(CFPB_API + str(r["complaint_id"]), timeout=60).json()["hits"]["hits"]
        except Exception:
            result["lookup error"] += 1; continue
        if not hits:
            result["not found in CFPB API"] += 1; continue
        src = hits[0]["_source"]
        same = (src.get("product") == r["product"] and src.get("issue") == r["issue"]
                and (src.get("date_received") or "")[:10] == (r["date_received"] or "")[:10])
        result["match" if same else "mismatch"] += 1
    return dict(result)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", type=int, default=40, help="how many IDs to check against the CFPB API")
    args = ap.parse_args()
    DATA_DIR.mkdir(exist_ok=True)
    meta = dataset_info() | {"seed": SEED, "page": PAGE, "min_date": MIN_DATE, "quotas": PRODUCT_QUOTAS,
                             "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    rows = fetch()
    meta["cfpb_verification"] = verify_against_cfpb(rows, args.verify)
    print("CFPB API check:", meta["cfpb_verification"])
    RAW_PATH.write_text(json.dumps({"meta": meta, "complaints": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(rows)} complaints to {RAW_PATH}")
