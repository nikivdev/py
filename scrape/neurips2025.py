#!/usr/bin/env python3
"""Download NeurIPS 2025 papers and PDFs from papercopilot/OpenReview.

Usage:
    uv run python -m scrape.neurips2025
    uv run python -m scrape.neurips2025 --limit 10
    uv run python -m scrape.neurips2025 --pdfs-only  # Skip JSON, just download PDFs
"""

import json
import sys
import time
from pathlib import Path

import httpx

OUTPUT_DIR = Path.home() / "done" / "neurips" / "2025"
DATA_URL = "https://raw.githubusercontent.com/papercopilot/paperlists/main/nips/nips2025.json"
OPENREVIEW_PDF_URL = "https://openreview.net/pdf?id={paper_id}"

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 2.0
REQUEST_DELAY = 0.3  # Delay between requests to avoid rate limiting


def download_papers_list() -> list[dict]:
    """Download the NeurIPS 2025 paper list from GitHub."""
    print("Downloading NeurIPS 2025 paper list...", file=sys.stderr)
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

    for attempt in range(MAX_RETRIES):
        try:
            with httpx.Client(timeout=60) as client:
                response = client.get(DATA_URL, headers=headers)
                response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            print(f"  Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}", file=sys.stderr)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))

    print("Failed to download paper list after all retries", file=sys.stderr)
    sys.exit(1)


def download_pdf(paper_id: str, output_path: Path, client: httpx.Client) -> bool:
    """Download a PDF from OpenReview with retries."""
    url = OPENREVIEW_PDF_URL.format(paper_id=paper_id)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.get(url, follow_redirects=True)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "pdf" in content_type or response.content[:4] == b"%PDF":
                output_path.write_bytes(response.content)
                return True
            else:
                print(f"  Not a PDF (content-type: {content_type}): {paper_id}", file=sys.stderr)
                return False

        except httpx.TimeoutException:
            print(f"  Timeout (attempt {attempt + 1}/{MAX_RETRIES}): {paper_id}", file=sys.stderr)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:  # Rate limited
                wait_time = RETRY_DELAY * (attempt + 2) * 2
                print(f"  Rate limited, waiting {wait_time}s...", file=sys.stderr)
                time.sleep(wait_time)
            elif e.response.status_code == 404:
                print(f"  PDF not found: {paper_id}", file=sys.stderr)
                return False
            else:
                print(f"  HTTP {e.response.status_code} (attempt {attempt + 1}/{MAX_RETRIES}): {paper_id}", file=sys.stderr)
        except httpx.HTTPError as e:
            print(f"  Error (attempt {attempt + 1}/{MAX_RETRIES}): {paper_id} - {e}", file=sys.stderr)

        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY * (attempt + 1))

    return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download NeurIPS 2025 papers and PDFs")
    parser.add_argument("--limit", type=int, help="Limit number of papers")
    parser.add_argument("--pdfs-only", action="store_true", help="Only download PDFs (skip JSON)")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY, help="Delay between downloads (seconds)")
    args = parser.parse_args()

    # Setup output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get paper list
    papers = download_papers_list()
    total = len(papers)
    print(f"Found {total} papers", file=sys.stderr)

    if args.limit:
        papers = papers[: args.limit]
        print(f"Limited to {len(papers)} papers", file=sys.stderr)

    # Stats
    json_saved = 0
    json_skipped = 0
    pdf_downloaded = 0
    pdf_skipped = 0
    pdf_failed = 0

    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

    with httpx.Client(timeout=60, headers=headers) as client:
        for i, paper in enumerate(papers):
            paper_id = paper.get("id", str(i))
            json_file = OUTPUT_DIR / f"{paper_id}.json"
            pdf_file = OUTPUT_DIR / f"{paper_id}.pdf"

            # Save JSON (unless pdfs-only)
            if not args.pdfs_only:
                if not json_file.exists():
                    with open(json_file, "w") as f:
                        json.dump(paper, f, indent=2)
                    json_saved += 1
                else:
                    json_skipped += 1

            # Download PDF
            if pdf_file.exists():
                pdf_skipped += 1
                status = "skip"
            else:
                print(f"[{i + 1}/{len(papers)}] Downloading {paper_id}...", file=sys.stderr)
                if download_pdf(paper_id, pdf_file, client):
                    pdf_downloaded += 1
                    status = "ok"
                else:
                    pdf_failed += 1
                    status = "FAIL"
                time.sleep(args.delay)

            # Progress every 100 papers
            if (i + 1) % 100 == 0:
                print(f"\n--- Progress: {i + 1}/{len(papers)} ---", file=sys.stderr)
                print(f"    PDFs: {pdf_downloaded} downloaded, {pdf_skipped} skipped, {pdf_failed} failed", file=sys.stderr)
                if not args.pdfs_only:
                    print(f"    JSON: {json_saved} saved, {json_skipped} skipped", file=sys.stderr)
                print("", file=sys.stderr)

    # Final summary
    print(f"\n{'=' * 50}", file=sys.stderr)
    print(f"DONE! Output: {OUTPUT_DIR}", file=sys.stderr)
    print(f"PDFs: {pdf_downloaded} downloaded, {pdf_skipped} skipped, {pdf_failed} failed", file=sys.stderr)
    if not args.pdfs_only:
        print(f"JSON: {json_saved} saved, {json_skipped} skipped", file=sys.stderr)

    # Save failed list if any
    if pdf_failed > 0:
        failed_file = OUTPUT_DIR / "failed.txt"
        failed_ids = [
            p.get("id") for p in papers
            if not (OUTPUT_DIR / f"{p.get('id')}.pdf").exists()
        ]
        with open(failed_file, "w") as f:
            f.write("\n".join(failed_ids))
        print(f"Failed IDs saved to: {failed_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
