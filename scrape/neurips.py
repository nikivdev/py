#!/usr/bin/env python3
"""Scrape NeurIPS 2024 papers from the virtual conference site.

Usage:
    uv run python -m scrape.neurips
    uv run python -m scrape.neurips --limit 10
    uv run python -m scrape.neurips --json
"""

import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

OUTPUT_DIR = Path.home() / "done" / "py" / "neurips-papers"

import httpx
from bs4 import BeautifulSoup


NEURIPS_VIRTUAL = "https://neurips.cc"
YEAR = 2024  # Most recent NeurIPS (December 2024)


@dataclass
class Paper:
    """A NeurIPS paper."""

    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: str | None = None
    poster_id: str | None = None


def get_paper_links(year: int = YEAR) -> list[tuple[str, str]]:
    """Get all paper links for a given year from virtual site.

    Returns list of (url, title) tuples.
    """
    url = f"{NEURIPS_VIRTUAL}/virtual/{year}/papers.html"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

    with httpx.Client(timeout=30) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    papers = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if f"/virtual/{year}/poster/" in href:
            full_url = f"{NEURIPS_VIRTUAL}{href}" if href.startswith("/") else href
            title = a.get_text(strip=True)
            if full_url not in [p[0] for p in papers]:
                papers.append((full_url, title))

    return papers


def scrape_paper(url: str, title_hint: str = "") -> Paper | None:
    """Scrape a single paper page from virtual site."""
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

    try:
        with httpx.Client(timeout=30) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
    except httpx.HTTPError as e:
        print(f"Failed to fetch {url}: {e}", file=sys.stderr)
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Title from <title> tag
    title_tag = soup.find("title")
    title = title_hint
    if title_tag:
        title_text = title_tag.get_text(strip=True)
        # Remove "NeurIPS Poster " prefix
        if "NeurIPS" in title_text:
            title = re.sub(r"^NeurIPS\s+(Poster|Oral|Spotlight)\s+", "", title_text)
        else:
            title = title_text

    # Authors - look for author links or spans
    authors = []
    author_div = soup.find("div", class_="authors") or soup.find("span", class_="author")
    if author_div:
        for a in author_div.find_all("a"):
            name = a.get_text(strip=True)
            if name and name not in authors:
                authors.append(name)

    # Try finding author names in specific pattern
    if not authors:
        for a in soup.find_all("a", href=re.compile(r"/virtual/\d+/author/")):
            name = a.get_text(strip=True)
            if name and name not in authors:
                authors.append(name)

    # Abstract - in abstract-text-inner div
    abstract = ""
    abstract_div = soup.find("div", class_="abstract-text-inner")
    if abstract_div:
        abstract = abstract_div.get_text(strip=True)
    else:
        # Fallback to any long paragraph
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 200:
                abstract = text
                break

    # PDF link
    pdf_url = None
    pdf_link = soup.find("a", href=re.compile(r"\.pdf", re.IGNORECASE))
    if pdf_link:
        href = pdf_link["href"]
        pdf_url = href if href.startswith("http") else f"{NEURIPS_VIRTUAL}{href}"

    # Extract poster ID from URL
    poster_id = None
    match = re.search(r"/poster/(\d+)", url)
    if match:
        poster_id = match.group(1)

    return Paper(
        title=title,
        authors=authors,
        abstract=abstract,
        url=url,
        pdf_url=pdf_url,
        poster_id=poster_id,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Scrape NeurIPS 2024 papers")
    parser.add_argument("--limit", type=int, help="Limit number of papers")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--year", type=int, default=YEAR, help="Year to scrape")
    parser.add_argument("--list-only", action="store_true", help="Only list papers, don't scrape details")
    args = parser.parse_args()

    print(f"Fetching NeurIPS {args.year} paper list...", file=sys.stderr)
    paper_links = get_paper_links(args.year)
    print(f"Found {len(paper_links)} papers", file=sys.stderr)

    if args.limit:
        paper_links = paper_links[: args.limit]

    if args.list_only:
        if args.json:
            print(json.dumps([{"url": url, "title": title} for url, title in paper_links], indent=2))
        else:
            for url, title in paper_links:
                print(f"{title}")
                print(f"  {url}\n")
        return

    # Setup output directory for individual papers
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    papers_dir = OUTPUT_DIR / str(args.year)
    papers_dir.mkdir(exist_ok=True)

    papers = []
    for i, (link, title) in enumerate(paper_links):
        # Extract poster ID for filename
        match = re.search(r"/poster/(\d+)", link)
        poster_id = match.group(1) if match else str(i)
        paper_file = papers_dir / f"{poster_id}.json"

        # Skip if already scraped
        if paper_file.exists():
            print(f"Skipping {i + 1}/{len(paper_links)}: {title[:50]}... (already scraped)", file=sys.stderr)
            with open(paper_file) as f:
                papers.append(json.load(f))
            continue

        print(f"Scraping {i + 1}/{len(paper_links)}: {title[:50]}...", file=sys.stderr)
        paper = scrape_paper(link, title)
        if paper:
            # Save individual paper immediately
            with open(paper_file, "w") as f:
                json.dump(asdict(paper), f, indent=2)
            papers.append(asdict(paper))

    # Also save combined file
    output_file = OUTPUT_DIR / f"neurips-{args.year}.json"
    with open(output_file, "w") as f:
        json.dump(papers, f, indent=2)
    print(f"Saved {len(papers)} papers to {output_file}", file=sys.stderr)

    if args.json:
        print(json.dumps(papers, indent=2))
    else:
        for p in papers:
            print(f"\n{'=' * 80}")
            print(f"Title: {p['title']}")
            if p['authors']:
                print(f"Authors: {', '.join(p['authors'])}")
            print(f"URL: {p['url']}")
            if p['pdf_url']:
                print(f"PDF: {p['pdf_url']}")
            if p['abstract']:
                abstract_preview = p['abstract'][:500] + "..." if len(p['abstract']) > 500 else p['abstract']
                print(f"\nAbstract: {abstract_preview}")


if __name__ == "__main__":
    main()
