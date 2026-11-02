#!/usr/bin/env python3
"""
Website URL Extraction Tool
============================
Crawls a website starting from a given URL, discovers internal (and
optionally external) links, and saves all unique URLs to a structured
JSON file.

Usage:
    python url_crawler.py https://example.com
    python url_crawler.py https://example.com --max-pages 100 --include-external
    python url_crawler.py https://example.com --output urls.json --delay 0.5

Author: Generated with Claude
"""

import argparse
import json
import sys
import time
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag
from urllib import robotparser

import requests
from bs4 import BeautifulSoup


class URLCrawler:
    """Crawls a website (breadth-first) and collects unique URLs."""

    def __init__(
        self,
        start_url: str,
        max_pages: int = 50, # it sets a default limit of 50 pages
        include_external: bool = False,
        delay: float = 0.0,
        timeout: int = 10,
        respect_robots: bool = True,
        user_agent: str = "Mozilla/5.0 (compatible; URLCrawlerBot/1.0)",
        verbose: bool = True, # printing text to the screen (
    ):
        self.start_url = start_url
        self.start_domain = urlparse(start_url).netloc
        self.max_pages = max_pages
        self.include_external = include_external
        self.delay = delay
        self.timeout = timeout
        self.respect_robots = respect_robots
        self.verbose = verbose

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

        self.visited_pages = set()      # pages we've fetched & parsed
        self.all_urls = set()           # every unique URL discovered (links)
        self.failed_urls = []           # (url, error) pairs for pages that failed to fetch

        self.robots_parser = None
        if self.respect_robots:
            self._load_robots_txt()

    def _log(self, message: str):
        if self.verbose:
            print(message)

    def _load_robots_txt(self):
        """Attempt to load and parse robots.txt for the target domain.

        Note: urllib.robotparser's .read() fails silently on network errors
        and leaves the parser in a "disallow everything" state. We fetch the
        content ourselves first so we can tell a real 'Disallow: /' apart
        from a network failure, and fail OPEN (allow crawling) on the latter.
        """
        parsed = urlparse(self.start_url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        try:
            resp = self.session.get(robots_url, timeout=self.timeout)
            if resp.status_code != 200:
                self._log(
                    f"[robots.txt] {robots_url} returned {resp.status_code}; "
                    f"proceeding without robots restrictions."
                )
                self.robots_parser = None
                return

            rp = robotparser.RobotFileParser()
            rp.parse(resp.text.splitlines())
            self.robots_parser = rp
            self._log(f"[robots.txt] Loaded rules from {robots_url}")
        except requests.RequestException as e:
            self._log(
                f"[robots.txt] Could not fetch robots.txt ({e}); "
                f"proceeding without robots restrictions."
            )
            self.robots_parser = None

    def _allowed_by_robots(self, url: str) -> bool:
        if not self.robots_parser:
            return True
        try:
            return self.robots_parser.can_fetch("*", url)
        except Exception:
            return True

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Strip fragments (#...) and trailing slashes for consistent dedup."""
        url, _frag = urldefrag(url)  # remove #fragment
        if url.endswith("/") and url.count("/") > 2:
            url = url.rstrip("/")
        return url

    def _is_same_domain(self, url: str) -> bool:
        return urlparse(url).netloc == self.start_domain

    def _is_crawlable_page(self, url: str) -> bool:
        """Decide whether we should fetch & parse this URL for more links."""
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        # Skip obvious non-HTML resources to save time/bandwidth
        skip_extensions = (
            ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
            ".css", ".js", ".zip", ".rar", ".mp4", ".mp3", ".avi", ".mov",
            ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".woff",
            ".woff2", ".ttf", ".eot", ".json", ".xml", ".rss",
        )
        if parsed.path.lower().endswith(skip_extensions):
            return False
        if not self.include_external and not self._is_same_domain(url):
            return False
        return True

    def _extract_links(self, html: str, base_url: str):
        """Parse HTML and return a set of absolute, normalized URLs."""
        soup = BeautifulSoup(html, "html.parser")
        links = set()

        # <a href="..."> is the primary source of links
        for tag in soup.find_all("a", href=True):
            href = tag["href"].strip()
            if not href or href.startswith(("mailto:", "tel:", "javascript:")):
                continue
            absolute = urljoin(base_url, href)
            absolute = self._normalize_url(absolute)
            links.add(absolute)

        return links

    def crawl(self):
        """Run the BFS crawl and populate self.all_urls."""
        start_normalized = self._normalize_url(self.start_url)
        queue = deque([start_normalized])
        queued = {start_normalized}

        self.all_urls.add(start_normalized)

        while queue and len(self.visited_pages) < self.max_pages:
            url = queue.popleft()

            if url in self.visited_pages:
                continue
            if not self._is_crawlable_page(url):
                continue
            if not self._allowed_by_robots(url):
                self._log(f"[skip:robots] {url}")
                continue

            self._log(f"[{len(self.visited_pages) + 1}/{self.max_pages}] Crawling: {url}")

            try:
                response = self.session.get(url, timeout=self.timeout)
                self.visited_pages.add(url)

                content_type = response.headers.get("Content-Type", "")
                if response.status_code != 200 or "text/html" not in content_type:
                    continue

                new_links = self._extract_links(response.text, url)

                for link in new_links:
                    self.all_urls.add(link)
                    if (
                        link not in queued
                        and link not in self.visited_pages
                        and self._is_crawlable_page(link)
                        and (self.include_external or self._is_same_domain(link))
                    ):
                        queued.add(link)
                        queue.append(link)

            except requests.RequestException as e:
                self._log(f"[error] Failed to fetch {url}: {e}")
                self.failed_urls.append((url, str(e)))
                self.visited_pages.add(url)  # don't retry it

            if self.delay:
                time.sleep(self.delay)

        return self.all_urls

    def save_to_json(self, output_path: str):
        """Save the collected URLs to a structured JSON file."""
        sorted_urls = sorted(self.all_urls)
        data = {"urls": sorted_urls}

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return data

    def save_detailed_json(self, output_path: str):
        """Save a more detailed JSON report (pages crawled, errors, etc.)."""
        data = {
            "start_url": self.start_url,
            "total_unique_urls": len(self.all_urls),
            "pages_crawled": len(self.visited_pages),
            "include_external": self.include_external,
            "urls": sorted(self.all_urls),
            "failed_urls": [{"url": u, "error": e} for u, e in self.failed_urls],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return data


def main():
    parser = argparse.ArgumentParser(
        description="Crawl a website and extract all URLs into a JSON file."
    )
    parser.add_argument("url", help="Starting URL to crawl (e.g. https://example.com)")
    parser.add_argument(
        "--max-pages", type=int, default=50,
        help="Maximum number of pages to crawl (default: 50)"
    )
    parser.add_argument(
        "--include-external", action="store_true",
        help="Also include external (off-domain) links in the output"
    )
    parser.add_argument(
        "--output", "-o", default="urls.json",
        help="Output JSON file path (default: urls.json)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.2,
        help="Delay in seconds between requests, be polite to servers (default: 0.2)"
    )
    parser.add_argument(
        "--timeout", type=int, default=10,
        help="Request timeout in seconds (default: 10)"
    )
    parser.add_argument(
        "--ignore-robots", action="store_true",
        help="Ignore robots.txt rules (use responsibly)"
    )
    parser.add_argument(
        "--detailed", action="store_true",
        help="Save a detailed report instead of the simple {urls: [...]} format"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress logging"
    )

    args = parser.parse_args()

    if not args.url.startswith(("http://", "https://")):
        args.url = "https://" + args.url

    crawler = URLCrawler(
        start_url=args.url,
        max_pages=args.max_pages,
        include_external=args.include_external,
        delay=args.delay,
        timeout=args.timeout,
        respect_robots=not args.ignore_robots,
        verbose=not args.quiet,
    )

    print(f"Starting crawl of: {args.url}")
    print(f"Max pages: {args.max_pages} | Include external: {args.include_external}\n")

    try:
        urls = crawler.crawl()
    except KeyboardInterrupt:
        print("\nCrawl interrupted by user. Saving partial results...")

    if args.detailed:
        data = crawler.save_detailed_json(args.output)
    else:
        data = crawler.save_to_json(args.output)

    print(f"\nDone. Pages crawled: {len(crawler.visited_pages)}")
    print(f"Unique URLs found: {len(crawler.all_urls)}")
    if crawler.failed_urls:
        print(f"Failed requests: {len(crawler.failed_urls)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()