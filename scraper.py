import json
import requests
# pyrefly: ignore [missing-import]
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import time
import os

def is_multiqos_url(url):
    parsed = urlparse(url)
    # We only want to scrape the multiqos.com domain, avoiding social links
    return parsed.netloc == "multiqos.com" or parsed.netloc == "www.multiqos.com"

def scrape_url(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Kill javascript and style tags
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            text = soup.get_text(separator=' ', strip=True)
            # Basic cleanup
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            return text
        else:
            print(f"Failed to fetch {url}: Status {response.status_code}")
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    return ""

def main():
    if not os.path.exists('urls.json'):
        print("urls.json not found!")
        return

    with open('urls.json', 'r') as f:
        data = json.load(f)

    all_urls = data.get("urls", [])
    multiqos_urls = [url for url in all_urls if is_multiqos_url(url)]
    
    print(f"Found {len(multiqos_urls)} URLs to scrape.")

    documents = []
    for i, url in enumerate(multiqos_urls):
        print(f"[{i+1}/{len(multiqos_urls)}] Scraping {url}...")
        text = scrape_url(url)
        if text:
            documents.append({
                "url": url,
                "text": f"Source: {url}\n\n{text}"
            })
        time.sleep(0.5)  # Be nice to the server

    with open('document.json', 'w') as f:
        json.dump(documents, f, indent=2)

    print(f"Scraped {len(documents)} documents and saved to document.json.")

if __name__ == "__main__":
    main()
