import arxiv
import os
import time
import requests


def search_papers(query: str, max_results: int = 5) -> list:
    """
    Search ArXiv for papers matching a query.
    Returns a list of paper information dictionaries.
    """
    print(f"Searching ArXiv for: {query}")

    client = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []

    for result in client.results(search):
        paper_info = {
            "title": result.title,
            "authors": [str(author) for author in result.authors],
            "abstract": result.summary,
            "arxiv_id": result.entry_id.split("/")[-1],
            "published": str(result.published.date()),
            "pdf_url": result.pdf_url,
        }
        papers.append(paper_info)
        print(f"Found: {result.title[:60]}...")

    return papers


def download_paper(paper_info: dict, save_dir: str = "data/papers") -> str:
    """
    Download a paper PDF using requests instead of arxiv library.
    Returns the path to the downloaded file.
    """
    os.makedirs(save_dir, exist_ok=True)

    arxiv_id = paper_info["arxiv_id"]
    filename = f"{arxiv_id}.pdf"
    filepath = os.path.join(save_dir, filename)

    # Don't download if already exists
    if os.path.exists(filepath):
        print(f"Already downloaded: {filename}")
        return filepath

    print(f"Downloading: {paper_info['title'][:60]}...")

    # Use requests to download directly
    pdf_url = paper_info["pdf_url"]

    headers = {
        "User-Agent": "Mozilla/5.0 (research project)"
    }

    response = requests.get(pdf_url, headers=headers, timeout=30)

    if response.status_code == 200:
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"Saved to: {filepath}")
    else:
        raise Exception(
            f"Failed to download {pdf_url}. "
            f"Status code: {response.status_code}"
        )

    # Be polite to ArXiv servers
    time.sleep(2)

    return filepath


def fetch_and_download_papers(query: str, max_results: int = 5) -> list:
    """
    Complete pipeline: search + download.
    Returns list of dicts with paper info + local filepath.
    """
    papers = search_papers(query, max_results)

    for paper in papers:
        filepath = download_paper(paper)
        paper["local_path"] = filepath

    print(f"\nSuccessfully fetched {len(papers)} papers")
    return papers

def get_paper_abstract_summary(papers: list) -> dict:
    """
    Returns a dict of arxiv_id -> abstract (first 500 chars).
    Used as quick summary before full LLM summary is generated.
    """
    summaries = {}
    for paper in papers:
        abstract = paper.get("abstract", "")
        summaries[paper["arxiv_id"]] = abstract[:600] + "..." \
            if len(abstract) > 600 else abstract
    return summaries