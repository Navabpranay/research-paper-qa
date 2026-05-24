import json
import os


def save_paper_summaries(summaries: dict, filepath: str = "paper_summaries.json"):
    """
    Save generated summaries to disk so we don't regenerate every time.
    summaries = {arxiv_id_or_filename: {"title": ..., "summary": ...}}
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"Summaries saved to {filepath}")


def load_paper_summaries(filepath: str = "paper_summaries.json") -> dict:
    """
    Load previously generated summaries from disk.
    Returns empty dict if file doesn't exist.
    """
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_all_summaries(papers: list, vector_store) -> dict:
    """
    Generate LLM summaries for all papers.
    Skips papers that already have saved summaries.
    Returns dict of paper_id -> summary text.
    """
    from src.generator import summarize_paper
    
    # Load existing summaries to avoid regenerating
    existing = load_paper_summaries()
    summaries = dict(existing)
    
    new_count = 0
    
    for paper in papers:
        paper_id = paper.get("arxiv_id") or paper.get("title", "unknown")
        
        # Skip if already summarized
        if paper_id in summaries:
            print(f"Summary already exists for: {paper['title'][:50]}")
            continue
        
        print(f"Generating summary for: {paper['title'][:50]}...")
        
        try:
            summary_text = summarize_paper(paper, vector_store)
            summaries[paper_id] = {
                "title": paper["title"],
                "authors": paper.get("authors", []),
                "published": paper.get("published", ""),
                "arxiv_id": paper.get("arxiv_id", ""),
                "summary": summary_text
            }
            new_count += 1
        except Exception as e:
            print(f"Could not summarize {paper['title'][:40]}: {e}")
            summaries[paper_id] = {
                "title": paper["title"],
                "authors": paper.get("authors", []),
                "published": paper.get("published", ""),
                "arxiv_id": paper.get("arxiv_id", ""),
                "summary": paper.get("abstract", "Summary not available.")[:500]
            }
    
    # Save updated summaries
    save_paper_summaries(summaries)
    print(f"Generated {new_count} new summaries.")
    
    return summaries