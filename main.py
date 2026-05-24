from dotenv import load_dotenv
from src.fetcher import fetch_and_download_papers
from src.processor import process_papers
from src.embedder import create_vector_store, load_vector_store, add_papers_to_store
from src.generator import answer_question
from src.evaluator import log_interaction, run_test_suite
import os

load_dotenv()


def ingest_papers(query: str, max_papers: int = 5):
    """
    Full ingestion pipeline:
    Search ArXiv → Download → Process → Store in ChromaDB
    """
    print("=" * 60)
    print(f"INGESTING PAPERS ON: {query}")
    print("=" * 60)
    
    # Fetch from ArXiv
    papers = fetch_and_download_papers(query, max_papers)
    
    # Process into chunks
    chunks = process_papers(papers)
    
    # Store in ChromaDB
    vector_store = add_papers_to_store(chunks)
    
    print(f"\nIngestion complete! {len(papers)} papers indexed.")
    return vector_store


def query_papers(question: str, vector_store):
    """
    Full query pipeline:
    Question → Retrieve → Generate → Display
    """
    print("\n" + "=" * 60)
    print(f"QUESTION: {question}")
    print("=" * 60)
    
    result = answer_question(question, vector_store)
    
    print(f"\nANSWER:\n{result['answer']}")
    
    print("\nSOURCES USED:")
    for i, source in enumerate(result["sources"], 1):
        print(f"  {i}. {source['title'][:60]}")
        print(f"     ArXiv ID: {source['arxiv_id']}")
        print(f"     Similarity Score: {source['similarity_score']}")
    
    log_interaction(result)
    return result


def interactive_mode(vector_store):
    """
    Interactive Q&A loop — keep asking questions until 'quit'
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE Q&A MODE")
    print("Type your question and press Enter.")
    print("Type 'quit' to exit.")
    print("=" * 60)
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        if not question:
            print("Please enter a question.")
            continue
        
        query_papers(question, vector_store)


'''if __name__ == "__main__":
    print("Script started!")
    
    # STEP 1: Choose your research topic
    RESEARCH_TOPIC = "large language model hallucination reduction"
    
    # STEP 2: Ingest papers (run once, then comment out)
    vector_store = ingest_papers(RESEARCH_TOPIC, max_papers=5)
    
    # To load existing papers instead of re-ingesting:
    # vector_store = load_vector_store()
    
    # STEP 3: Ask some test questions
    test_questions = [
        "What methods do these papers propose to reduce hallucinations in LLMs?",
        "What evaluation metrics are used to measure hallucination?",
        "What are the main causes of hallucination according to these papers?",
    ]
    
    run_test_suite(vector_store, test_questions)
    
    # STEP 4: Interactive mode
    interactive_mode(vector_store)'''

if __name__ == "__main__":
    print("Script started!")
    
    # Papers already ingested — just load existing store
    vector_store = load_vector_store()
    
    # Run test questions
    test_questions = [
        "What methods do these papers propose to reduce hallucinations in LLMs?",
        "What evaluation metrics are used to measure hallucination?",
        "What are the main causes of hallucination according to these papers?",
    ]
    
    run_test_suite(vector_store, test_questions)
    
    # Interactive mode
    interactive_mode(vector_store)