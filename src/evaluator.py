import pandas as pd
import os
from datetime import datetime


def log_interaction(result: dict, log_file: str = "evaluation_log.csv"):
    """
    Log every question-answer interaction for evaluation.
    This is your evaluation harness.
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": result["question"],
        "answer": result["answer"][:500],  # truncate long answers
        "num_sources": len(result["sources"]),
        "source_titles": " | ".join([s["title"][:30] for s in result["sources"]]),
        "avg_similarity_score": sum(
            s["similarity_score"] for s in result["sources"]
        ) / len(result["sources"]) if result["sources"] else 0
    }
    
    # Load existing log or create new one
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        new_row = pd.DataFrame([log_entry])
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = pd.DataFrame([log_entry])
    
    df.to_csv(log_file, index=False)
    print(f"Interaction logged to {log_file}")


def run_test_suite(vector_store, test_questions: list) -> pd.DataFrame:
    """
    Run a set of test questions and log all results.
    This is your prompt regression test.
    """
    from src.generator import answer_question
    
    results = []
    
    print(f"Running test suite with {len(test_questions)} questions...")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nTest {i}/{len(test_questions)}: {question[:60]}...")
        
        result = answer_question(question, vector_store)
        log_interaction(result)
        
        results.append({
            "question": question,
            "answer_length": len(result["answer"]),
            "num_sources": len(result["sources"]),
            "has_citation": any(
                s["title"] in result["answer"] 
                for s in result["sources"]
            ),
            "avg_score": sum(
                s["similarity_score"] for s in result["sources"]
            ) / len(result["sources"])
        })
    
    df = pd.DataFrame(results)
    print("\nTest Suite Results:")
    print(df.to_string())
    return df