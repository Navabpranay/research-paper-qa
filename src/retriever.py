def get_retriever(vector_store, k: int = 5):
    """
    Create a retriever from the vector store.
    k = number of chunks to retrieve per query.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    return retriever


def retrieve_relevant_chunks(vector_store, query: str, k: int = 5) -> list:
    """
    Find the most relevant chunks for a given query.
    Returns list of Document objects with content and metadata.
    """
    print(f"Searching for chunks relevant to: {query[:60]}...")
    
    results = vector_store.similarity_search_with_score(
        query=query,
        k=k
    )
    
    chunks_with_scores = []
    for doc, score in results:
        chunks_with_scores.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "similarity_score": round(float(score), 4)
        })
    
    print(f"Retrieved {len(chunks_with_scores)} relevant chunks")
    return chunks_with_scores


def format_context(chunks: list) -> str:
    """
    Format retrieved chunks into a context string for the LLM.
    Each chunk is labeled with its source paper.
    """
    context_parts = []
    
    for i, chunk in enumerate(chunks, 1):
        source = f"[Source {i}: {chunk['metadata'].get('title', 'Unknown')[:50]}]"
        content = chunk["content"]
        context_parts.append(f"{source}\n{content}")
    
    return "\n\n---\n\n".join(context_parts)