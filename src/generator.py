from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()


def get_llm():
    """
    Initialize Groq LLM client.
    Uses llama3-70b which is free and very capable.
    """
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1  # low temperature = more factual, less creative
    )
    return llm


def create_prompt_template():
    """
    Create the prompt that instructs the LLM how to answer.
    This is the most important part — prompt engineering.
    """
    template = """You are an expert AI research assistant. 
You help researchers understand and compare academic papers.

You have been given relevant excerpts from research papers to answer the user's question.
Base your answer ONLY on the provided context. 
If the context doesn't contain enough information, say so clearly.
Always cite which paper your information comes from.

CONTEXT FROM PAPERS:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
- Answer the question thoroughly based on the context
- Cite specific papers when making claims (use paper titles)
- If multiple papers agree or disagree, highlight that
- If the answer isn't in the context, say "The provided papers don't address this directly"
- Use clear, academic but accessible language

YOUR ANSWER:"""

    prompt = ChatPromptTemplate.from_template(template)
    return prompt


def generate_answer(question: str, context: str) -> str:
    """
    Generate an answer to the question using retrieved context.
    """
    llm = get_llm()
    prompt = create_prompt_template()
    
    # Create the chain
    chain = prompt | llm
    
    print("Generating answer...")
    
    response = chain.invoke({
        "context": context,
        "question": question
    })
    
    return response.content


def answer_question(question: str, vector_store) -> dict:
    """
    Complete pipeline: retrieve + generate.
    Returns answer and source information.
    """
    from src.retriever import retrieve_relevant_chunks, format_context
    
    # Step 1: Retrieve relevant chunks
    chunks = retrieve_relevant_chunks(vector_store, question, k=5)
    
    # Step 2: Format into context
    context = format_context(chunks)
    
    # Step 3: Generate answer
    answer = generate_answer(question, context)
    
    # Step 4: Package result
    result = {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "title": chunk["metadata"].get("title", "Unknown"),
                "arxiv_id": chunk["metadata"].get("arxiv_id", ""),
                "authors": chunk["metadata"].get("authors", ""),
                "similarity_score": chunk["similarity_score"]
            }
            for chunk in chunks
        ],
        "retrieved_chunks": chunks
    }
    
    return result

def summarize_paper(paper_metadata: dict, vector_store) -> str:
    """
    Generate a one-paragraph summary of a specific paper.
    Retrieves chunks only from that paper then summarizes.
    """
    from src.retriever import retrieve_relevant_chunks
    
    # Search specifically for this paper's content
    paper_title = paper_metadata.get("title", "")
    
    # Get chunks from this specific paper using its title as filter
    results = vector_store.similarity_search(
        query=f"main contributions methodology results conclusion of {paper_title}",
        k=6,
        filter={"title": paper_title}
    )
    
    if not results:
        # Fallback without filter
        results = vector_store.similarity_search(
            query=f"main contributions methodology results of {paper_title}",
            k=4
        )
    
    # Build context from retrieved chunks
    context = "\n\n".join([doc.page_content for doc in results])
    
    llm = get_llm()
    
    summary_prompt = ChatPromptTemplate.from_template("""
You are a research assistant. Based on the following excerpts from a research paper, 
write exactly ONE clear paragraph (4-6 sentences) summarizing:
- What problem this paper addresses
- What approach or method they propose
- What results or findings they report

Be concise and precise. Do not use bullet points. Write in plain academic English.

Paper Title: {title}

Paper Excerpts:
{context}

ONE PARAGRAPH SUMMARY:""")
    
    chain = summary_prompt | llm
    
    response = chain.invoke({
        "title": paper_title,
        "context": context[:3000]  # limit context size
    })
    
    return response.content