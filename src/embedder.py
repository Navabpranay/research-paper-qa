from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os


def get_embedding_model():
    """
    Load the sentence transformer embedding model.
    This runs locally - no API key needed.
    """
    print("Loading embedding model (first time may take a minute to download)...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    print("Embedding model loaded!")
    return embeddings


def create_vector_store(chunks: list, persist_dir: str = "chroma_db") -> Chroma:
    """
    Create ChromaDB vector store from document chunks.
    Embeds all chunks and stores them on disk.
    """
    print(f"Creating vector store with {len(chunks)} chunks...")
    
    embeddings = get_embedding_model()
    
    # Create and persist the vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="research_papers"
    )
    
    print(f"Vector store created and saved to: {persist_dir}")
    return vector_store


def load_vector_store(persist_dir: str = "chroma_db") -> Chroma:
    """
    Load existing vector store from disk.
    Use this instead of create_vector_store if papers already ingested.
    """
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(
            f"No vector store found at {persist_dir}. "
            "Run ingestion first."
        )
    
    print("Loading existing vector store...")
    embeddings = get_embedding_model()
    
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="research_papers"
    )
    
    print("Vector store loaded!")
    return vector_store


def add_papers_to_store(chunks: list, persist_dir: str = "chroma_db") -> Chroma:
    """
    Add new papers to existing vector store (or create new one).
    """
    if os.path.exists(persist_dir):
        print("Adding to existing vector store...")
        embeddings = get_embedding_model()
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="research_papers"
        )
        vector_store.add_documents(chunks)
    else:
        vector_store = create_vector_store(chunks, persist_dir)
    
    return vector_store