from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(filepath: str) -> str:
    """
    Extract all text from a PDF file.
    Returns the full text as a single string.
    """
    print(f"Extracting text from: {filepath}")
    
    reader = PdfReader(filepath)
    full_text = ""
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:  # some pages might be images, skip those
            full_text += f"\n[Page {page_num + 1}]\n{text}"
    
    print(f"Extracted {len(full_text)} characters from {len(reader.pages)} pages")
    return full_text


def chunk_text(text: str, paper_metadata: dict) -> list:
    """
    Split text into overlapping chunks for better retrieval.
    Returns list of LangChain Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,       # each chunk is ~800 characters
        chunk_overlap=150,    # 150 characters overlap between chunks
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Create chunks
    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[{
            "title": paper_metadata["title"],
            "authors": ", ".join(paper_metadata["authors"][:3]),
            "arxiv_id": paper_metadata["arxiv_id"],
            "published": paper_metadata["published"],
            "source": paper_metadata["local_path"]
        }]
    )
    
    print(f"Created {len(chunks)} chunks from paper: {paper_metadata['title'][:50]}...")
    return chunks


def process_papers(papers: list) -> list:
    """
    Process all papers: extract text + chunk.
    Returns all chunks from all papers combined.
    """
    all_chunks = []
    
    for paper in papers:
        text = extract_text_from_pdf(paper["local_path"])
        chunks = chunk_text(text, paper)
        all_chunks.extend(chunks)
    
    print(f"\nTotal chunks across all papers: {len(all_chunks)}")
    return all_chunks

def process_uploaded_pdf(filepath: str, paper_metadata: dict) -> list:
    """
    Process a user-uploaded PDF file.
    Same as process_papers but for a single uploaded file.
    """
    print(f"Processing uploaded file: {filepath}")
    text = extract_text_from_pdf(filepath)
    chunks = chunk_text(text, paper_metadata)
    print(f"Created {len(chunks)} chunks from uploaded file")
    return chunks