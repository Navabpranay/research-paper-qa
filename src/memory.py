from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from src.generator import get_llm
from src.retriever import retrieve_relevant_chunks, format_context


def initialize_chat_history() -> list:
    """
    Initialize empty chat history.
    Returns list of message dicts.
    """
    return []


def format_history_for_display(chat_history: list) -> list:
    """
    Format chat history for Streamlit display.
    Returns list of {"role": "user/assistant", "content": "..."}
    """
    display = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            display.append({
                "role": "user",
                "content": msg.content
            })
        elif isinstance(msg, AIMessage):
            display.append({
                "role": "assistant", 
                "content": msg.content
            })
    return display


def answer_with_memory(
    question: str,
    vector_store,
    chat_history: list
) -> tuple:
    """
    Answer a question considering the full conversation history.
    Returns (answer_text, updated_chat_history, sources)
    """
    
    # Step 1: Retrieve relevant chunks
    chunks = retrieve_relevant_chunks(vector_store, question, k=5)
    context = format_context(chunks)
    
    # Step 2: Build conversation-aware prompt
    system_prompt = """You are an expert AI research assistant with memory of our conversation.
You help researchers understand academic papers.

Use the provided context from papers to answer questions.
If a follow-up question refers to something mentioned earlier in our conversation, 
use that context to give a more precise answer.
Always cite which paper your information comes from.
If the answer isn't in the context, say so clearly.

CONTEXT FROM PAPERS:
{context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    llm = get_llm()
    chain = prompt | llm
    
    # Step 3: Generate answer with history
    response = chain.invoke({
        "context": context,
        "chat_history": chat_history,
        "question": question
    })
    
    answer = response.content
    
    # Step 4: Update history
    updated_history = chat_history + [
        HumanMessage(content=question),
        AIMessage(content=answer)
    ]
    
    # Keep only last 10 exchanges to avoid token limit
    if len(updated_history) > 20:
        updated_history = updated_history[-20:]
    
    return answer, updated_history, chunks


def clear_history() -> list:
    """Reset conversation history."""
    return []