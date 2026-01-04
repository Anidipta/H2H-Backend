import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

app = FastAPI(title="Farmer AI Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DISEASE_COLLECTION = "citrus_diseases"
SCHEME_COLLECTION = "government_schemes"
PERSIST_DIRECTORY = "./chroma_db"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

vectorstores_cache = {}
llm_cache = None

class ChatRequest(BaseModel):
    message: str
    chat_history: List[Dict[str, str]] = []

class ChatResponse(BaseModel):
    success: bool
    intent: str
    answer: str
    sources: str = ""

def initialize_llm():
    # Initialize Groq LLM with Llama model for conversational AI
    global llm_cache
    if llm_cache is None:
        llm_cache = ChatGroq(
            temperature=0.7,
            model_name="llama-3.3-70b-versatile",
            groq_api_key=GROQ_API_KEY
        )
    return llm_cache

def load_and_chunk_documents(pdf_path: str, collection_name: str) -> List[Document]:
    # Load PDF documents and split into chunks with metadata
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    for i, chunk in enumerate(chunks):
        chunk.metadata["collection"] = collection_name
        chunk.metadata["chunk_id"] = i
        chunk.metadata["source_file"] = pdf_path
    
    return chunks

def initialize_vector_stores() -> Dict[str, Any]:
    # Initialize or load vector stores for both knowledge bases
    global vectorstores_cache
    
    if vectorstores_cache:
        return vectorstores_cache
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    if os.path.exists(PERSIST_DIRECTORY):
        disease_vectorstore = Chroma(
            collection_name=DISEASE_COLLECTION,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        
        scheme_vectorstore = Chroma(
            collection_name=SCHEME_COLLECTION,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
    else:
        disease_chunks = load_and_chunk_documents("docs/CitrusPlantPestsAndDiseases.pdf", DISEASE_COLLECTION)
        scheme_chunks = load_and_chunk_documents("docs/GovernmentSchemes.pdf", SCHEME_COLLECTION)
        
        disease_vectorstore = Chroma.from_documents(
            documents=disease_chunks,
            embedding=embeddings,
            collection_name=DISEASE_COLLECTION,
            persist_directory=PERSIST_DIRECTORY
        )
        
        scheme_vectorstore = Chroma.from_documents(
            documents=scheme_chunks,
            embedding=embeddings,
            collection_name=SCHEME_COLLECTION,
            persist_directory=PERSIST_DIRECTORY
        )
    
    vectorstores_cache = {
        "disease": disease_vectorstore,
        "scheme": scheme_vectorstore
    }
    
    return vectorstores_cache

def classify_intent(query: str, llm: ChatGroq) -> str:
    # Classify user query intent using Groq LLM
    classification_prompt = f"""You are an intent classifier for farmer queries.

Classify the following query into ONE of these categories:
- "disease": Query about plant diseases, pests, symptoms, treatment, plant health
- "scheme": Query about government schemes, subsidies, financial aid, eligibility, applications
- "hybrid": Query about BOTH diseases/pests AND government schemes/financial support

Query: {query}

Respond with ONLY one word: disease, scheme, or hybrid"""

    result = llm.invoke(classification_prompt)
    intent = result.content.strip().lower()
    
    if intent not in ["disease", "scheme", "hybrid"]:
        intent = "disease"
    
    return intent

def retrieve_documents(query: str, vectorstores: Dict[str, Any], intent: str, k: int = 4) -> List[Document]:
    # Retrieve relevant documents based on detected intent
    if intent == "disease":
        retriever = vectorstores["disease"].as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)
    elif intent == "scheme":
        retriever = vectorstores["scheme"].as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)
    else:
        disease_retriever = vectorstores["disease"].as_retriever(search_kwargs={"k": k//2})
        scheme_retriever = vectorstores["scheme"].as_retriever(search_kwargs={"k": k//2})
        disease_docs = disease_retriever.invoke(query)
        scheme_docs = scheme_retriever.invoke(query)
        return disease_docs + scheme_docs

def format_sources(docs: List[Document]) -> str:
    # Format document sources with page numbers
    if not docs:
        return "No sources available"
    sources = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "N/A")
        sources.append(f"[{i}] Page {page}")
    return " | ".join(sources)

def generate_conversational_response(query: str, docs: List[Document], llm: ChatGroq, chat_history: List[Dict[str, str]]) -> str:
    # Generate natural conversational response with helpful follow-up suggestions
    if not docs:
        return "I couldn't find specific information about that in my knowledge base. Could you rephrase your question or ask about something related to citrus diseases or government farming schemes?"
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    history_text = ""
    if chat_history:
        history_text = "\n\nRecent conversation:\n"
        for msg in chat_history[-3:]:
            history_text += f"Farmer: {msg.get('user', '')}\nAssistant: {msg.get('assistant', '')}\n"
    
    response_prompt = f"""You are a friendly and helpful agricultural advisor having a natural conversation with a farmer. Be warm, conversational, and supportive.

{history_text}

Knowledge base information:
{context}

Current question: {query}

Instructions:
1. Answer the farmer's question clearly, to the point a bit and conversationally
2. Use simple, everyday language - avoid jargon
3. Be encouraging and supportive
4. After your main answer, add ONE helpful follow-up suggestion like:
   - "If you need help with [related topic], I'm here to assist!"
   - "Would you like to know more about [specific aspect]?"
   - "I can also help you with information about [related topic] if needed."
   - "Feel free to ask if you have questions about [related aspect]!"

Keep your response natural and friendly, like talking to a neighbor.

Answer:"""

    result = llm.invoke(response_prompt)
    return result.content

@app.on_event("startup")
async def startup_event():
    # Initialize vector stores and LLM on application startup
    print("🚀 Initializing Farmer AI Assistant...")
    initialize_vector_stores()
    initialize_llm()
    print("✅ Backend ready!")

@app.post("/query", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Main chat endpoint that processes farmer queries and returns responses
    try:
        if not request.message.strip():
            return ChatResponse(
                success=False,
                intent="unknown",
                answer="Please enter a message.",
                sources=""
            )
        
        llm = initialize_llm()
        vectorstores = initialize_vector_stores()
        
        intent = classify_intent(request.message, llm)
        
        docs = retrieve_documents(request.message, vectorstores, intent, k=4)
        
        answer = generate_conversational_response(
            request.message,
            docs,
            llm,
            request.chat_history
        )
        
        sources = format_sources(docs)
        
        return ChatResponse(
            success=True,
            intent=intent,
            answer=answer,
            sources=sources
        )
        
    except Exception as e:
        return ChatResponse(
            success=False,
            intent="error",
            answer=f"I encountered an issue: {str(e)}. Please try again.",
            sources=""
        )

@app.get("/health")
async def health_check():
    # Check if vector stores and LLM are initialized
    return {
        "status": "healthy",
        "vectorstores_loaded": bool(vectorstores_cache),
        "llm_loaded": llm_cache is not None
    }

@app.get("/")
async def root():
    return {
        "status": "Farmer AI Assistant API is running", 
        "version": "1.0",
        "endpoints": [
            {"/query": "Process farmer queries and return responses"},
            {"/health": "Check if vector stores and LLM are initialized"}
        ]
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)