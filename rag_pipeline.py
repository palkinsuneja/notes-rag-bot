import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

NOTES_FOLDER = "sample_notes"
CHROMA_DB_PATH = "./chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 3
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_and_chunk_documents(folder_path: str):
    documents = []
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    for txt_file in folder.glob("*.txt"):
        loader = TextLoader(str(txt_file), encoding="utf-8")
        documents.extend(loader.load())
        print(f"  Loaded: {txt_file.name}")
    for pdf_file in folder.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())
        print(f"  Loaded: {pdf_file.name}")
    if not documents:
        raise ValueError(f"No .txt or .pdf files found in {folder_path}")
    print(f"Total documents loaded: {len(documents)}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

def build_vector_store(chunks, persist_directory: str = CHROMA_DB_PATH):
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print("Embedding chunks and storing in ChromaDB...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print(f"Done! Vector DB saved at: {persist_directory}")
    return vectordb

def load_existing_vector_store(persist_directory: str = CHROMA_DB_PATH):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vectordb

PROMPT_TEMPLATE = """You are a helpful study assistant. Answer the question using ONLY the context provided below. If the answer is not in the context, say "This information is not in your notes." Never hallucinate or make up answers.

Context:
{context}

Question: {question}

Answer:"""

def answer_question(vectordb, question: str):
    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})
    relevant_chunks = retriever.invoke(question)
    if not relevant_chunks:
        return {"answer": "No relevant content found in your notes.", "sources": []}
    context = "\n\n---\n\n".join([chunk.page_content for chunk in relevant_chunks])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    final_prompt = prompt.format(context=context, question=question)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    response = llm.invoke(final_prompt)
    sources = [
        {"content_preview": chunk.page_content[:150] + "...",
         "source_file": chunk.metadata.get("source", "unknown")}
        for chunk in relevant_chunks
    ]
    return {"answer": response.content, "sources": sources}

def setup_pipeline(force_rebuild: bool = False):
    db_exists = os.path.exists(CHROMA_DB_PATH) and os.listdir(CHROMA_DB_PATH)
    if db_exists and not force_rebuild:
        print("Loading existing vector DB...")
        vectordb = load_existing_vector_store()
    else:
        print("Building new vector DB...")
        chunks = load_and_chunk_documents(NOTES_FOLDER)
        vectordb = build_vector_store(chunks)
    return vectordb