import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_DIR = "./chroma_db"

def load_documents(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".txt"):
            loader = TextLoader(path)
        else:
            continue
        docs.extend(loader.load())
    return docs

def ingest_docs(folder_path="./docs"):
    print(f"📂 Loading documents from '{folder_path}'...")
    documents = load_documents(folder_path)

    if not documents:
        print("❌ No documents found. Add .pdf or .txt files to the docs/ folder.")
        return None

    print(f"✅ Loaded {len(documents)} page(s). Splitting into chunks...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    print(f"🔍 Creating embeddings for {len(chunks)} chunks...")

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        chunks,
        embedding,
        persist_directory=CHROMA_DIR
    )

    print("✅ Vectorstore ready!\n")
    return vectorstore