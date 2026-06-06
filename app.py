import os
import gc
import shutil
import time
import chromadb
import streamlit as st
from dotenv import load_dotenv
from ingest import ingest_docs
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="AskMyDocs", page_icon="📄", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Outfit:wght@300;400;500;600&display=swap');

    :root {
        --bg:           #12102a;
        --bg-panel:     #1a1635;
        --bg-card:      #221d3d;
        --bg-input:     #1e1a38;
        --border:       #3d3370;
        --border-soft:  #2e2550;
        --accent:       #7c3aed;
        --accent-2:     #9d5cf6;
        --text-1:       #ede9fe;
        --text-2:       #c4b5fd;
        --text-3:       #9580d4;
        --success:      #34d399;
        --success-bg:   #0d2e22;
        --warning:      #fbbf24;
        --warning-bg:   #2a1f06;
        --tag-bg:       #2e2550;
        --tag-text:     #c4b5fd;
        --tag-border:   #4c3d8a;
    }

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
        background-color: var(--bg) !important;
        color: var(--text-1) !important;
    }

    .stApp {
        background-color: var(--bg) !important;
        background-image: radial-gradient(circle, rgba(124,58,237,0.07) 1px, transparent 1px);
        background-size: 28px 28px;
    }

    .block-container {
        padding-top: 3rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        max-width: 100% !important;
    }

    .main-title {
        font-family: 'Cinzel', serif !important;
        font-size: 3.2rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em !important;
        margin: 0 0 0.3rem 0 !important;
        line-height: 1.15 !important;
        background: linear-gradient(135deg, #ede9fe 30%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .subtitle {
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.92rem !important;
        color: var(--text-3) !important;
        font-weight: 300 !important;
        margin: 0 0 2.2rem 0 !important;
        letter-spacing: 0.04em !important;
    }

    .user-bubble {
        background: var(--accent);
        border-radius: 18px 18px 4px 18px;
        padding: 12px 18px;
        margin: 10px 0 10px auto;
        max-width: 70%;
        width: fit-content;
        color: #ffffff;
        font-size: 0.95rem;
        line-height: 1.55;
        font-family: 'Outfit', sans-serif;
    }

    .bot-bubble {
        background: var(--bg-card);
        border: 0.5px solid var(--border);
        border-radius: 18px 18px 18px 4px;
        padding: 14px 18px;
        margin: 10px auto 10px 0;
        max-width: 80%;
        width: fit-content;
        color: var(--text-1);
        font-size: 0.95rem;
        line-height: 1.65;
        font-family: 'Outfit', sans-serif;
    }

    .bot-label {
        font-family: 'Cinzel', serif;
        font-size: 0.62rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        color: var(--text-3);
        margin-bottom: 7px;
        text-transform: uppercase;
    }

    .source-tag {
        display: inline-block;
        background: var(--tag-bg);
        border: 0.5px solid var(--tag-border);
        color: var(--tag-text);
        font-size: 0.71rem;
        font-weight: 500;
        padding: 3px 11px;
        border-radius: 20px;
        margin: 5px 3px 0 0;
        font-family: 'Outfit', sans-serif;
    }

    [data-testid="stSidebar"] {
        background-color: var(--bg-panel) !important;
        border-right: 0.5px solid var(--border-soft) !important;
    }

    [data-testid="stSidebar"] > div {
        padding-top: 2rem !important;
    }

    .sidebar-title {
        font-family: 'Cinzel', serif !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: var(--text-1) !important;
        letter-spacing: 0.08em !important;
        margin-bottom: 2px !important;
    }

    .sidebar-sub {
        font-size: 0.78rem !important;
        color: var(--text-3) !important;
        font-weight: 300 !important;
        margin-top: 0 !important;
        font-family: 'Outfit', sans-serif !important;
    }

    .stButton > button {
        background: var(--accent) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        padding: 0.55rem 1.2rem !important;
        width: 100% !important;
        transition: all 0.15s ease !important;
        letter-spacing: 0.02em !important;
    }

    .stButton > button:hover {
        background: var(--accent-2) !important;
        transform: translateY(-1px) !important;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.76rem;
        font-weight: 500;
        font-family: 'Outfit', sans-serif;
    }

    .badge-ready {
        background: var(--success-bg);
        color: var(--success);
        border: 0.5px solid rgba(52,211,153,0.35);
    }

    .badge-waiting {
        background: rgba(149,128,212,0.07);
        color: var(--text-3);
        border: 0.5px solid var(--border-soft);
    }

    hr { border-color: var(--border-soft) !important; }

    [data-testid="stFileUploader"] {
        background: var(--bg-input) !important;
        border: 0.5px dashed var(--border) !important;
        border-radius: 10px !important;
    }

    [data-testid="stChatInput"] textarea {
        background: var(--bg-card) !important;
        border: 0.5px solid var(--border) !important;
        border-radius: 12px !important;
        font-family: 'Outfit', sans-serif !important;
        color: var(--text-1) !important;
        font-size: 0.95rem !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--text-3) !important;
    }

    .stSuccess {
        background: var(--success-bg) !important;
        border: 0.5px solid rgba(52,211,153,0.3) !important;
        color: var(--success) !important;
        border-radius: 8px !important;
    }

    .stWarning {
        background: var(--warning-bg) !important;
        border: 0.5px solid rgba(251,191,36,0.3) !important;
        color: var(--warning) !important;
        border-radius: 8px !important;
    }

    .stSpinner > div { border-top-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None


def release_vectorstore():
    """Properly release Chroma client before deletion (Windows fix)."""
    if st.session_state.vectorstore is not None:
        try:
            # Delete all documents from the collection to release handles
            st.session_state.vectorstore.delete_collection()
        except Exception:
            pass
        try:
            st.session_state.vectorstore._client._system.stop()
        except Exception:
            pass
        st.session_state.vectorstore = None
        st.session_state.chain = None
        gc.collect()
        time.sleep(1)


def safe_rmtree(path, retries=5):
    """Retry folder deletion to handle Windows file locks."""
    for i in range(retries):
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
            return True
        except PermissionError:
            time.sleep(0.8)
    # Last resort: try to delete file by file
    if os.path.exists(path):
        for root, dirs, files in os.walk(path, topdown=False):
            for f in files:
                try:
                    os.remove(os.path.join(root, f))
                except Exception:
                    pass
            for d in dirs:
                try:
                    os.rmdir(os.path.join(root, d))
                except Exception:
                    pass
        try:
            os.rmdir(path)
        except Exception:
            pass
    return True


def build_chain(vectorstore):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on the provided documents.
Use the following context to answer the question. If you don't know the answer from the context, say so.

Context:
{context}"""),
        ("human", "{input}"),
    ])

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["input"])),
            source_docs=lambda x: retriever.invoke(x["input"])
        )
        | RunnablePassthrough.assign(
            answer=prompt | llm | StrOutputParser()
        )
    )
    return chain


# ── Sidebar ──
with st.sidebar:
    st.markdown("<p class='sidebar-title'>AskMyDocs</p>", unsafe_allow_html=True)
    st.markdown("<p class='sidebar-sub'>Upload documents, ask anything.</p>", unsafe_allow_html=True)
    st.divider()

    uploaded_files = st.file_uploader(
        "Upload PDFs or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Index Documents"):
            save_dir = "./docs"

            # Release lock BEFORE deleting
            release_vectorstore()
            safe_rmtree(save_dir)
            os.makedirs(save_dir, exist_ok=True)
            safe_rmtree("./chroma_db")

            with st.spinner("Indexing your documents..."):
                for f in uploaded_files:
                    with open(os.path.join(save_dir, f.name), "wb") as out:
                        out.write(f.read())

                vs = ingest_docs(save_dir)
                if vs:
                    st.session_state.vectorstore = vs
                    st.session_state.chain = build_chain(vs)
                    st.session_state.messages = []
                    st.success(f"✓ {len(uploaded_files)} file(s) indexed!")

    st.divider()

    if st.session_state.vectorstore:
        st.markdown("<span class='status-badge badge-ready'>● Ready to chat</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='status-badge badge-waiting'>○ No documents loaded</span>", unsafe_allow_html=True)

    st.divider()

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if st.button("Clear Documents"):
        release_vectorstore()
        safe_rmtree("./chroma_db")
        safe_rmtree("./docs")
        st.session_state.messages = []
        st.success("Documents cleared!")
        st.rerun()


# ── Main area ──
st.markdown("<h1 class='main-title'>AskMyDocs</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Ask questions about your uploaded documents</p>", unsafe_allow_html=True)

# Chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        answer = msg["content"]
        sources = msg.get("sources", [])
        source_tags = "".join([f"<span class='source-tag'>📄 {s}</span>" for s in sources])
        st.markdown(
            f"<div class='bot-bubble'><div class='bot-label'>AskMyDocs</div>{answer}"
            + (f"<br><br>{source_tags}" if source_tags else "")
            + "</div>",
            unsafe_allow_html=True
        )

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.chain:
        st.warning("Please upload and index documents first using the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f"<div class='user-bubble'>{prompt}</div>", unsafe_allow_html=True)

        with st.spinner("Thinking..."):
            result = st.session_state.chain.invoke({"input": prompt})
            answer = result["answer"]
            source_docs = result.get("source_docs", [])

            seen = set()
            sources = []
            for doc in source_docs:
                src = os.path.basename(doc.metadata.get("source", "unknown"))
                page = doc.metadata.get("page", "")
                label = src + (f" p.{page+1}" if page != "" else "")
                if label not in seen:
                    sources.append(label)
                    seen.add(label)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
        st.rerun()
 