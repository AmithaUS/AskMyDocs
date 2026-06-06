# AskMyDocs 

A **RAG (Retrieval-Augmented Generation)** based document Q&A application that lets you upload PDFs or TXT files and chat with them using natural language — powered by LLaMA 3.3 via Groq.

---

##  Demo

> Upload a document → Index it → Ask questions → Get accurate answers with source references.

---

##  Tech Stack

| Component | Technology |
|---|---|
| LLM | LLaMA 3.3 70B (via Groq) |
| Orchestration | LangChain 1.3 |
| Vector Store | ChromaDB |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| UI | Streamlit |
| Language | Python 3.11 |

---

##  How It Works

```
Upload PDF/TXT
      ↓
Split into chunks (RecursiveCharacterTextSplitter)
      ↓
Generate embeddings (HuggingFace)
      ↓
Store in ChromaDB
      ↓
User asks a question
      ↓
Retrieve top-k relevant chunks
      ↓
Send context + question to LLaMA 3.3 via Groq
      ↓
Return answer with source references
```

---

##  Project Structure

```
rag-doc-chat/
├── app.py              # Streamlit UI
├── chat.py             # Terminal chat loop
├── ingest.py           # Document loading & embedding
├── main.py             # Entry point for terminal mode
├── requirements.txt    # Dependencies
├── .env.example        # Environment variable template
└── .gitignore
```

---

##  Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/rag-doc-chat.git
cd rag-doc-chat
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
```
Open `.env` and add your Groq API key:
```
GROQ_API_KEY=your_actual_groq_api_key
```
Get your free API key at [console.groq.com](https://console.groq.com)

### 5. Run the app
```bash
streamlit run app.py
```

---

## 💬 Usage

1. Open the app in your browser (`http://localhost:8501`)
2. Upload one or more PDF or TXT files using the sidebar
3. Click **Index Documents** and wait for processing
4. Type your question in the chat input
5. Get answers with source file references

---

##  Terminal Mode

You can also run the app in terminal mode (no UI):
```bash
python main.py
```
Place your documents in the `docs/` folder first.

---

##  Key Dependencies

```
langchain
langchain-core
langchain-community
langchain-groq
chromadb==0.5.23
sentence-transformers
streamlit
python-dotenv
pypdf
```

---

##  Author

**Amitha U S**  
BSc Statistics | Aspiring Data Scientist  
[LinkedIn]([www.linkedin.com/in/amitha-u-s-36165b361]) • [GitHub](https://github.com/AmithaUS)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
