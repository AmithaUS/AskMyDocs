from ingest import ingest_docs
from chat import chat_loop
import os
 
os.makedirs("docs", exist_ok=True)
 
vectorstore = ingest_docs("./docs")
 
if vectorstore:
    chat_loop(vectorstore)
