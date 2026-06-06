import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
 
load_dotenv()
 
def chat_loop(vectorstore):
    llm = ChatGroq(
        model="llama3-8b-8192",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )
 
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on the provided documents.
Use the following context to answer the question. If you don't know the answer from the context, say so.
 
Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
 
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    chat_history = []
 
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
 
    print("💬 Chat with your documents. Type 'exit' to quit.\n")
 
    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
 
        # Retrieve relevant docs
        source_docs = retriever.invoke(question)
        context = format_docs(source_docs)
 
        # Build and run chain
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "input": question,
            "context": context,
            "chat_history": chat_history,
        })
 
        # Update memory manually
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))
 
        print(f"\n🤖 AI: {answer}\n")
 
        if source_docs:
            print("📄 Sources:")
            seen = set()
            for doc in source_docs:
                src = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "")
                label = f"  - {src}" + (f", page {page+1}" if page != "" else "")
                if label not in seen:
                    print(label)
                    seen.add(label)
        print()
 