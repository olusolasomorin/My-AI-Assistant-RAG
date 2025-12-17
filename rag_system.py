import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

print("Api Key found")

DOCS_DIR = "./documents"
DB_DIR = "./chroma_db"

def rag_system() :
    # Document Loading
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"Created {DOCS_DIR}. Please put your PDFs/TXT files there and run again.")
        return None
    
    print(f"📂 Loading documents from {DOCS_DIR}...")
    pdf_loader = DirectoryLoader(DOCS_DIR, glob="./*.pdf", loader_cls=PyPDFLoader)

    docs = []
    docs.extend(pdf_loader.load())

    if not docs:
        print("❌ No documents found. Add files to the 'documents' folder.")
        return None
    
    print(f"✅ Loaded {len(docs)} documents.")

    # Document Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 400,
        chunk_overlap = 80,
    )

    chunks = text_splitter.split_documents(docs)
    print(f"✂  Split into {len(chunks)} chunks.")
    # for i, chunk in enumerate(chunks[:3]):
    #     print(f"\nChunk {i+1}: {chunk.page_content}")

    # Embedding
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_type=api_key
    )

    # Vector Store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # RAG Chain with memory
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
    )

    # History retriever
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # system prompt
    qa_system_prompt = (
        "You are an AI assistant for Olusola. Your sole purpose is to "
        "answer questions about Olusola based on the provided documents."
        "\n\n"
        "Guidelines:"
        "1. Refer to the owner as 'Olusola'."
        "2. If the user asks a question unrelated to Olusola, politely decline and state that you are strictly trained to answer questions about Olusola."
        "3. If the answer is not in the provided context, say you don't have that information."
        "4. Keep answers concise (max 3 sentences)."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

def start_chat():
    chain = rag_system()
    if not chain:
        return

    chat_history = []
    print("\n🤖 OpenAI Assistant Ready! (Type 'exit' to quit)\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break

        # Invoke the chain
        response = chain.invoke({"input": query, "chat_history": chat_history})
        
        print(f"AI: {response['answer']}")
        
        # Show Sources
        # print("\n   Sources:")
        # seen_sources = set()
        # for doc in response["context"]:
        #     source = doc.metadata.get("source", "unknown")
        #     if source not in seen_sources:
        #         print(f"   - {os.path.basename(source)}")
        #         seen_sources.add(source)
        # print("-" * 30)

        # Update History
        chat_history.extend([
            HumanMessage(content=query),
            AIMessage(content=response["answer"])
        ])

if __name__ == "__main__":
    start_chat()
