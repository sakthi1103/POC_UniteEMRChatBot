import streamlit as st
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --------------------------------
# Load environment variables
# --------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")

# --------------------------------
# Page config
# --------------------------------
st.set_page_config(page_title="UniteEMR Assist", layout="wide")
st.title("UniteEMR Assist")

# st.sidebar.title("Navigation")
# st.sidebar.page_link("app.py", label="💬 Chat")
# st.sidebar.page_link("pages/1_Chat_History.py", label="📜 Chat History")

# --------------------------------
# MongoDB connection (cached)
# --------------------------------
@st.cache_resource
def get_mongo_collection():
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    return db[MONGO_COLLECTION]

mongo_col = get_mongo_collection()

# --------------------------------
# Session state init
# --------------------------------
if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

# --------------------------------
# Mongo save helper
# --------------------------------
def save_message(role: str, content: str):
    mongo_col.update_one(
        {"session_id": st.session_state.chat_session_id},
        {
            "$setOnInsert": {
                "session_id": st.session_state.chat_session_id,
                "created_at": datetime.utcnow()
            },
            "$push": {
                "messages": {
                    "role": role,
                    "content": content,
                    "timestamp": datetime.utcnow()
                }
            }
        },
        upsert=True
    )

# --------------------------------
# Load retriever (cached)
# --------------------------------
@st.cache_resource
def get_retriever():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})

retriever = get_retriever()

# --------------------------------
# Build conversational RAG chain
# --------------------------------
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=st.session_state.memory,
    return_source_documents=True,
    output_key="answer"
)

# --------------------------------
# Render chat history
# --------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------
# User input
# --------------------------------
prompt = st.chat_input("Ask a question...")

if prompt:
    # ---- User message ----
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    save_message("user", prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    # ---- Assistant response ----
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa_chain({"question": prompt})
            answer = result["answer"]

        st.markdown(answer)

        with st.expander("📚 Sources"):
            for doc in result["source_documents"]:
                st.write(doc.metadata.get("source"))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    assistant_payload = {
        "answer": answer,
        "sources": [
            doc.metadata.get("source")
            for doc in result.get("source_documents", [])
        ]
    }

    save_message("assistant", assistant_payload)

# --------------------------------
# Clear conversation
# --------------------------------
if st.button("🧹 Clear conversation"):
    mongo_col.delete_one(
        {"session_id": st.session_state.chat_session_id}
    )

    st.session_state.messages = []
    st.session_state.memory.clear()
    st.session_state.chat_session_id = str(uuid.uuid4())
    st.rerun()
