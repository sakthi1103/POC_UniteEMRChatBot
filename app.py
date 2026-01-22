import streamlit as st
import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --------------------------------
# Load secrets
# --------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# --------------------------------
# Page config
# --------------------------------
st.set_page_config(page_title="UniteEMR Assist", layout="wide")
st.title("UniteEMR Assist")

# --------------------------------
# Session state init
# --------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"   # 👈 REQUIRED
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
    output_key="answer"  # 👈 important
)

# --------------------------------
# Render chat history
# --------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------
# User input (chat-style)
# --------------------------------
prompt = st.chat_input("Ask a question...")

if prompt:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa_chain({"question": prompt})
            answer = result["answer"]

        st.markdown(answer)

        # Optional: sources
        with st.expander("📚 Sources"):
            for doc in result["source_documents"]:
                st.write(doc.metadata.get("source"))

    # Save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

# --------------------------------
# Clear conversation
# --------------------------------
if st.button("🧹 Clear conversation"):
    st.session_state.messages = []
    st.session_state.memory.clear()
    st.rerun()
