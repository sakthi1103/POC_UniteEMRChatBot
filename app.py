import streamlit as st
import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA

# --------------------------------
# Load env
# --------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# --------------------------------
# UI (must be first Streamlit calls)
# --------------------------------
st.set_page_config(page_title="📘 UniteEMR Assist", layout="wide")
st.title("📘 UniteEMR Assist")
# --------------------------------
# Cached backend objects (NO st.* inside)
# --------------------------------
@st.cache_resource
def load_qa_chain():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

# --------------------------------
# Chat UI
# --------------------------------
question = st.text_input("Ask any question about UniteEMR:")

if question:
    with st.spinner("🔍 Thinking..."):
        qa_chain = load_qa_chain()
        result = qa_chain({"query": question})

    st.subheader("Answer")
    st.write(result["result"])

    with st.expander("📚 Source URLs"):
        for doc in result["source_documents"]:
            st.write(doc.metadata.get("source"))


