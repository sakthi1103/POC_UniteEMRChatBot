import streamlit as st
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime

# --------------------------------
# Load env
# --------------------------------
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")

# --------------------------------
# Page config
# --------------------------------
# st.set_page_config(page_title="Chat History", layout="wide")
# st.title("📜 Chat History")

# --------------------------------
# Mongo connection (cached)
# --------------------------------
@st.cache_resource
def get_mongo_collection():
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    return db[MONGO_COLLECTION]

mongo_col = get_mongo_collection()

# --------------------------------
# Sidebar navigation
# --------------------------------
# st.sidebar.title("Navigation")
# st.sidebar.page_link("app.py", label="💬 Chat")
# st.sidebar.page_link("pages/1_Chat_History.py", label="📜 Chat History")

# --------------------------------
# Fetch all chat sessions
# --------------------------------
sessions = list(
    mongo_col.find(
        {},
        {
            "_id": 0,
            "session_id": 1,
            "created_at": 1,
            "messages": 1
        }
    ).sort("created_at", -1)
)

if not sessions:
    st.info("No chat history found.")
    st.stop()

# --------------------------------
# Session selector
# --------------------------------
session_map = {
    f"{s['created_at'].strftime('%d-%m-%Y %H:%M:%S')} | {s['session_id'][:8]}": s
    for s in sessions
}

selected_label = st.selectbox(
    "Select a chat session",
    session_map.keys()
)

selected_session = session_map[selected_label]

st.divider()
st.subheader("Conversation")

# --------------------------------
# Render messages
# --------------------------------
# for msg in selected_session["messages"]:
#     role = msg["role"]
#     content = msg["content"]

#     with st.chat_message(role):
#         st.markdown(content)


for msg in selected_session["messages"]:
    role = msg["role"]
    content = msg["content"]

    # ✅ NORMALIZE CONTENT (PASTE HERE)
    if isinstance(content, dict):
        content = content.get("answer", "")

    # ---- USER MESSAGE ----
    if role == "user":
        st.markdown(
            f"""
            <div style="
                background-color:#1f2937;
                padding:14px 16px;
                border-radius:12px;
                margin-bottom:10px;
                display:flex;
                align-items:center;
                gap:12px;
            ">
                <div style="
                    background-color:#facc15;
                    width:36px;
                    height:36px;
                    border-radius:10px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    font-weight:bold;
                ">👤</div>
                <div style="font-size:16px; font-weight:600;">
                    {content}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ---- ASSISTANT MESSAGE ----
    else:
        safe_content = content.replace("\n", "<br>")

        st.markdown(
            f"""
            <div style="
                background-color:#020617;
                padding:18px 20px;
                border-radius:14px;
                margin-bottom:24px;
                border-left:4px solid #facc15;
            ">
                <div style="
                    font-weight:600;
                    margin-bottom:10px;
                    display:flex;
                    align-items:center;
                    gap:8px;
                ">
                    🤖 Assistant
                </div>
                <div style="line-height:1.7;">
                    {safe_content}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

