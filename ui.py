import json
import os

import requests
import streamlit as st

try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
        os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    elif "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        os.environ["GEMINI_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from agent_sqlite_persistence import DB_PATH, builder

st.set_page_config(page_title="Production AI Agent", page_icon="🤖", layout="wide")

st.sidebar.title("Session Control")
st.sidebar.markdown("Test multi-tenant isolation and persistent memory.")

thread_id = st.sidebar.text_input(
    "Active Session ID (thread_id)",
    value="user-yasir-101",
    help="Change this ID to test memory isolation between different users.",
)

if st.sidebar.button("Clear UI Chat Screen"):
    st.session_state["messages"] = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Backend Architecture:**\n"
    "**LangGraph ReAct Loop**\n"
    "**SQLite Persistence (`SqliteSaver`)**\n"
    "**FastAPI Server-Sent Events (SSE)**\n"
    "**Internal RAG Knowledge Base**"
)

st.title("Production AI Travel & Enterprise Agent")
st.caption(
    f"Connected to FastAPI Backend at `http://127.0.0.1:8000` | Active Session: `{thread_id}`"
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"].replace("$", "USD "))

if user_prompt := st.chat_input(
    "Ask about your trip, weather, budget, or company policies..."
):
    st.session_state["messages"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):

        def sse_stream_generator():
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/chat/stream",
                    json={"prompt": user_prompt, "thread_id": thread_id},
                    stream=True,
                    timeout=(0.2, 60),
                )

                if response.status_code != 200:
                    yield f"Backend Error: HTTP {response.status_code}"
                    return
                for raw_line in response.iter_lines(decode_unicode=True):
                    line = str(raw_line) if raw_line else ""
                    if line and line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            payload = json.loads(data_str)
                            if "token" in payload:
                                token = payload["token"].replace("$", "USD ")
                                yield token
                        except json.JSONDecodeError:
                            pass
            except requests.exceptions.ConnectionError:
                with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
                    agent_app = builder.compile(checkpointer=checkpointer)
                    config = {"configurable": {"thread_id": thread_id}}
                    for chunk, meta in agent_app.stream(
                        {"messages": [HumanMessage(content=user_prompt)]},
                        config=config,  # type: ignore
                        stream_mode="messages",
                    ):
                        if (
                            isinstance(meta, dict)
                            and meta.get("langgraph_node") == "chatbot"
                        ):
                            raw_content = getattr(chunk, "content", None)
                            if raw_content:
                                text = raw_content
                                if (
                                    isinstance(text, list)
                                    and len(text) > 0
                                    and isinstance(text[0], dict)
                                    and "text" in text[0]
                                ):
                                    text = text[0]["text"]
                                if isinstance(text, str) and text:
                                    yield text.replace("$", "USD ")

        full_response = st.write_stream(sse_stream_generator)

    st.session_state["messages"].append({"role": "assistant", "content": full_response})
