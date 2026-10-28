import os
import sqlite3

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


@st.cache_resource
def get_agent():
    """Compile the LangGraph agent ONCE and cache it with HITL interuption enabled."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["sensitive_tools"],
    )


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
    f"Architecture: LangGraph ReAct + SQLite Persistence | Active Session: `{thread_id}`"
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

        def agent_stream_generator():
            agent_app = get_agent()
            config = {"configurable": {"thread_id": thread_id}}
            for chunk, meta in agent_app.stream(
                {"messages": [HumanMessage(content=user_prompt)]},
                config=config,  # type: ignore
                stream_mode="messages",
            ):
                if isinstance(meta, dict) and meta.get("langgraph_node") == "chatbot":
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

        full_response = st.write_stream(agent_stream_generator)
    if isinstance(full_response, str) and full_response.strip():
        st.session_state["messages"].append(
            {"role": "assistant", "content": full_response}
        )
    st.rerun()

agent_app = get_agent()
config = {"configurable": {"thread_id": thread_id}}
current_state = agent_app.get_state(config)  # type: ignore

if current_state.next and "sensitive_tools" in current_state.next:
    last_msg = current_state.values.get("messages", [])[-1]
    pending_calls = getattr(last_msg, "tool_calls", [])

    for call in pending_calls:
        if call.get("name") == "book_trip_flight":
            dest = call.get("args", {}).get("destination", "unknown")
            cost = call.get("args", {}).get("total_cost", 0.0)

            with st.container():
                st.warning("**Manager Approval Required: High-Risk Financial Action**")
                st.markdown(
                    f"The agent has queued a non-reversible booking transaction:\n\n"
                    f"- **Destination:** {dest}\n"
                    f"- **Committed Funds:** {cost:,.2f} USD\n\n"
                    f"Execution is currently halted in SQLite. Click below to authorize funds."
                )
                if st.button("Approve & Commit Corporate Funds", type="primary"):
                    with st.spinner("Authorizing funds and completing booking..."):
                        res = agent_app.invoke(None, config=config)  # type: ignore
                        final_msg = res["messages"][-1]
                        final_text = getattr(final_msg, "content", "")
                        if (
                            isinstance(final_text, list)
                            and len(final_text) > 0
                            and isinstance(final_text[0], dict)
                        ):
                            final_text = final_text[0].get("text", "")
                        st.session_state["messages"].append(
                            {
                                "role": "assistant",
                                "content": str(final_text).replace("$", "USD "),
                            }
                        )
                        st.rerun()
