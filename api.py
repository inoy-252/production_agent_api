import json
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

from agent_sqlite_persistence import DB_PATH, builder


class ChatRequest(BaseModel):
    """The incoming message payload from the client."""

    prompt: str = Field(
        description="The message or question for the AI agent.",
        examples=["Hi!, I am planning a 5 day trip to Tokyo with a $1500 budget. "],
    )
    thread_id: str | None = Field(
        default=None,
        description="Optional session ID. If omitted, the server generates a fresh one .",
        examples=["user-yasir-101"],
    )


class ChatResponse(BaseModel):
    """The outgoing structures response sent back to the client."""

    response: str = Field(description="The final synthesized answer from the agent.")
    thread_id: str = Field(description="The active conversation session ID.")


app = FastAPI(
    title="Production Agent API",
    description="Production-grade AI Agent API with SQLite persistence and multi-turn conversational memory.",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    """Liveness probe for monitoring tools and cloud orchestrators."""
    return {"status": "healthy", "service": "production-agent-api"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """Executes a conversation turn with persistent Langgraph agent."""
    session_id = request.thread_id or f"session-{uuid.uuid4().hex[:8]}"

    try:
        with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
            agent_app = builder.compile(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": session_id}}

            events = agent_app.stream(
                {"messages": [HumanMessage(content=request.prompt)]},
                config,
                stream_mode="values",
            )
            last_message = None
            for event in events:
                if "messages" in event:
                    last_message = event["messages"][-1]
            content = last_message.content if last_message else "No response generated."
            if (
                isinstance(content, list)
                and len(content) > 0
                and isinstance(content[0], dict)
                and "text" in content[0]
            ):
                content = content[0]["text"]

            return ChatResponse(response=str(content), thread_id=session_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent Error: {e!s}")


@app.post("/chat/stream")
def chat_stream_endpoint(request: ChatRequest):
    """Streams agent response token-by-token using Server-Sent-Events (SSE)."""
    session_id = request.thread_id or f"session-{uuid.uuid4().hex[:8]}"

    def event_generator():
        yield f"data: {json.dumps({'thread_id': session_id})}\n\n"
        try:
            with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
                agent_app = builder.compile(checkpointer=checkpointer)
                config = {"configurable": {"thread_id": session_id}}

                for chunk, meta in agent_app.stream(
                    {"messages": [HumanMessage(content=request.prompt)]},
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
                                yield f"data: {json.dumps({'token': text})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
