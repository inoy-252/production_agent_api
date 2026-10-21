# Project 4 — Chunk 2: Production FastAPI Web Service & Pydantic Validation

## 1. The Core Problem: Terminal Scripts vs. Web APIs
* Terminal scripts cannot be used by web applications, mobile apps, or enterprise microservices.
* In production, an AI agent must be wrapped in an **HTTP REST API** that accepts JSON payloads, validates inputs defensively, maintains session state, and returns structured data.

---

## 2. Pydantic v2 Validation Schemas
Pydantic acts as the gatekeeper for incoming HTTP traffic, ensuring malformed requests never reach the LLM or trigger costly API calls:

* **`ChatRequest`:**
  * `prompt: str = Field(description="...", examples=[...])` (Mandatory user query).
  * `thread_id: Optional[str] = Field(default=None, description="...", examples=[...])` (Optional session identifier).
* **`ChatResponse`:**
  * `response: str` (The final synthesized answer).
  * `thread_id: str` (The active session key returned to the client).

---

## 3. Dynamic Session Lifecycle (`uuid.uuid4()`)
* **First Turn (New Session):**
  When a client sends `thread_id: null` or omits it, the server dynamically generates a unique session key using `uuid.uuid4().hex[:8]` (e.g., `session-8fd9d081`).
* **Subsequent Turns (Existing Session):**
  The client passes back the generated `thread_id`. The server loads the session checkpoint from `agent_memory.db`, achieving continuous multi-turn recall across HTTP calls.

---

## 4. Endpoints Built
* **`GET /health`:** Lightweight monitoring probe returning service health status (`200 OK`).
* **`POST /chat`:** Stateful conversation endpoint connecting the Pydantic request to the LangGraph ReAct agent with SQLite persistence.
* **Interactive Docs:** Automatically generated Swagger UI available at `/docs`.
