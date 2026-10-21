# Project 4 — Chunk 3: Real-Time Token Streaming with Server-Sent Events (SSE)

## 1. What are Tokens?
* LLMs do not read or write entire words or paragraphs at once.
* Sentences are split into **tokens** (roughly 3-4 characters or 0.75 of an English word).
* The LLM generates one token at a time in a continuous probabilistic loop.

---

## 2. Why Streaming Matters: TTFT (Time To First Token)
* In standard HTTP requests (`POST /chat`), the client must wait for all 150+ tokens to finish generating before receiving any response (3–5 seconds wait time).
* In streaming (`POST /chat/stream`), the client receives Token 1 immediately (~200ms TTFT), producing a responsive, live typing user experience.

---

## 3. Server-Sent Events (SSE) Protocol
* **MIME Type:** `Content-Type: text/event-stream`.
* **Framing:** Every event starts with `data: ` and terminates with double newlines `\n\n`.
* **Payload Structure:**
  * Initial Handshake: `data: {"thread_id": "..."}\n\n`
  * Token Stream: `data: {"token": "..."}\n\n`
  * Completion Signal: `data: [DONE]\n\n`
* **Advantage over WebSockets:** One-way, lightweight, runs over standard HTTP, and is natively supported by modern browsers via `EventSource`.

---

## 4. LangGraph Streaming Architecture
* **`stream_mode="messages"`:** Instructs LangGraph to yield individual message chunks (`AIMessageChunk`) rather than complete node states.
* **`StreamingResponse`:** FastAPI's streaming wrapper that keeps the HTTP connection open while consuming from an inner Python generator (`yield`).
* **Concurrent Persistence:** Checkpointing to SQLite remains fully functional during streaming.
