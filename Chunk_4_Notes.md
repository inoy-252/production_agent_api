# Project 4 — Chunk 4: Autonomous RAG Knowledge Integration (RAG as a Tool)

## 1. The Core Paradigm Shift: Static RAG vs. Agentic RAG
* **Traditional / Static RAG:** Every user query is blindly passed through a vector search pipeline, even if the user only said "Hello" or asked for a simple math calculation.
* **Agentic RAG ("RAG as a Tool"):** The retrieval engine is exposed as a tool (`@tool search_knowledge_base`). The agent's LLM acts as the autonomous router, deciding *if*, *when*, and *what* to search based on conversational intent.

---

## 2. Knowledge Base Structure & Keyword Matching
* Internal enterprise documents (policies, safety runbooks, per diems) are structured into a searchable repository.
* **Defensive String Matching:**
  `if any(word in text.lower() or word in title.lower() for word in query_words):`
  * Employs C-speed substring searching (`word in text.lower()`) rather than allocating large word lists in RAM.
  * Captures keyword variants and folder-based topic titles without requiring exact string equality.

---

## 3. Multi-Tool Synthesis & Zero Hallucination
* The agent can execute multiple disparate tools within a single conversational turn:
  1. `search_knowledge_base`: Fetches internal company rules (e.g., flight luggage limits, emergency hotlines).
  2. `calculate_budget`: Performs exact mathematical calculations.
  3. `get_current_weather`: Checks external environmental conditions.
* Answers are strictly **grounded** in retrieved documents, completely preventing hallucinated numbers or policies.

---

## 4. End-to-End API Integration
* The LangGraph workflow automatically serves the new RAG tool over both:
  * `POST /chat`: Standard JSON REST endpoint.
  * `POST /chat/stream`: Live token-by-token Server-Sent Events (SSE) stream.
* All conversation turns, retrieved knowledge, and session memories are durably persisted in `agent_memory.db` under the active `thread_id`.
