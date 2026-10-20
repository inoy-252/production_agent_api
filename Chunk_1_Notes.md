# Project 4 — Chunk 1: Persistent Agent Memory with SqliteSaver

## 1. The Core Problem: RAM vs. Durable Persistence
* In Project 3, we used `MemorySaver`, which stores state snapshots in temporary RAM.
* If a production web server reboots, crashes, or is deployed with a new release, **all user conversations are wiped instantly**.
* In Project 4, we replace `MemorySaver` with **`SqliteSaver`**, committing every turn, tool call, and state transition directly to an SQLite database file (`agent_memory.db`) on disk.

---

## 2. Under the Hood: Automatic Table Creation
When you connect to SQLite with `SqliteSaver.from_conn_string("agent_memory.db")`:
1. If `agent_memory.db` does not exist on disk, SQLite automatically creates the empty file.
2. `SqliteSaver` executes built-in schema creation commands:
   * **`checkpoints` table:** Stores frozen snapshots of the graph state (messages, thread ID, checkpoint ID, timestamp).
   * **`writes` table:** Stores pending or intermediate state changes (channel updates, tool calls) before they are finalized.
3. If the file already exists, it reads the previous history and appends new steps without data loss.

---

## 3. Session & Tenant Isolation (`thread_id`)
* Checkpoints are partitioned by `thread_id`:
  `config = {"configurable": {"thread_id": "user-yasir-101"}}`
* Each user's conversation is strictly isolated. When User A asks about their trip details, User B's thread cannot read or leak User A's data.

---

## 4. System Prompt Guardrails (Controlling Agent Proactivity)
* Frontier models are trained to be helpful assistants. Without a system prompt, when an agent detects entities matching a tool (e.g. city or budget numbers), it may autonomously call those tools without the user asking.
* By injecting a **`SystemMessage`**:
  ```python
  system_instruction = SystemMessage(
      content="You are a professional travel assistant. Only execute tools when the user explicitly asks for weather or budget calculations."
  )
  ```
  We establish strict behavioral guardrails, ensuring tools only execute upon explicit user command.
