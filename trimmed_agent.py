"""
Trimmed Context Agent Pipeline
Integration of sliding window context pruning into the LangGraph execution loop.
"""

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import MessagesState

from context_trimmer import get_sliding_window_messages


def prepare_trimmed_context(
    state: MessagesState,
    system_instruction: SystemMessage,
    max_history_turns: int = 4,
) -> list[BaseMessage]:
    """
    Constructs the pruned message payload for the language model.

    Extracts accumulated messages from graph state, applies the sliding window
    trimmer to cap token growth, and prepends the system instruction.
    """
    raw_messages = state.get("messages", [])
    windowed = get_sliding_window_messages(
        raw_messages,
        max_history_turns=max_history_turns,
    )
    return [system_instruction] + [m for m in windowed if m.type != "system"]
