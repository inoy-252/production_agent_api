"""
Trimmed Context Agent Pipeline
Integration of sliding window context pruning into the LangGraph execution loop.
"""

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import MessagesState

from context_trimmer import sliding_window_messages


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
    raw_message = state.get("messages", [])
    windowed = sliding_window_messages(
        raw_message,
        max_history_turns=max_history_turns,
    )
    return [system_instruction] + [m for m in windowed if m.type != "system"]


if __name__ == "__main__":
    from langchain_core.messages import AIMessage, HumanMessage

    print("=" * 65)
    print("TESTING PREPARE_TRIMMED_CONTEXT")
    print("=" * 65)

    mock_state: MessagesState = {
        "messages": [
            HumanMessage(content="Hello!"),
            AIMessage(content="Hi! How can I assist your travels?"),
            HumanMessage(content="What is the weather in Tokyo?"),
            AIMessage(content="Tokyo is 15°C and clear."),
            HumanMessage(content="Book my ticket to Tokyo."),
        ]
    }
    test_system_prompt = SystemMessage(content="You are a professional travel agent.")

    final_payload = prepare_trimmed_context(
        state=mock_state,
        system_instruction=test_system_prompt,
        max_history_turns=2,
    )
    print(f"Total Messages in State  : {len(mock_state['messages'])}")
    print(f"Messages Sent to Gemini  : {len(final_payload)}\n")
    print("--- Final Assembled Payload ---")
    for msg in final_payload:
        prefix = f"[{msg.type.upper()}]"
        print(f"{prefix:<10}: {msg.content}")
    print("=" * 65)
