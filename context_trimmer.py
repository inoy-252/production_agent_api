"""
Context Trimmer & Sliding Window History Manager
Project 4 - Chunk 10: Slashing Compounding History Costs

Prevents runaway token growth in stateful multi-turn agents by enforcing
a sliding window over conversation history while preserving critical system prompts.
"""

from typing import Sequence
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages,
)


def get_sliding_window_messages(
    messages: Sequence[BaseMessage],
    max_history_turns: int = 4,
) -> list[BaseMessage]:
    """
    Trims chat history to a fixed number of recent message turns.
    
    Guarantees:
    1. System instructions are ALWAYS preserved at the root (index 0).
    2. Cuts older conversational turns to prevent compounding prompt token bills.
    3. Always starts on a HumanMessage so the LLM context remains coherent.
    """
    # Using message count heuristic for deterministic sliding window
    trimmed = trim_messages(
        messages,
        max_tokens=max_history_turns,
        token_counter=len,  # Counts discrete message blocks
        strategy="last",
        start_on="human",
        include_system=True,
        allow_partial=False,
    )
    return list(trimmed)


def calculate_token_reduction_ratio(
    raw_message_count: int,
    trimmed_message_count: int,
) -> float:
    """Calculates the percentage reduction in conversation payload."""
    if raw_message_count <= 0:
        return 0.0
    reduction = (raw_message_count - trimmed_message_count) / raw_message_count
    return round(reduction * 100, 2)


if __name__ == "__main__":
    print("=" * 65)
    print("✂️  CONTEXT TRIMMER BENCHMARK & DEMONSTRATION")
    print("=" * 65)

    # Simulate an 8-turn conversation that suffered from compounding history
    simulated_history: list[BaseMessage] = [
        SystemMessage(content="You are a production travel agent assistant."),
        HumanMessage(content="Hi! I am planning a vacation."),
        AIMessage(content="Hello! Where would you like to travel?"),
        HumanMessage(content="I want to visit Tokyo next month."),
        AIMessage(content="Tokyo is fantastic. What is your estimated budget?"),
        HumanMessage(content="My budget is $3,000 USD."),
        AIMessage(content="Got it. $3,000 USD for Tokyo."),
        HumanMessage(content="Can you check the current weather in Tokyo?"),
        AIMessage(content="The current weather in Tokyo is 15°C and clear."),
        HumanMessage(content="Great, now book my flight to Tokyo."),
    ]

    print(f"📊 Original History Length : {len(simulated_history)} messages")

    # Apply sliding window of last 4 conversational turns
    pruned_history = get_sliding_window_messages(simulated_history, max_history_turns=4)

    savings_pct = calculate_token_reduction_ratio(len(simulated_history), len(pruned_history))
    print(f"✂️  Trimmed History Length  : {len(pruned_history)} messages")
    print(f"💰 Context Payload Reduction: {savings_pct}%\n")

    print("--- Active Context Passed to LLM ---")
    for msg in pruned_history:
        prefix = f"[{msg.type.upper()}]"
        print(f"{prefix:<10}: {msg.content}")
    print("=" * 65)
