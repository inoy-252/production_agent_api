"""
Token Economics & LLM Cost Auditor
Project 6 - Chunk 1: Pricing Catalog and Cost Calculation Engine
"""

from dataclasses import dataclass

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from agent_sqlite_persistence import builder


@dataclass
class ModelPricing:
    """Pricing rates in USD per 1 Million tokens."""

    input_per_mill: float
    output_per_mill: float


PRICING_CATALOG: dict[str, ModelPricing] = {
    "gemini-3.5-flash-lite": ModelPricing(
        input_per_mill=0.075,
        output_per_mill=0.30,
    ),
    "gemini-3.5-pro": ModelPricing(
        input_per_mill=2.50,
        output_per_mill=10.0,
    ),
}


def calculate_turn_cost(
    model_name: str, input_tokens: int, output_tokens: int
) -> dict[str, float]:
    """Calculates exact dollar cost for a single conversational turn."""
    pricing = PRICING_CATALOG.get(model_name, PRICING_CATALOG["gemini-3.5-flash-lite"])

    input_cost = (input_tokens / 1_000_000) * pricing.input_per_mill
    output_cost = (output_tokens / 1_000_000) * pricing.output_per_mill
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def extract_token_usage(message) -> tuple[int, int, int]:
    """Safely extracts (input_tokens, output_tokens, total_tokens) from and LLM response message."""
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = input_tokens + output_tokens
        return input_tokens, output_tokens, total_tokens
    return 0, 0, 0


def run_cost_audit() -> None:
    """Executes a multi-turn conversation and audits the compounding token costs."""
    print("=" * 75)
    print(" RUNNING MULTI-TURN AGENT TOKEN & COST AUDIT")
    print("=" * 75)

    agent = builder.compile(checkpointer=MemorySaver())
    session_thread = "audit-session-yasir"
    model_name = "gemini-3.5-flash-lite"

    conversation_turns = [
        "Hi, I'm Yasir. I'm planning a 5-day business trip to Tokyo.",
        "What is the current weather in Tokyo?",
        "What is our company meal allowance per diem policy for Tokyo?",
    ]

    session_input_tokens = 0
    session_output_tokens = 0
    session_total_cost = 0

    for turn_idx, user_query in enumerate(conversation_turns, start=1):
        print(f'\n--- Turn {turn_idx}: "{user_query}" ---')

        config = {"configurable": {"thread_id": session_thread}}
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_query)]},
            config=config,  # type: ignore
        )
        last_message = result["messages"][-1]
        in_tok, out_tok, tot_tok = extract_token_usage(last_message)

        session_input_tokens += in_tok
        session_output_tokens += out_tok

        costs = calculate_turn_cost(model_name, in_tok, out_tok)
        turn_cost = costs["total_cost"]
        session_total_cost += turn_cost

        print(f"  Tokens Used : Input: {in_tok} | Output: {out_tok} | Total: {tot_tok}")
        print(f"  Turn Cost   : ${turn_cost:.6f} USD")
        print(f"  Session Sum : ${session_total_cost:.6f} USD")

    pro_costs = calculate_turn_cost(
        "gemini-3.5-pro", session_input_tokens, session_output_tokens
    )
    pro_total = pro_costs["total_cost"]
    savings = pro_total - session_total_cost
    savings_pct = (savings / pro_total) * 100 if pro_total > 0 else 0.0

    print("\n" + "=" * 75)
    print("📊 EXECUTIVE TOKEN ECONOMICS & COST REPORT")
    print("=" * 75)
    print(f"Total Session Turns       : {len(conversation_turns)}")
    print(f"Total Session Input Tokens: {session_input_tokens}")
    print(f"Total Session Output Tokens: {session_output_tokens}")
    print(f"Total Session Tokens      : {session_input_tokens + session_output_tokens}")
    print("-" * 75)
    print(f"Cost on gemini-3.5-flash-lite: ${session_total_cost:.6f} USD")
    print(f"Cost on gemini-3.5-pro       : ${pro_total:.6f} USD")
    print(
        f"Cost Savings (Our Agent)     : ${savings:.6f} USD ({savings_pct:.1f}% Cheaper!)"
    )
    print("=" * 75)


if __name__ == "__main__":
    run_cost_audit()
