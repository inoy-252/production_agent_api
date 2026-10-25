"""
Human-in-the-Loop (HITL) Financial Governance Agent
Project 6 - Chunk 1: Safe vs Sensitive Tool Definitions
"""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()


@tool
def get_current_weather(city: str) -> str:
    """Gets real time destination weather."""
    return f"The current weather in {city} is 15°C and clear with cool evening winds."


@tool
def calculate_budget(total_budget: float, days: int) -> str:
    """Calculates daily spending budget."""
    if days <= 0:
        return "Error: Days must be greater than 0."
    daily = total_budget / days
    return f"{daily:.2f} USD per day (Total: {total_budget:.2f} USD over {days} days.)"


@tool
def book_trip_flight(destination: str, total_cost: float) -> str:
    """Books a corporate flight and commits company financial funds. SENSITIVE."""
    return f"Success: Flight to {destination} confirmed and charger {total_cost:.2f} USD to company card."


SAFE_TOOLS = [get_current_weather, calculate_budget]
SENSITIVE_TOOLS = [book_trip_flight]
ALL_TOOLS = SAFE_TOOLS + SENSITIVE_TOOLS

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
llm_with_tools = llm.bind_tools(ALL_TOOLS)

system_instruction = SystemMessage(
    content=(
        "You are an enterprise travel assistant. "
        "You can check destination weather, calculate daily budgets, and book flights. "
        "When a user asks to book a trip or flight, call the book_trip_flight tool with the destination and total cost."
    )
)


def chatbot_node(state: MessagesState) -> dict:
    """The brain node: invokes gemini with accumulated messages."""
    messages = [system_instruction] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def route_tools(state: MessagesState) -> str:
    """Routes between safe tools, sensitive tools, and END."""
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None)
    if not tool_calls:
        return "end"

    sensitive_names = [t.name for t in SENSITIVE_TOOLS]
    for call in tool_calls:
        if call["name"] in sensitive_names:
            return "sensitive_tools"

    return "safe_tools"


builder = StateGraph(MessagesState)

builder.add_node("chatbot", chatbot_node)
builder.add_node("safe_tools", ToolNode(SAFE_TOOLS))
builder.add_node("sensitive_tools", ToolNode(SENSITIVE_TOOLS))

builder.add_edge(START, "chatbot")
builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {
        "safe_tools": "safe_tools",
        "sensitive_tools": "sensitive_tools",
        "end": END,
    },
)

builder.add_edge("safe_tools", "chatbot")
builder.add_edge("sensitive_tools", "chatbot")

app = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["sensitive_tools"],
)


def run_hitl_demo() -> None:
    """Demonstrates selective human-in-the-loop interruption."""
    session_config = {"configurable": {"thread_id": "manager-review-101"}}
    print("=" * 70)
    print(" HUMAN-IN-THE-LOOP (HITL) FINANCIAL GOVERNANCE DEMO")
    print("=" * 70)

    print("\n STEP 1: Testing Safe Tool (Tokyo Weather)...")
    res1 = app.invoke(
        {"messages": [HumanMessage(content="What is the weather in Tokyo?")]},
        config=session_config,  # type: ignore
    )
    print(" Response Received (Zero Pause):")
    print(f"   Agent: {res1['messages'][-1].content}")

    print("\n" + "-" * 70)
    print(" STEP 2: Requesting Flight Booking ($1,500 USD)...")
    print("-" * 70)
    # We invoke the agent with the booking request
    app.invoke(
        {
            "messages": [
                HumanMessage(content="Please book my flight to Tokyo for 1500 USD.")
            ]
        },
        config=session_config,  # type: ignore
    )

    current_state = app.get_state(session_config)  # type: ignore
    print("\n SECURITY ROADBLOCK TRIGGERED!")
    print(f"   Graph Next Step  : {current_state.next}")

    pending_tool_calls = current_state.values["messages"][-1].tool_calls
    print(f"   Pending Action   : {pending_tool_calls[0]['name']}")
    print(f"   Booking Details  : {pending_tool_calls[0]['args']}")
    print("   Status           : PAUSED on disk. Zero dollars spent.")
    print("\n" + "=" * 70)
    print(" HUMAN MANAGER ARRIVES FOR APPROVAL")
    print("=" * 70)
    print(
        f"Reviewing Request: Book flight to {pending_tool_calls[0]['args']['destination']} for ${pending_tool_calls[0]['args']['total_cost']} USD."
    )
    print("Manager Action: Clicking [ APPROVE ]...")

    print("\n STEP 4: Waking up agent with app.invoke(None)...")
    res_final = app.invoke(None, config=session_config)  # type: ignore
    print("\n EXECUTION COMPLETED!")
    last_msg = res_final["messages"][-1].content
    if (
        isinstance(last_msg, list)
        and len(last_msg) > 0
        and isinstance(last_msg[0], dict)
    ):
        last_msg = last_msg[0].get("text", "")
    print(f"   Agent Final Response: {last_msg}")
    print("=" * 70)


if __name__ == "__main__":
    run_hitl_demo()
