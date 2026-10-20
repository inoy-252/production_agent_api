from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


@tool
def get_current_weather(city: str) -> str:
    """Get the current weather forecast for a specified city."""
    city_clean = city.strip().lower()
    mock_weather = {
        "tokyo": "15°C, Clear with cool evening winds",
        "london": "11°C, Light drizzle",
        "paris": "14°C, Mild and cloudy",
        "new york": "18°C, Partly sunny",
    }
    return mock_weather.get(city_clean, f"20 C, Pleasent and sunny in {city}")


@tool
def calculate_budget(total_budget: float, days: int) -> str:
    """Calculate the daily spending allowance for a travel trip."""
    if days <= 0:
        return "Error: Number of days must be greater than zero."
    daily = total_budget / days
    return f"Daily spending allowance is ${daily:.2f} per day over {days} days."


tools = [get_current_weather, calculate_budget]

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
llm_with_tools = llm.bind_tools(tools)

system_instruction = SystemMessage(
    content=(
        "You are a professional travel assistant. "
        "Be friendly and concise. "
        "Only execute tools when the user explicitly asks for weather or budget calculations."
    )
)


def chatbot_node(state: MessagesState) -> dict:
    """The brain node: invokes Gemini with accumulated messages."""
    messages = [system_instruction] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


builder = StateGraph(MessagesState)

builder.add_node("chatbot", chatbot_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")

DB_PATH = "agent_memory.db"


def run_agent_turn(prompt: str, thread_id: str):
    """Executes a single turn of the agent using the SQLite checkpointer."""
    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        app = builder.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        print(f"\n--- [Thread: {thread_id}] User: {prompt} ---")

        events = app.stream(
            {"messages": [HumanMessage(content=prompt)]}, config, stream_mode="values"
        )
        last_message = None
        for event in events:
            if "messages" in event:
                last_message = event["messages"][-1]

        if last_message:
            print(f"Agent: {last_message.content}")


if __name__ == "__main__":
    yasir_session = "user-yasir-101"
    emran_session = "user-emran-101"

    print("=== Turn 1: Seed information into Yasir's session ===")
    run_agent_turn(
        "Hi! I am Yasir. I'm going to Tokyo for 5 days with a $1500 budget.",
        thread_id=yasir_session,
    )

    print("\n=== TURN 2: Implicit Recall (Yasir asks without repeating details) ===")
    run_agent_turn(
        "What's the weather in my trip city, and what is my daily allowance?",
        thread_id=yasir_session,
    )
    print("\n=== TURN 3: Tenant Isolation (Yasir asks for his budget) ===")
    run_agent_turn("What is my daily budget?", thread_id=emran_session)
