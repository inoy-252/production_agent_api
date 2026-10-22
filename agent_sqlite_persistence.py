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
    return f"Daily spending allowance is {daily:.2f} USD per day over {days} days."


COMPANY_KNOWLEDGE = {
    "baggage_policy": (
        "Company Travel Policy Section 4: All employees on international business travel "
        "are permitted up to two checked bags (max 23kg each) reimbursed by the company. "
        "Overweight bag fees are not covered."
    ),
    "meal_allowance": (
        "Company Travel Policy Section 7: Daily meal per diem for Tier 1 international cities "
        "(including Tokyo, London, NYC) is fixed at 75 USD per day. Alcohol expenses require "
        "pre-approval from finance."
    ),
    "emergency_protocol": (
        "Travel Safety Runbook: In case of medical emergencies or natural disasters in Tokyo, "
        "contact the 24/7 Global SOS hotline at +1-800-555-0199 and report your status to "
        "travel-safety@company.com within 2 hours."
    ),
}


@tool
def search_knowledge_base(query: str) -> str:
    """Searches internal company travel policies, baggage limits, meal per diems, and safety runbooks."""

    stop_words = {
        "company",
        "travel",
        "policy",
        "for",
        "the",
        "in",
        "what",
        "is",
        "our",
    }
    query_words = [w for w in query.lower().split() if w not in stop_words]

    scored_matches = []
    for title, text in COMPANY_KNOWLEDGE.items():
        score = sum(1 for w in query_words if w in text.lower() or w in title.lower())
        if score > 0:
            clean_text = text.replace("$", "USD ")
            scored_matches.append((score, f"[{title.upper()}]: {clean_text}"))

    scored_matches.sort(key=lambda x: x[0], reverse=True)
    if scored_matches:
        return scored_matches[0][1]
    return "No matching company policy found for this query."


tools = [get_current_weather, calculate_budget, search_knowledge_base]

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
llm_with_tools = llm.bind_tools(tools)

system_instruction = SystemMessage(
    content=(
        "You are a professional enterprise travel assistant. "
        "Be friendly and concise. "
        "You have access to internal company travel policies and safety runbooks via search_knowledge_base. "
        "CRITICAL BEHAVIORAL RULES: "
        "- When a user introduces themselves or shares travel plans, simply greet them warmly and ask how you can help. Do NOT execute any tools yet. "
        "- ONLY call get_current_weather when the user explicitly asks for the weather. "
        "- ONLY call calculate_budget when the user explicitly requests their budget calculation or daily allowance. "
        "- ONLY call search_knowledge_base when the user asks about company policies, baggage allowances, or safety protocols. "
        "- Always write currency amounts cleanly (e.g. '340.00 USD' or '1,700 USD'). Never write duplicate currency codes like 'USD 340.00 USD', and never use the '$' dollar sign symbol."
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
    print("\n=== TURN 3: Tenant Isolation (Emran asks for his budget) ===")
    run_agent_turn("What is my daily budget?", thread_id=emran_session)

    print("\n=== TURN 4: Autonomous RAG Knowledge Search ===")
    run_agent_turn(
        "What is our company's baggage allowance policy for international flights, and who do I contact in an emergency in Tokyo?",
        thread_id=yasir_session,
    )
