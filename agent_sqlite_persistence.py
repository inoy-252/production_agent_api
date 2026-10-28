from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from trimmed_agent import prepare_trimmed_context

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


@tool
def book_trip_flight(destination: str, total_cost: float) -> str:
    """Books a corporate flight and commits company financial funds. SENSITIVE."""
    return f"Success: Flight to {destination} confirmed and charged {total_cost} USD to comapny card"


SAFE_TOOLS = [get_current_weather, calculate_budget, search_knowledge_base]
SENSITIVE_TOOLS = [book_trip_flight]
ALL_TOOLS = SAFE_TOOLS + SENSITIVE_TOOLS

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
llm_with_tools = llm.bind_tools(ALL_TOOLS)

system_instruction = SystemMessage(
    content=(
        "You are a dedicated enterprise travel and corporate policy assistant. "
        "Be friendly, professional, and concise.\n\n"
        "STRICT DOMAIN SCOPE & ZERO-LEAK REFUSAL:\n"
        "- Your ONLY role is assisting employees with business travel planning, trip budgeting, destination weather, and internal company travel policies.\n"
        "- You are STRICTLY PROHIBITED from answering general knowledge, trivia, history, coding, cooking, recipes, sports, entertainment, or any topics outside corporate travel.\n"
        "- ZERO-LEAK RULE: If the user asks ANY question outside this domain (e.g. recipes, baking, coding, trivia, general facts), state the refusal below and STOP immediately:\n"
        "  'I am strictly configured as an enterprise travel assistant. I can only assist with travel arrangements, trip budgets, destination weather, and internal company policies.'\n"
        "- ABSOLUTE PROHIBITION: NEVER provide any partial answers, tips, hints, ingredients, steps, or code. NEVER say 'I am a travel assistant, BUT here is a recipe'. State the refusal and NOTHING ELSE.\n"
        "- For company policies, NEVER guess or use external knowledge. Only use information retrieved from the search_knowledge_base tool or details shared by the user in this conversation.\n\n"
        "CRITICAL TOOL CALLING RULES:\n"
        "- When a user introduces themselves or shares travel plans, simply greet them warmly and ask how you can help. Do NOT execute any tools yet.\n"
        "- ONLY call get_current_weather when the user explicitly asks for the weather.\n"
        "- ONLY call calculate_budget when the user explicitly requests their budget calculation or daily allowance.\n"
        "- ONLY call search_knowledge_base when the user asks about company policies, baggage allowances, or safety protocols.\n"
        "- ONLY call book_trip_flight when the user explicitly asks to book a flight or trip, providing the destination and total cost."
        "- Always write currency amounts cleanly (e.g. '340.00 USD' or '1,700 USD'). Never write duplicate currency codes like 'USD 340.00 USD', and never use the '$' dollar sign symbol."
    )
)


def chatbot_node(state: MessagesState) -> dict:
    """The brain node: invokes Gemini with trimmed message context."""
    messages = prepare_trimmed_context(state, system_instruction, max_history_turns=6)
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def route_tools(state: MessagesState) -> str:
    """Selectively routes to sensitive_tools or safe_tools."""
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
    {"safe_tools": "safe_tools", "sensitive_tools": "sensitive_tools", "end": END},
)
builder.add_edge("safe_tools", "chatbot")
builder.add_edge("sensitive_tools", "chatbot")

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
