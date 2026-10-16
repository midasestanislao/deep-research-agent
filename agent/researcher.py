"""Phase 2 — Researcher: researches one topic on the web and returns notes with sources.

An agent ⇄ tools loop (search + think_tool) with a hard turn limit, followed by a
final node that compresses the raw search results into clean notes with [n] citations.
"""
import os
from datetime import datetime
from typing import Annotated, Sequence, TypedDict

from ddgs import DDGS
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, filter_messages
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

MAX_TURNS = 10    # model calls at most; on the last one it must answer without tools


def get_today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


class ResearcherState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    turns: int
    notes: str


@tool
def search(query: str) -> str:
    """Searches the internet for current information. Returns up to 3 results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
    except Exception as e:
        return f"The search failed: {e}"

    if not results:
        return "No results found."

    return "\n\n".join(f"{r['title']}\n{r['body']}\n({r['href']})" for r in results)


@tool
def think_tool(reflection: str) -> str:
    """Use it AFTER each search to reflect before continuing: what information
    you found, what is still missing, and whether you need to search more or
    can already answer."""
    return f"Reflection recorded: {reflection}"


tools = [search, think_tool]

model = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)
model_with_tools = model.bind_tools(tools)
model_without_tools = model

RESEARCHER_PROMPT = """You are a research assistant. Today is {date}.
You will receive a brief describing what the user wants to research.
Use search to look up information and think_tool to think between searches.

How to research:
1. Read the brief: what concrete information do you need?
2. Start with broad searches, then make them more specific.
3. After EACH search, use think_tool: what did I find? what is missing?
   do I keep searching or can I already answer?
4. Stop when you can answer the brief well, or when your last
   2 searches added nothing new.

Limits:
- Simple questions: 2-3 searches at most.
- Complex questions: up to 5 searches.

Final answer: in the language the brief is written in (English brief → English,
Spanish brief → Spanish), with the information found and the URLs of the sources you used."""

COMPRESS_PROMPT = """You are an assistant that cleans up the results of a research session.
Today is {date}.

Research brief:
{brief}

Search results:
{results}

Write CLEAN NOTES:
- Keep ALL the data relevant to the brief (names, dates, figures).
  Do not over-summarize.
- Remove duplicates and anything unrelated to the brief.
- After each fact, cite its source with a number in brackets, e.g. [1].
- End with a "Sources" section listing the numbered URLs.
- Do not invent anything that is not in the results."""


def research(state: ResearcherState) -> dict:
    current_model = model_without_tools if state["turns"] >= MAX_TURNS else model_with_tools
    prompt = RESEARCHER_PROMPT.format(date=get_today())
    response = current_model.invoke([SystemMessage(content=prompt)] + state["messages"])
    return {"messages": [response], "turns": state["turns"] + 1}


def compress(state: ResearcherState) -> dict:
    results = filter_messages(state["messages"], include_names=["search"])
    prompt = COMPRESS_PROMPT.format(
        date=get_today(),
        brief=state["messages"][0].content,
        results="\n\n".join(m.content for m in results),
    )
    response = model_without_tools.invoke([HumanMessage(content=prompt)])
    return {"notes": response.content}


def should_continue(state: ResearcherState) -> str:
    if state["messages"][-1].tool_calls:
        return "continue"
    return "end"


graph = StateGraph(ResearcherState)
graph.add_node("research", research)
graph.add_node("tools", ToolNode(tools=tools))
graph.add_node("compress", compress)

graph.add_edge(START, "research")
graph.add_conditional_edges("research", should_continue, {"continue": "tools", "end": "compress"})
graph.add_edge("tools", "research")
graph.add_edge("compress", END)

app = graph.compile()


if __name__ == "__main__":
    topic = input("Topic to research: ").strip()
    result = app.invoke({"turns": 0, "messages": [HumanMessage(content=topic)]})
    print(f"\nTurns: {result['turns']}\n\nNOTES:\n{result['notes']}")
