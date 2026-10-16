"""Deep research agent — entry point.

Flow:
    main()  Phase 1: conversation with the user until a brief is obtained (scope.app)
    brief → START → supervisor ⇄ supervisor_tools → write_report → END
                              │ for each ConductResearch:
                              └─► researcher.app, in parallel (asyncio.gather)

The report is saved to outputs/report_<date>_<time>.md.

Run (from the repository root):  uv run python agent/deep_research.py
"""
import asyncio
import operator
import os
import time
from datetime import datetime
from typing import Annotated, Literal, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from pydantic import BaseModel, Field

from scope import app as scope_app
from researcher import app as researcher_app

load_dotenv()

MAX_ROUNDS = 6              # supervisor calls at most (reflection rounds count too)
MAX_RESEARCHERS = 3         # topics per round, stated to the model in the prompt

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")


def get_today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


llm = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


# ---------- Supervisor tools ----------
class ConductResearch(BaseModel):
    """Delegates the research of ONE topic to a researcher. Use it once per distinct topic."""
    research_topic: str = Field(description="The topic to research, described in detail in one sentence, in the language the brief is written in.")


class ResearchComplete(BaseModel):
    """Use it when the notes received already cover the brief and no more research is needed."""


@tool
def think_tool(reflection: str) -> str:
    """Use it to think BEFORE delegating (which topics do I need?) and AFTER receiving
    notes (do they cover the brief? what is missing?). Write your reasoning in reflection."""
    return f"Reflection recorded: {reflection}"


supervisor_model = llm.bind_tools([ConductResearch, ResearchComplete, think_tool])

SUPERVISOR_PROMPT = """You are a research supervisor. Today is {date}.
You receive a brief. You do NOT research yourself: you DELEGATE with ConductResearch, one form per topic.

How to work:
- Use think_tool before delegating and after receiving notes, to reason about your decision.
- Split the brief into FEW independent topics. In a comparison, one topic per compared
  element (for example, a country), and that topic must include ALL the aspects the
  brief asks for. Do not create one topic per aspect.
- Simple questions: 1 or 2 topics. At most {max_researchers} topics per round.
- When you receive the researchers' notes, check whether they cover the brief.
  If something important is missing, delegate ONLY what is missing. If it is covered, use ResearchComplete.
- Do not repeat a topic that already has notes."""

REPORT_PROMPT = """You are a writer. Today is {date}.
Write a Markdown report that answers this brief:

{brief}

Use ONLY the information in these notes (each note includes its numbered sources):

{notes}

Rules:
- Structure: title, a short introduction, one section per topic or compared element,
  and a conclusion that answers the brief directly.
- Cite sources with [n] after each fact, and end with a "Sources" section
  listing the URLs. If two notes repeat the same URL, include it only once.
- If the notes do not cover something the brief asks for, say so explicitly. Do not invent.
- Write in the language the brief is written in (English brief → English, Spanish brief → Spanish)."""


# ---------- State ----------
class ResearchState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]   # brief, supervisor tool calls and notes
    notes: Annotated[list[str], operator.add]                 # notes from every researcher
    rounds: int
    report: str
    file: str


# ---------- Nodes ----------
async def supervisor(state: ResearchState) -> Command[Literal["supervisor_tools"]]:
    prompt = SUPERVISOR_PROMPT.format(date=get_today(), max_researchers=MAX_RESEARCHERS)
    response = await supervisor_model.ainvoke([SystemMessage(content=prompt)] + state["messages"])

    print(f"\n--- Round {state['rounds'] + 1}: the supervisor requests {[c['name'] for c in response.tool_calls]}")
    return Command(
        goto="supervisor_tools",
        update={"messages": [response], "rounds": state["rounds"] + 1},
    )


async def supervisor_tools(state: ResearchState) -> Command[Literal["supervisor", "write_report"]]:
    last = state["messages"][-1]
    requests = [c for c in last.tool_calls if c["name"] == "ConductResearch"]
    reflections = [c for c in last.tool_calls if c["name"] == "think_tool"]
    finished = any(c["name"] == "ResearchComplete" for c in last.tool_calls)

    if finished:
        print("    Done: the supervisor declared the research complete")
        return Command(goto="write_report", update={})
    if not last.tool_calls:
        print("    Done: the supervisor requested nothing")
        return Command(goto="write_report", update={})
    if state["rounds"] >= MAX_ROUNDS:
        print(f"    Done: reached the limit of {MAX_ROUNDS} rounds")
        return Command(goto="write_report", update={})

    tool_messages = []
    for r in reflections:
        print(f"    💭 {r['args']['reflection'][:120]}")
        tool_messages.append(ToolMessage(content=think_tool.invoke(r["args"]), tool_call_id=r["id"]))
    if not requests:
        return Command(goto="supervisor", update={"messages": tool_messages})

    print(f"    Launching {len(requests)} researchers in parallel...")
    start = time.perf_counter()
    tasks = [
        researcher_app.ainvoke({"turns": 0, "messages": [HumanMessage(content=c["args"]["research_topic"])]})
        for c in requests
    ]
    results = await asyncio.gather(*tasks)

    for result, request in zip(results, requests):
        print(f"    ✓ {request['args']['research_topic']}")
        tool_messages.append(ToolMessage(content=result["notes"], tool_call_id=request["id"]))
    print(f"    ({time.perf_counter() - start:.0f} s)")

    return Command(
        goto="supervisor",
        update={"messages": tool_messages, "notes": [r["notes"] for r in results]},
    )


def write_report(state: ResearchState) -> dict:
    prompt = REPORT_PROMPT.format(
        date=get_today(),
        brief=state["messages"][0].content,
        notes="\n\n---\n\n".join(state["notes"]),
    )
    print("\n📝 Writing the report...")
    report = llm.invoke([HumanMessage(content=prompt)]).content

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    path = os.path.join(OUTPUTS_DIR, f"report_{datetime.now().strftime('%Y-%m-%d_%H%M')}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)

    return {"report": report, "file": os.path.relpath(path)}


# ---------- Graph ----------
graph = StateGraph(ResearchState)
graph.add_node("supervisor", supervisor)
graph.add_node("supervisor_tools", supervisor_tools)
graph.add_node("write_report", write_report)
graph.add_edge(START, "supervisor")
graph.add_edge("write_report", END)
research_app = graph.compile()


# ---------- Program ----------
async def main():
    config = {"configurable": {"thread_id": "1"}}
    brief = None

    while True:
        text = input("User: ").strip()
        if not text:
            continue
        if text.lower() == "exit":
            break

        result = scope_app.invoke({"messages": [HumanMessage(content=text)]}, config)
        print(f"AI: {result['messages'][-1].content}")

        if "research_brief" in result:
            brief = result["research_brief"]
            print(f"\nBRIEF:\n{brief}")
            break

    if brief is None:
        print("No brief, nothing to research.")
        return

    print("\n🔎 Researching (this takes a few minutes)...")
    start = time.perf_counter()
    result = await research_app.ainvoke(
        {"messages": [HumanMessage(content=brief)], "notes": [], "rounds": 0}
    )

    print(f"\n{'=' * 70}\n{result['report']}\n{'=' * 70}")
    print(f"\n💾 Report saved to {result['file']}  ({time.perf_counter() - start:.0f} s total)")


if __name__ == "__main__":
    asyncio.run(main())
