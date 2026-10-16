"""Phase 1 — Scoping: turns the conversation with the user into a research brief.

If the request is too vague, the graph ends with a clarifying question for the user.
If it is clear enough, it produces the brief that will guide the research.
"""
import os
from datetime import datetime
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


def get_today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


class ClarifyWithUser(BaseModel):
    """Decides whether the user must be asked a question before researching."""
    need_clarification: bool = Field(description="True ONLY if the topic, the place or the goal is missing. If the request already says what to research, False. When in doubt, False")
    question: str = Field(description="The question for the user, short and in the language the user wrote in ('' if no question is needed)")
    verification: str = Field(description="Message saying you are starting the research, summarizing what you understood, in the language the user wrote in ('' if a question is needed)")


class ResearchQuestion(BaseModel):
    """Detailed research question that will guide the researcher."""
    research_brief: str = Field(description="A detailed research question, in first person and in the language the user wrote in")


class ScopeState(MessagesState):
    research_brief: str


CLARIFY_PROMPT = """These are the messages exchanged with the user so far:
<Messages>
{messages}
</Messages>

Today is {date}.

Decide whether you need to ask the user a question before starting the research.
- Ask ONLY if the topic, the place or the goal of the research is missing.
- If the request already says what to research, do NOT ask: the researcher
  will decide any missing details.
- Do not ask for information that can be found on the internet.
- If you ALREADY asked a question before, do not ask again.
- When in doubt, do NOT ask.
Reply in the language the user used in their messages above: if they wrote in
English, reply in English; if in Spanish, in Spanish; and so on."""

BRIEF_PROMPT = """These are the messages exchanged with the user so far:
<Messages>
{messages}
</Messages>

Today is {date}.

Turn this conversation into ONE detailed research question.
- Include ALL the details and preferences the user gave.
- Do not invent preferences the user did not state. If a detail is missing
  (dates, budget, sources...), say it is open.
- Do not add goals the user did not ask for (for example "the impact" or
  "the influence" if they did not mention them).
- Write it in first person, as if the user were asking.
- Write it in the language the user used in their messages above: if they wrote
  in English, write in English; if in Spanish, in Spanish; and so on."""


def conversation_as_text(state: ScopeState) -> str:
    return "\n".join(f"{m.type}: {m.content}" for m in state["messages"])


def clarify(state: ScopeState) -> Command[Literal["write_brief", "__end__"]]:
    prompt = CLARIFY_PROMPT.format(messages=conversation_as_text(state), date=get_today())
    decision = model.with_structured_output(ClarifyWithUser).invoke([HumanMessage(content=prompt)])

    if decision.need_clarification:
        return Command(goto=END, update={"messages": [AIMessage(content=decision.question)]})
    return Command(goto="write_brief", update={"messages": [AIMessage(content=decision.verification)]})


def write_brief(state: ScopeState) -> dict:
    prompt = BRIEF_PROMPT.format(messages=conversation_as_text(state), date=get_today())
    response = model.with_structured_output(ResearchQuestion).invoke([HumanMessage(content=prompt)])
    return {"research_brief": response.research_brief}


graph = StateGraph(ScopeState)
graph.add_node("clarify", clarify)
graph.add_node("write_brief", write_brief)
graph.add_edge(START, "clarify")
graph.add_edge("write_brief", END)

app = graph.compile(checkpointer=InMemorySaver())


def main():
    config = {"configurable": {"thread_id": "1"}}

    while True:
        text = input("User: ").strip()
        if not text:
            continue
        if text.lower() == "exit":
            break

        result = app.invoke({"messages": [HumanMessage(content=text)]}, config)
        print(f"AI: {result['messages'][-1].content}")

        if "research_brief" in result:
            print(f"\nBRIEF:\n{result['research_brief']}")
            break


if __name__ == "__main__":
    main()
