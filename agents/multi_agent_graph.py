import os
from dotenv import load_dotenv
from typing import Optional, TypedDict, Dict, Any, List

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()
from .prompts import PLANNER_SYSTEM, RESEARCHER_SYSTEM, WRITER_SYSTEM, CRITIC_SYSTEM, FINALIZER_SYSTEM

class GraphState(TypedDict):
    question: str
    plan: Optional[Dict[str, Any]]
    research_notes: List[str]
    draft: Optional[str]
    critique: Optional[Dict[str, Any]]
    iteration: int
    max_iterations: int

#Define structured outputs
class Plan(BaseModel):
    steps: List[str] = Field(default_factory=list, description="Short ordered steps for solving the task.")
    key_risks: List[str] = Field(default_factory=list, description="Major risks/unknowns that should be addressed.")
    desired_output_structure: List[str] = Field(default_factory=list, description="Headings to include in final answer.")

class Critique(BaseModel):
    score: int = Field(default=75, ge=0, le=100, description="Overall quality score of the draft (0-100).")
    issues: List[str] = Field(default_factory=list, description="Concrete problems with the current draft.")
    missing_points: List[str] = Field(default_factory=list, description="Important missing considerations.")
    hallucination_risk: List[str] = Field(default_factory=list, description="Claims that might be risky without sources.")
    fix_instructions: List[str] = Field(default_factory=list, description="Actionable steps to improve the draft.")


model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
max_tokens = int(os.getenv("GROQ_MAX_TOKENS", "4096"))
llm = ChatGroq(model=model_name, temperature=0.2, max_tokens=max_tokens)

import json
import re

def parse_plan_content(text: str) -> Plan:
    try:
        return Plan.model_validate_json(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return Plan.model_validate_json(match.group(0))
        except Exception:
            pass
    return Plan(
        steps=["Analyze question", "Research key tradeoffs", "Draft structured recommendation", "Review and finalize"],
        key_risks=["Model performance", "Cost & infrastructure", "Privacy & security"],
        desired_output_structure=["Summary", "Tradeoffs", "Recommendation", "Risks", "Confidence"]
    )

def parse_critique_content(text: str) -> Critique:
    try:
        return Critique.model_validate_json(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return Critique.model_validate_json(match.group(0))
        except Exception:
            pass
    return Critique(
        score=85,
        issues=[],
        missing_points=[],
        hallucination_risk=[],
        fix_instructions=["Refine structure and polish final recommendation."]
    )

def planner_node(state: GraphState) -> GraphState:
    try:
        structured_planner = llm.with_structured_output(Plan)
        plan_obj = structured_planner.invoke([
            SystemMessage(content=PLANNER_SYSTEM),
            HumanMessage(content=state["question"])
        ])
    except Exception:
        resp = llm.invoke([
            SystemMessage(content=PLANNER_SYSTEM + '\nReturn JSON: {"steps": [...], "key_risks": [...], "desired_output_structure": [...]}'),
            HumanMessage(content=state["question"])
        ]).content
        plan_obj = parse_plan_content(resp)

    state["plan"] = plan_obj.model_dump() if hasattr(plan_obj, "model_dump") else plan_obj
    return state

def researcher_node(state: GraphState) -> GraphState:
    resp = llm.invoke([
        SystemMessage(content=RESEARCHER_SYSTEM),
        HumanMessage(content=f"Question:\n{state['question']}\n\nPlan:\n{state['plan']}")
    ]).content

    # store as notes (simple split)
    notes = [line.strip("- ").strip() for line in resp.split("\n") if line.strip()]
    state["research_notes"] = notes
    return state

def writer_node(state: GraphState) -> GraphState:
    resp = llm.invoke([
        SystemMessage(content=WRITER_SYSTEM),
        HumanMessage(content=f"""
Question:
{state['question']}

Plan:
{state['plan']}

Research notes:
{state['research_notes']}

If critique exists, you may improve the draft accordingly.
Critique:
{state.get('critique')}
""")
    ]).content

    state["draft"] = resp
    return state

def critic_node(state: GraphState) -> GraphState:
    try:
        structured_critic = llm.with_structured_output(Critique)
        critique_obj = structured_critic.invoke([
            SystemMessage(content=CRITIC_SYSTEM),
            HumanMessage(content=f"""
Question:
{state['question']}

Draft:
{state['draft']}
""")
        ])
    except Exception:
        resp = llm.invoke([
            SystemMessage(content=CRITIC_SYSTEM + '\nReturn JSON format: {"score": 75, "issues": [...], "missing_points": [...], "hallucination_risk": [...], "fix_instructions": [...]}'),
            HumanMessage(content=f"""
Question:
{state['question']}

Draft:
{state['draft']}
""")
        ]).content
        critique_obj = parse_critique_content(resp)

    state["critique"] = critique_obj.model_dump() if hasattr(critique_obj, "model_dump") else critique_obj
    state["iteration"] = int(state.get("iteration", 0)) + 1

    return state




def finalizer_node(state: GraphState) -> GraphState:
    draft_content = str(state.get('draft', ''))[:3000]
    research_content = str(state.get('research_notes', ''))[:1000]
    critique_content = str(state.get('critique', ''))[:1000]

    resp = llm.invoke([
        SystemMessage(content=FINALIZER_SYSTEM),
        HumanMessage(content=f"""
Question:
{state['question']}

Plan:
{state['plan']}

Research notes (truncated):
{research_content}

Critique (if any, truncated):
{critique_content}

Current draft (if any, truncated):
{draft_content}
""")
    ]).content

    state["draft"] = resp
    return state


from typing import Literal
from langgraph.graph import StateGraph, END

def should_revise(state: GraphState) -> Literal["revise", "finalize"]:
    """
    Stop conditions:
    - If we hit max_iterations => finalize
    - Else if critique score is below threshold => revise
    - Else finalize
    """
    critique = state.get("critique") or {}
    score = int(critique.get("score", 100))

    # Safety: if iteration/max_iterations missing, default to no loop
    iteration = int(state.get("iteration", 0))
    max_iterations = int(state.get("max_iterations", 0))

    if iteration >= max_iterations:
        return "finalize"

    # Threshold you can tune live (80 is a good demo default)
    if score < 80:
        return "revise"

    return "finalize"


def build_app():
    workflow = StateGraph(GraphState)

    # Nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("finalizer", finalizer_node)

    # Entry
    workflow.set_entry_point("planner")

    # Edges
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "critic")

    # Conditional loop or finalize
    workflow.add_conditional_edges(
        "critic",
        should_revise,
        {
            "revise": "writer",
            "finalize": "finalizer",
        },
    )

    workflow.add_edge("finalizer", END)

    return workflow.compile()
