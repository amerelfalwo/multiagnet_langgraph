# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LangGraph demonstration project comparing single-agent and multi-agent workflows using LangChain and Groq. The project showcases how to build a complex multi-agent system with specialized agents (Planner, Researcher, Writer, Critic, Finalizer) that collaborate through a state graph to produce well-reasoned recommendations.

## Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt
```

Required environment variables in `.env`:
- `GROQ_API_KEY` (required)
- `GROQ_MODEL` (optional, default: `openai/gpt-oss-20b`)
- `LANGSMITH_API_KEY` (optional but recommended for tracing)
- `LANGCHAIN_TRACING_V2=true`
- `LANGCHAIN_PROJECT=langgraph-youtube-demo`
- `LANGCHAIN_ENDPOINT=https://api.smith.langchain.com`

## Running the Application

```bash
python3 main.py
```

This runs both single-agent and multi-agent workflows on a demo question, printing results to console.

## Architecture

### Single Agent (agents/single_agent.py)
Simple baseline implementation using a single LLM call with a structured system prompt.

### Multi-Agent Graph (agents/multi_agent_graph.py)
Core LangGraph workflow implementing a specialized agent pipeline:

1. **Planner**: Creates structured plan with steps, risks, and output headings (uses structured output)
2. **Researcher**: Generates research notes from general knowledge (no web browsing)
3. **Writer**: Drafts answer based on plan and research notes
4. **Critic**: Evaluates draft, identifies issues, assigns quality score (uses structured output)
5. **Conditional Loop**: If critique score < 80 and iteration < max_iterations, returns to Writer for revision
6. **Finalizer**: Produces polished final output incorporating any critique

### Graph State (GraphState TypedDict)
Shared state passed between all nodes:
- `question`: User's question
- `plan`: Structured plan from Planner (Dict)
- `research_notes`: List of research findings
- `draft`: Current draft text
- `critique`: Structured critique from Critic (Dict with issues, score, fix_instructions)
- `iteration`: Current iteration count
- `max_iterations`: Maximum allowed iterations (default: 2)

### Structured Outputs
The project uses Pydantic models for structured LLM outputs:
- `Plan`: Contains steps, key_risks, desired_output_structure
- `Critique`: Contains issues, missing_points, hallucination_risk, score, fix_instructions

### Conditional Logic
`should_revise()` function in multi_agent_graph.py:124 determines workflow branching:
- Returns "finalize" if iteration >= max_iterations
- Returns "revise" if critique score < 80
- Otherwise returns "finalize"

## Key Dependencies

- `langchain==1.2.6`: Core framework
- `langchain-groq==1.1.3`: Groq integration
- `langgraph==1.0.6`: State graph workflow orchestration
- `langsmith==0.6.4`: Tracing and observability
- `pydantic==2.12.5`: Structured outputs and validation

## Model Configuration

All agents use `openai/gpt-oss-20b` (or specified via `GROQ_MODEL`) with temperature 0.2 (configured in agents/single_agent.py and agents/multi_agent_graph.py).
