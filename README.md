# Multi-Agent LangGraph Demo

A Python-based AI application that demonstrates how to orchestrate a team of specialized AI agents using [LangGraph](https://python.langchain.com/docs/langgraph/) and the [Groq API](https://groq.com/).

This project aims to answer complex business or technical questions by breaking them down into steps, researching the components, drafting a response, critiquing it, and finalizing the answer.

## Workflow Overview

The system uses a state graph passing a `GraphState` containing `question`, `plan`, `research_notes`, `draft`, and `critique`. The graph consists of the following agents (nodes):
1. **Planner Agent**: Generates a structured plan with steps, key risks, and output headings.
2. **Researcher Agent**: Gathers practical insights and points for each section of the plan.
3. **Writer Agent**: Combines the plan and research notes into a cohesive first draft.
4. **Critic Agent**: Reviews the draft for missing points, overconfidence, and hallucination risks, scoring it out of 100. If the score is too low, the workflow can revise the draft.
5. **Finalizer Agent**: Takes the drafted answer and the critique to output a polished, final response.

## Prerequisites

- Python 3.12+
- `uv` (Fast Python package installer and resolver)

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/amerelfalwo/multiagnet_langgraph.git
   cd multiagnet_langgraph
   ```

2. **Set up the Environment:**
   Copy the `.env.example` file to create your own `.env` file:
   ```bash
   cp .env.example .env
   ```
   
   Fill in your API keys in the `.env` file:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   LANGSMITH_API_KEY=your_langsmith_api_key_here
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_PROJECT=langgraph-demo
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   GROQ_MODEL=openai/gpt-oss-20b
   ```

3. **Install Dependencies:**
   This project uses `uv` for dependency management:
   ```bash
   uv sync
   ```

## Usage

Run the main script to see both a single-agent response and the multi-agent LangGraph response for comparison.
```bash
uv run main.py
```

The script asks the default question:
*"Should a startup use open-source LLMs or closed models in 2026? Consider cost, speed, privacy, and reliability."*

## Architecture Details

- **`main.py`**: The entry point of the program. Triggers both single and multi-agent workflows.
- **`agents/single_agent.py`**: A simple direct query to the LLM for a baseline response.
- **`agents/multi_agent_graph.py`**: The core LangGraph workflow definitions, robust JSON parsing, node functions, conditional edges, and state models.
- **`agents/prompts.py`**: System prompts defining the persona and precise instructions for each agent.