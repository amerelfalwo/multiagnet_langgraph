import os
from dotenv import load_dotenv

load_dotenv()  # loads .env into os.environ

assert os.getenv("GROQ_API_KEY"), "Missing GROQ_API_KEY in .env"
# LangSmith is optional, but recommended
# assert os.getenv("LANGSMITH_API_KEY"), "Missing LANGSMITH_API_KEY in .env"

print("Tracing:", os.getenv("LANGCHAIN_TRACING_V2"))
print("Project:", os.getenv("LANGCHAIN_PROJECT"))


from agents.single_agent import single_agent_answer
from agents.multi_agent_graph import build_app

def run():
    q = "Should a startup use open-source LLMs or closed models in 2026? Consider cost, speed, privacy, and reliability."

    print("\n===== Single Agent =====")
    print(single_agent_answer(q))

    print("\n===== Multi-Agent (LangGraph) =====")
    app = build_app()
    result = app.invoke({
        "question": q,
        "plan": None,
        "research_notes": [],
        "draft": None,
        "critique": None,
        "iteration": 0,
        "max_iterations": 2,
    })
    print(result["draft"])

if __name__ == "__main__":
    run()
