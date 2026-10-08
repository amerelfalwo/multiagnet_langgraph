import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

load_dotenv()

SINGLE_AGENT_SYSTEM = """You are a helpful AI.
Task: Provide a well-reasoned recommendation to the user question.
Rules:
- Make your best effort without browsing the web.
- Be structured: Summary, Pros, Cons, Recommendation, Risks, Confidence (0-100).
"""

model_name = os.getenv("GROQ_MODEL", "meta-llama/llama-3.1-405b-instruct")
max_tokens = int(os.getenv("GROQ_MAX_TOKENS", "4096"))
llm = ChatGroq(model=model_name, temperature=0.2, max_tokens=max_tokens)
def single_agent_answer(question: str) -> str:
    msgs = [
        SystemMessage(content=SINGLE_AGENT_SYSTEM),
        HumanMessage(content=question),
    ]
    return llm.invoke(msgs).content