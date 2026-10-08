# langgraph-demo

In first step, you will need to create a python virtual env
```
virtualenv venv
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Create `.env` file and fill your api keys:
```
GROQ_API_KEY=your_groq_key_here
GROQ_MODEL=openai/gpt-oss-20b
LANGSMITH_API_KEY=your_langsmith_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=langgraph-youtube-demo
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com

```

# Running the app
```
python3 main.py
```