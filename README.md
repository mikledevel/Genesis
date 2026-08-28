# Genesis AI

Genesis AI is an all-in-one platform that bundles four products behind a single login:

- **A conversational AI agent** (Groq-backed) that can build ML pipelines, train models, and generate code through natural-language chat
- **A no-code AutoML engine** — upload a CSV and get a trained, versioned model (logistic regression, random forest, XGBoost, and more)
- **A chatbot/agent builder + marketplace** — build a bot with a system prompt and knowledge base, publish it, and charge other users per message
- **A model marketplace** — publish a trained model and charge other users per prediction call

Built with FastAPI (Python) on the backend and a vanilla JS/HTML frontend, using Groq for LLM inference.

## Features

- Chat-driven dataset analysis, model training, and retraining
- Model registry with versioning and lineage tracking
- Custom pipeline builder with reusable blocks
- Knowledge-base retrieval (RAG) per bot
- Prompt Lab: versioned system prompts with A/B testing
- Telegram bot deployment
- Marketplace with per-call/per-message billing
- Per-user API keys for external programmatic access
- Quota management and usage/cost observability for LLM calls

## Requirements

- Python 3.11+
- A [Groq API key](https://console.groq.com)
- (Optional) Docker, for sandboxed execution of AI-generated code

## Setup

```bash
git clone https://github.com/misha622/Genesis.git
cd Genesis

python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Configure environment variables:

```bash
cp .env.example .env
```

Edit `.env` and set:

```
GROQ_API_KEY=your_real_groq_api_key
SECRET_KEY=<generate with the command below>
```

Generate a secure secret key:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

> **Never commit `.env`.** It's already in `.gitignore` — only `.env.example` (with placeholder values) belongs in version control.

## Running

```bash
uvicorn app.main:app --reload --port 8000
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## Running tests

```bash
python -m pytest tests/ -v
```

A few tests that execute AI-generated code in a sandbox require a running Docker daemon; without it, those specific tests are skipped/fail gracefully rather than running unsandboxed code.

## Project structure

```
app/
  main.py              # FastAPI routes
  core/                # Agent, engine (training/registry), marketplace, memory, codegen, etc.
  db/                  # SQLite persistence layer
static/                # Frontend (dashboard, agent chat, pipeline builder)
tests/                 # pytest suite
```

## Security

If you discover a security issue, please open an issue or reach out directly rather than filing a public report with exploit details.

## License

TBD