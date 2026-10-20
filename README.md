# LangChain Hello World

A simple Python example that uses a LangChain prompt and chat model to turn a sample biography into a short summary and two interesting facts.

## Setup

Requires Python 3.14+, uv, and an OpenAI API key.

```bash
git clone --branch project/hello-world https://github.com/Sud-m/langchain.git
cd langchain
uv sync
```

Create a `.env` file in the project root:

```dotenv
OPENAI_API_KEY=your-api-key
```

## Run

```bash
uv run python main.py
```

The example uses `gpt-5`. Edit the sample text, prompt, or model in `main.py` to experiment. An alternative `ChatOllama` configuration is included as a comment for local models.
