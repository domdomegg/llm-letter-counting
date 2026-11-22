# llm-letter-counting

The eval behind [Can LLMs count letters yet? tldr: Yes.](https://adamjones.me/blog/llm-letter-counting-2026/)

It asks a model how many times a letter appears in each of 20 common English words, with a 2,000-token budget. Answers must end with an `ANSWER: <number>` line. A response with no parseable answer, including one that runs out of budget mid-thought, scores as no answer rather than wrong.

Built on [Inspect](https://inspect.aisi.org.uk/).

## Running it

```bash
uv venv && uv pip install -r requirements.txt

# Anthropic API
export ANTHROPIC_API_KEY=...
inspect eval task.py --model anthropic/claude-haiku-4-5

# Local models via Ollama
inspect eval task.py --model ollama/gemma4:e2b
inspect eval task.py --model ollama/qwen3.5:4b -T reasoning=none   # thinking off
```

`inspect view` opens the transcripts.

The blog post's results were run once per model (`--epochs 1`), with Ollama 0.32.4 on an M2 Pro MacBook with 16 GB of RAM.
