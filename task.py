"""Can LLMs count letters? A minimal Inspect eval.

    inspect eval task.py --model anthropic/claude-haiku-4-5
    inspect eval task.py --model ollama/qwen3.5:4b -T reasoning=none
"""

import re

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    NOANSWER,
    AnswerPattern,
    Score,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState, generate

TOKEN_BUDGET = 2000

# (word, letter, count): common English words, answers spread across 0/1/2/3.
ITEMS = [
    ("discussion", "s", 3),
    ("combination", "i", 2),
    ("settlement", "t", 3),
    ("orange", "i", 0),
    ("following", "n", 1),
    ("wonderful", "s", 0),
    ("different", "t", 1),
    ("statistics", "s", 3),
    ("population", "o", 2),
    ("beautiful", "e", 1),
    ("vegetable", "r", 0),
    ("everything", "r", 1),
    ("helicopter", "o", 1),
    ("elephant", "e", 2),
    ("statistics", "n", 0),
    ("hospital", "l", 1),
    ("television", "e", 2),
    ("beginning", "a", 0),
    ("management", "n", 2),
    ("remarkable", "r", 2),
]

PROMPT = (
    'How many times does the letter "{letter}" appear in the string below (case-insensitive)?\n\n'
    "{text}\n\n"
    f"You have a budget of {TOKEN_BUDGET} tokens. Reason briefly if you need to, but you "
    "must commit to an answer within the budget.\n\n"
    "Respond with your final answer on its own line in exactly this format:\n"
    "ANSWER: <number>"
)


def parse_answer(completion: str) -> int | None:
    """The integer from a trailing `ANSWER: <n>` line, or None if the format wasn't followed."""
    match = re.search(AnswerPattern.LINE, completion)
    if match is None:
        return None
    payload = match.group(1).strip().rstrip(".")
    return int(payload) if re.fullmatch(r"[+-]?\d+", payload) else None


@scorer(metrics=[accuracy(), stderr()])
def letter_count():
    """Exact match. No parseable answer (e.g. budget ran out) scores NOANSWER, not INCORRECT."""

    async def score(state: TaskState, target: Target) -> Score:
        answer = parse_answer(state.output.completion)
        if answer is None:
            value = NOANSWER
        else:
            value = CORRECT if answer == int(target.text) else INCORRECT
        return Score(
            value=value,
            answer="" if answer is None else str(answer),
            metadata={"stop_reason": state.output.stop_reason},
        )

    return score


@task
def letter_counting(reasoning: str | None = None) -> Task:
    """`reasoning=none` turns thinking off for Ollama models.

    Inspect drops effort="none" rather than sending it, so it goes via extra_body.
    """
    return Task(
        dataset=MemoryDataset(
            [
                Sample(input=PROMPT.format(letter=letter, text=word), target=str(count))
                for word, letter, count in ITEMS
            ]
        ),
        solver=generate(),
        scorer=letter_count(),
        config=GenerateConfig(
            max_tokens=TOKEN_BUDGET,
            extra_body={"reasoning": {"effort": reasoning}} if reasoning else None,
        ),
    )
