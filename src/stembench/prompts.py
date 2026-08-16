"""Prompt templates. Exact text is versioned and hashed into every run manifest."""

from __future__ import annotations

from stembench.schemas import sha256_of

MC_ANSWER_CONFIDENCE_V1_EN = """Solve the following multiple-choice problem. Reason briefly if needed, then give your final answer in exactly this format:

Answer: <letter>
Confidence: <integer 0-100>

Problem:
{question}
{choices}"""

MC_ANSWER_CONFIDENCE_V1_RU = """Решите следующую задачу с вариантами ответа. При необходимости кратко рассуждайте, затем укажите итоговый ответ строго в следующем формате:

Answer: <буква>
Confidence: <целое число 0-100>

Задача:
{question}
{choices}"""

FREE_ANSWER_CONFIDENCE_V1_EN = """Solve the following problem. Give the final answer in exactly this format:

Answer: <final answer>
Confidence: <integer 0-100>

Problem:
{question}"""

FREE_ANSWER_CONFIDENCE_V1_RU = """Решите следующую задачу. Укажите итоговый ответ строго в следующем формате:

Answer: <ответ>
Confidence: <целое число 0-100>

Задача:
{question}"""


def _format_choices_mc(labels_texts: list[tuple[str, str]]) -> str:
    return "\n".join(f"{label}. {text}" for label, text in labels_texts)


def render_mc(
    question: str, choices: list[str], language: str = "en", labels: list[str] | None = None
) -> str:
    """Render an MC prompt. `choices` are ordered texts (index 0 -> labels[0])."""
    labels = labels or ["A", "B", "C", "D", "E", "F"]
    assert len(choices) <= len(labels)
    body = _format_choices_mc(list(zip(labels[: len(choices)], choices, strict=True)))
    tpl = MC_ANSWER_CONFIDENCE_V1_RU if language == "ru" else MC_ANSWER_CONFIDENCE_V1_EN
    return tpl.format(question=question, choices=body)


def render_free(question: str, language: str = "en") -> str:
    tpl = FREE_ANSWER_CONFIDENCE_V1_RU if language == "ru" else FREE_ANSWER_CONFIDENCE_V1_EN
    return tpl.format(question=question)


TEMPLATES = {
    "mc_answer_confidence_v1": {
        "en": MC_ANSWER_CONFIDENCE_V1_EN,
        "ru": MC_ANSWER_CONFIDENCE_V1_RU,
    },
    "free_answer_confidence_v1": {
        "en": FREE_ANSWER_CONFIDENCE_V1_EN,
        "ru": FREE_ANSWER_CONFIDENCE_V1_RU,
    },
}


def template_text(name: str, language: str) -> str:
    return TEMPLATES[name][language]


def template_hash(name: str, language: str) -> str:
    return sha256_of({"template": name, "language": language,
                      "text": TEMPLATES[name][language]})
