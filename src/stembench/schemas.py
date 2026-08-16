"""Pydantic schemas: validated contracts for items, run configs, records, manifests."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class Language(str, Enum):
    RU = "ru"
    EN = "en"


class Subject(str, Enum):
    MATH = "math"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"


class Difficulty(str, Enum):
    SCHOOL = "school"
    UNIVERSITY = "university"
    OLYMPIAD = "olympiad"


class AnswerType(str, Enum):
    MC = "mc"
    EXACT = "exact"
    NUMERIC = "numeric"


class Split(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_of(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Choice(BaseModel):
    label: str  # "A" | "B" | "C" | "D"
    text: str


class Tolerance(BaseModel):
    rel: Optional[float] = None  # relative tolerance, e.g. 0.01 = 1%
    abs: Optional[float] = None  # absolute tolerance


class VerifierRecord(BaseModel):
    method: str  # e.g. "symbolic", "numeric_recompute", "dimensional"
    passed: bool
    detail: str = ""


class BenchmarkItem(BaseModel):
    """One language-specific record of the original bilingual STEM benchmark."""

    item_id: str  # f"{pair_id}-{language}"
    pair_id: str  # links ru/en variants of one semantic item
    language: Language
    subject: Subject
    topic: str
    difficulty: Difficulty
    difficulty_rubric: str = ""  # evidence for the difficulty assignment
    question: str
    answer_type: AnswerType
    choices: list[Choice] = Field(default_factory=list)  # required for MC
    canonical_answer: str  # letter for MC; canonical string/number otherwise
    acceptable_alternatives: list[str] = Field(default_factory=list)
    tolerance: Optional[Tolerance] = None  # for NUMERIC
    units: str = ""
    solution: str  # worked solution or concise rationale
    provenance: str = "original_procedural"
    license: str = "CC-BY-4.0"
    author: str = "LLM-STEMBench procedural generator"
    creation_method: str = "template+parameters, seeded"
    translator: str = "parallel templates authored bilingually"
    verifier: list[VerifierRecord] = Field(default_factory=list)
    annotation_version: str = "0"
    quality_flags: list[str] = Field(default_factory=list)
    split: Split = Split.TEST
    contamination_notes: str = "generated 2026-08-17; parameters post-cutoff"

    @field_validator("choices")
    @classmethod
    def mc_needs_choices(cls, v: list[Choice], info):  # noqa: ANN001
        if info.data.get("answer_type") == AnswerType.MC and not v:
            raise ValueError("MC items require choices")
        return v


class MCItem(BaseModel):
    """A multiple-choice item as presented to a model (any dataset)."""

    item_id: str
    dataset: str
    subject: str
    language: Language = Language.EN
    difficulty: str = "unknown"
    question: str
    choices: list[str]  # ordered, index 0 = A
    gold: int  # 0-based index of correct choice


class FreeResponseItem(BaseModel):
    item_id: str
    dataset: str
    subject: str
    language: Language = Language.EN
    difficulty: str = "unknown"
    question: str
    answer_type: AnswerType = AnswerType.EXACT
    gold: str  # canonical reference answer
    alternatives: list[str] = Field(default_factory=list)
    tolerance: Optional[Tolerance] = None
    units: str = ""


class DecodingSettings(BaseModel):
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: Optional[float] = None
    seed: Optional[int] = None
    notes: str = ""


class ModelSpec(BaseModel):
    provider: str  # registry key, e.g. "openrouter"
    model: str  # exact model ID at the provider
    display_name: str = ""
    decoding: DecodingSettings = Field(default_factory=DecodingSettings)
    n_items: Optional[int] = None  # per-model cap (budget), None = all

    @property
    def key(self) -> str:
        return f"{self.provider}::{self.model}"


class SampleSpec(BaseModel):
    n: int
    seed: int = 42
    stratify_by: str = "subject"
    split: str = "test"


class RunConfig(BaseModel):
    run_id: str
    stage: int  # 1 or 2
    dataset: str  # "mmlu_stem" | "stembench_v1"
    dataset_revision: str = ""
    sample: SampleSpec
    models: list[ModelSpec]
    languages: list[Language] = Field(default_factory=lambda: [Language.EN])
    prompt_template: str = "mc_answer_confidence_v1"
    output_dir: str = "results"
    request_timeout_s: int = 180
    max_retries: int = 3
    dry_run: bool = False
    notes: str = ""


class Usage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class LogprobTop(BaseModel):
    token: str
    logprob: float


class ResponseRecord(BaseModel):
    """Normalized per-item, per-model record. One JSONL line."""

    run_id: str
    item_id: str
    dataset: str
    dataset_revision: str = ""
    subject: str = ""
    language: str = "en"
    difficulty: str = "unknown"
    answer_type: str = "mc"
    provider: str
    model: str  # exact model ID requested
    model_reported: str = ""  # model ID echoed by provider, if any
    decoding: dict[str, Any] = Field(default_factory=dict)
    prompt_hash: str = ""
    prompt_text: str = ""
    started_at: str = ""
    finished_at: str = ""
    latency_ms: Optional[int] = None
    raw_response: str = ""
    finish_reason: str = ""
    usage: Optional[Usage] = None
    logprobs_raw: Optional[list[dict[str, Any]]] = None  # provider top_logprobs, if exposed
    parsed_answer: Optional[str] = None
    parse_method: str = ""
    reference_answer: str = ""
    correctness: Optional[bool] = None  # None = unparseable/abstained
    self_reported_confidence: Optional[float] = None  # 0..1
    confidence_provenance: list[str] = Field(default_factory=list)
    estimated_cost: Optional[float] = None
    error_status: str = ""  # "", "rate_limited", "timeout", "provider_error", "parse_failure"
    code_commit: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


class Manifest(BaseModel):
    run_id: str
    stage: int
    dataset: str
    dataset_revision: str = ""
    dataset_snapshot: dict[str, Any] = Field(default_factory=dict)
    sample: dict[str, Any] = Field(default_factory=dict)
    prompt_template: str = ""
    prompt_template_text: str = ""
    prompt_template_hash: str = ""
    models: list[dict[str, Any]] = Field(default_factory=list)
    budget_usage: dict[str, Any] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    status: str = "running"  # running|complete|partial
    started_at: str = ""
    finished_at: str = ""
    git_commit: str = ""
    code_dirty: bool = False
    notes: str = ""
