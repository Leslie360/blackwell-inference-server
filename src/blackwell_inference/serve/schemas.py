"""Pydantic schemas for OpenAI-compatible API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "blackwell"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class TranscriptionRequest(BaseModel):
    model: str
    language: str | None = None
    response_format: str = "json"


class TranscriptionResponse(BaseModel):
    text: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.0
    use_spec: bool = False
    gamma: int = 4
    ngram: int = 3


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: dict[str, int]


class BenchmarkRequest(BaseModel):
    task: str  # "attention" or "asr"
    backend: str = "sdpa"
    model: str | None = None
    audio: str | None = None
    batch_size: int = 1
    seq_len: int = 2048
    repeats: int = 10


class BenchmarkResponse(BaseModel):
    task: str
    backend: str
    latency_ms: float
    throughput: float
    details: dict[str, Any]
