# API Reference

Base URL: `http://localhost:8000`

## Health

### `GET /health`

Returns service status.

```json
{"status":"ok","cuda":true,"models":1}
```

## Models

### `GET /v1/models`

Lists loaded models.

```json
{
  "object": "list",
  "data": [{"id":"qwen3-asr","object":"model","created":0,"owned_by":"blackwell"}]
}
```

## Audio Transcriptions

### `POST /v1/audio/transcriptions`

Transcribes an audio file with Qwen3-ASR.

**Request**: `multipart/form-data`
- `file`: audio file (wav/mp3/m4a)
- `model`: optional, default `qwen3-asr`
- `language`: optional

**Response**

```json
{"text": "Hello world."}
```

## Chat Completions

### `POST /v1/chat/completions`

Placeholder for text LLM support.

**Request**

```json
{
  "model": "qwen3-0.6b",
  "messages": [{"role":"user","content":"hello"}],
  "max_tokens": 256,
  "temperature": 0.0
}
```

**Response**

```json
{
  "id": "chatcmpl-blackwell",
  "object": "chat.completion",
  "created": 1720000000,
  "model": "qwen3-0.6b",
  "choices": [{"index":0,"message":{"role":"assistant","content":"..."},"finish_reason":"stop"}],
  "usage": {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}
}
```

## Benchmark

### `POST /v1/benchmark`

Runs an attention or ASR benchmark.

**Attention request**

```json
{
  "task": "attention",
  "backend": "sdpa",
  "batch_size": 1,
  "seq_len": 2048,
  "repeats": 10
}
```

**ASR request**

```json
{
  "task": "asr",
  "backend": "compile",
  "model": "/path/to/Qwen3-ASR-0.6B",
  "audio": "/path/to/audio.wav",
  "batch_size": 1
}
```

**Response**

```json
{
  "task": "attention",
  "backend": "sdpa",
  "latency_ms": 0.057,
  "throughput": 37.5,
  "details": {"seq_len":1024,"head_dim":64,"max_err":0.0}
}
```

## Profile

### `GET /v1/profile?backend=sdpa&seq_len=2048`

Returns a torch.profiler kernel summary for the given backend.

**Response**

```json
{
  "backend": "sdpa",
  "batch": 1,
  "heads": 8,
  "seq_len": 2048,
  "head_dim": 64,
  "causal": false,
  "total_kernel_ms": 0.206,
  "top_kernels": [...]
}
```

### `GET /v1/profiles`

Lists saved profile summaries.

```json
{"profiles": ["profile_summary_sdpa_n2048.json", ...]}
```
