# Model Providers & Adapter Framework

OpenBench AI provides a unified adapter architecture in `openbench.adapters` so local inference servers, local Python functions, Hugging Face pipelines, and deterministic test mocks share a single contract.

---

## 1. Registered Providers

| Provider Identifier | Class | Type | Dependencies | Description |
| :--- | :--- | :--- | :--- | :--- |
| `ollama` | `OllamaAdapter` | Local HTTP API | None (`urllib.request` stdlib) | Connects to a local Ollama server (`http://localhost:11434`). |
| `openai_compatible` (`openai`, `openai-compatible`) | `OpenAICompatibleAdapter` | HTTP API (v0.5.0) | None (`urllib.request` stdlib) | Any server exposing `/v1/chat/completions`: vLLM, LM Studio, llama.cpp server, Ollama's OpenAI endpoint and hosted APIs. |
| `huggingface_transformers` (`huggingface`, `hf`) | `HuggingFaceAdapter` | Local Pipeline (Experimental) | `transformers`, `torch` (optional) | Runs local Hugging Face `text-generation` pipelines with `local_files_only=True` by default. |
| `local_callable` | `LocalCallableAdapter` | In-Process Python Callable | None | Wraps any Python function `fn(prompt: str) -> str \| dict`. |
| `mock` | `MockModelProvider` | Deterministic Test Mock | None | Offline mock provider (`is_mock=True`) for tests, CI, and failure simulation. Never represents a real model. |
| `local_lookup` | `DeterministicLookupAdapter` | In-Memory Dictionary Lookup | None | Returns canned responses mapped by record ID (`is_mock=True`). |

Plugins can add more providers; see [plugins.md](plugins.md).

---

## OpenAI-compatible servers (v0.5.0)

`openai_compatible` sends each prompt as a single user message to `POST {endpoint}/chat/completions` and reads `choices[0].message.content`. It works with any server that follows that format.

```bash
# vLLM / llama.cpp server / LM Studio on this machine (no key needed)
openbench run --dataset datasets/reasoning/sample.jsonl --provider openai_compatible \
  --endpoint http://localhost:8000/v1 --model my-model --seed 7

# A hosted API: the key is read from an environment variable at request time
export OPENAI_API_KEY=...            # or any variable, named with --api-key-env
openbench run --dataset datasets/reasoning/sample.jsonl --provider openai_compatible \
  --endpoint https://api.example.com/v1 --model some-model --max-requests-per-minute 60
```

| Profile key (`provider:`) | CLI flag | Default | Meaning |
| --- | --- | --- | --- |
| `endpoint` | `--endpoint` | `http://localhost:8000/v1` | Base URL, including the `/v1` prefix if the server uses one. |
| `api_key_env` | `--api-key-env` | `OPENAI_API_KEY` | Name of the environment variable holding the key. If it is unset, no `Authorization` header is sent. |
| `system_prompt` | - | none | Optional system message sent before every prompt. |
| `allow_remote_http` | - | `false` | Plain `http://` is only allowed for `localhost`/`127.0.0.1` unless this is `true`. |

Behaviour:
- **Seeds.** `generation.seed` (or `--seed`) is sent as `seed`, and lineage records `seed_status = passed_to_provider`. Servers may ignore it.
- **Tokens.** `usage.prompt_tokens` and `usage.completion_tokens` are recorded when the server returns them, so `performance.pricing` cost estimates work.
- **Rate limits.** HTTP 429 is classified as `rate_limit` and retried; a `Retry-After` header is honoured (capped by `execution.max_retry_after_seconds`). Use `--max-requests-per-minute` to stay under a provider's limit. See [reliability.md](reliability.md).
- **Security.** The key is read from the environment for each request and is never written to configs, caches, run directories, logs or reports. Error messages are scrubbed of credentials. When a run is resumed, the key is read from `OPENAI_API_KEY` unless the profile names another variable.
- `openbench providers check openai_compatible --endpoint ... --model ...` calls `GET {endpoint}/models` and reports whether the model is listed.

---

## 2. CLI Provider Inspection

```bash
# List all registered model providers
openbench providers list

# Check connectivity and readiness of a provider
openbench providers check mock
openbench providers check ollama --endpoint http://localhost:11434 --model llama3.2
openbench providers check huggingface --model gpt2
```
