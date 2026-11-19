# Installation Guide

OpenBench AI requires **Python 3.10 or newer** (the test suite is run on Python 3.10, 3.11, 3.12 and 3.13 on Linux) and requires zero third-party runtime dependencies for its core library and CLI.

## 1. Local Editable Installation (Recommended for Development)

```bash
git clone https://github.com/lakshaychhaba0001-sys/openbench-ai.git
cd openbench-ai

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade pip and install OpenBench AI with dev extras
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## 2. Core-Only Installation (Zero Third-Party Dependencies)

```bash
python -m pip install -e .
```

## 3. Verify CLI & Package Import

```bash
openbench --version
# Expected output (Python version and platform will differ):
# openbench 0.5.0 (Python 3.12.4, Linux-6.8.0-x86_64-with-glibc2.39)

python -c "import openbench; print(openbench.__version__)"
# Expected output: 0.5.0
```

## 4. Optional Extras

| Extra | Installs | Used for |
| --- | --- | --- |
| `dev` | pytest, ruff, PyYAML, jsonschema | Tests, linting, YAML profiles, JSON Schema checks |
| `config` | PyYAML | YAML profiles |
| `metrics` | jsonschema | `structured_output` JSON Schema validation |
| `semantic` | sentence-transformers | Neural `semantic_similarity` |
| `huggingface` | transformers, torch | The experimental Hugging Face adapter |
| `stats-ref` | numpy, scipy, statsmodels | **Tests only**: the statistical reference cross-checks in `tests/reference`. The library never imports them. |
| `docs` | mkdocs, mkdocs-material | Building the documentation site |
| `all` | All runtime integrations above | |

Statistics, repeated runs, paired comparison, ranking, sample-size planning, the LLM judge, human preferences, HTML reports and the dashboard need no extra. Audits need none either, except the semantic near-duplicate check, which uses `semantic`.

## 5. Offline / Air-Gapped Environment Installation

```bash
python -m pip install --no-build-isolation --no-deps -e .
```

## 6. Running with Docker

OpenBench AI can be run in a container without installing Python locally. The image is built on `python:3.12-slim` with the core package only (no optional extras), runs as a non-root user, and uses `openbench` as its entrypoint.

### Build the Image

```bash
docker build -t openbench .
```

Verify the container works:

```bash
docker run --rm openbench --version
```

### Mounting a Working Directory (Non-Root Permissions)

Because the container runs as a non-root user, mounting a host directory with `-v "$PWD:/work"` can fail with permission errors when writing evaluation outputs unless the container process matches your host user's UID and GID. Pass `--user "$(id -u):$(id -g)"` so OpenBench can write results to the mounted folder:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" \
  -w /work \
  openbench run \
  --dataset datasets/reasoning/sample.jsonl \
  --provider mock \
  --model demo-model \
  --experiments-dir /work/results \
  --no-cache
```

### Reaching an Ollama Server Running on the Host

* **Linux (`--network host` — recommended):** Sharing the host network lets the container reach Ollama at `http://localhost:11434`, which remains a loopback address and passes OpenBench's endpoint security checks without extra configuration:

```bash
docker run --rm \
  --network host \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" \
  -w /work \
  openbench run \
  --dataset datasets/reasoning/sample.jsonl \
  --provider ollama \
  --model llama3.2 \
  --endpoint http://localhost:11434 \
  --experiments-dir /work/results
```

* **macOS and Windows (Docker Desktop):** The host machine is exposed inside the container at `http://host.docker.internal:11434` (start Ollama on the host with `OLLAMA_HOST=0.0.0.0 ollama serve` so it listens beyond localhost). Note that because `host.docker.internal` is a non-loopback hostname, OpenBench's endpoint security validation requires enabling `allow_remote_http=True` (see the [Local Ollama Integration Guide](ollama.md)).
