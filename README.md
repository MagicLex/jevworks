# jevworks

[SemIf](https://github.com/TheoLeeCJ/SemIf) (formerly OpenJEV) semantic-if decisions served as a Hopsworks model deployment.

A decision is one forward pass. The model never generates text: the predictor builds a letter-choice prompt, reads the last-position logits at the answer-slot tokens, and softmaxes over those slots only.

## Layout

| Path | Role |
| --- | --- |
| `jevworks/semif/` | Engine vendored from SemIf (MIT). Only change: CPU device fallback. |
| `predictor.py` | Hopsworks Python predictor. Loads the registry model, scores rows. |
| `deploy.py` | Creates the deployment from the model registry. |
| `tests/test_decide.py` | Integration test against the running deployment. |
| `requirements.txt` | Installed into the inference environment. |

## Setup

```bash
export HOPSWORKS_HOST=eu-west.cloud.hopsworks.ai
export HOPSWORKS_API_KEY=...
export HOPSWORKS_PROJECT=jevworks
```

Model, environment, deployment:

```bash
python -c 'import hopsworks; hopsworks.login().get_model_registry().hf_download("Qwen/Qwen3-0.6B", selected_formats=["safetensors"])'
hops env clone jevworks-inference --from torch-inference-pipeline
hops env install -f requirements.txt jevworks-inference
python deploy.py
```

`deploy.py` flips the imported model's framework from `LLM` to `TORCH` so the Python model server is used instead of vLLM, uploads `predictor.py`, and starts the deployment with the CPU resources given by `--cores` and `--memory`. Pass `--gpus 1` for a GPU instance.

## Request

```bash
hops deployment predict semif --data '{"inputs": [{
  "id": "ticket-1",
  "state": "Password reset succeeded but every login still returns account locked.",
  "question": "Which team should handle this ticket?",
  "options": [
    {"id": "billing", "description": "Billing and refunds"},
    {"id": "account-access", "description": "Authentication, lockouts and account recovery"}
  ]}]}'
```

Response, one entry per row:

```json
{"predictions": [{
  "id": "ticket-1",
  "option_ids": ["billing", "account-access"],
  "probabilities": [0.03, 0.97],
  "option_logits": [12.1, 15.6],
  "input_tokens": 118,
  "forward_seconds": 0.41,
  "total_seconds": 0.43,
  "prompt_sha256": "...",
  "prompt_version": "direct-options-v1",
  "model": {"source": "...", "revision": "hopsworks:Qwen3_0_6B/1", "dtype": "float32", "device": "cpu"},
  "readout": "native full-vocabulary last-position logits restricted to declared answer slots",
  "probability_status": "conditional option score; uncalibrated as decision confidence"
}]}
```

Rows need `id`, `state` (string, object or array), `question`, and 2 to 16 `options` with unique `id` and a `description`. A row whose prompt exceeds 4096 tokens is rejected, never truncated.

## Predictor environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `SEMIF_DEVICE` | `auto` | `cuda`, `mps`, `cpu`, or `auto` (first available in that order) |
| `SEMIF_DTYPE` | `float32` on CPU, `bfloat16` otherwise | Weight dtype |
| `SEMIF_THREADS` | torch default | Intra-op thread count on CPU |

## Test

```bash
pip install -e '.[deploy]'
pytest
```
