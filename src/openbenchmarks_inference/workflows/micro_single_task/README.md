# Micro Single-Task Inference Benchmark

`micro-single-task` measures the responsiveness, generation speed, task success, and operational reliability a single latency-sensitive user experiences across inference providers.

Its central question is straightforward: when the model, prompt, task, and request settings are held fixed, which provider makes the user wait the least?

## What this workflow measures

The workflow runs GLM 5.3 Flash on deliberately easy, short, single-turn tasks. Expected task success is high, so the comparison emphasizes provider-side inference behavior rather than differences in model intelligence.

The fixed request contract is:

- GLM 5.3 Flash on every provider;
- OpenAI-compatible streaming chat completions;
- reasoning effort set to the provider-equivalent `low` configuration;
- temperature `0` and top-p `1`;
- JSON-object output with strict local schema validation;
- at most 256 completion tokens;
- one attempt and a 20-second timeout;
- one turn per task;
- per-provider concurrency of one;
- serverless or shared endpoints, with no hidden warmup requests.

The runner uses one pinned local `tiktoken` `o200k_base` tokenizer for every provider. This keeps token counts and speed calculations comparable even when provider usage payloads differ.

## Task families

The full run selects 200 items from each family, for 600 question rounds and 600 requests per provider.

### Meeting-notes lookup

The model reads a short meeting record and returns one requested fact as `{ "answer": string }`. Evaluation normalizes whitespace and letter case, then checks the canonical answer and permitted aliases.

### Ticket triage

The model assigns a support ticket both a category and priority. Evaluation requires the exact lowercase category/priority pair; partial credit is not used.

### Contract-terms extraction

The model extracts ten typed commercial-contract fields. Evaluation requires all fields and compares strings, ISO dates, booleans, integers, and nulls by type and value. The report also retains field-level diagnostics, but task success requires an exact object.

The dataset is not included in this repository. See [`docs/dataset-format.md`](../../../../docs/dataset-format.md) for its layout and validation contract.

## Providers

The bundled workflow contains adapters for:

- Z.AI
- Together AI
- Fireworks AI
- Baseten
- Telnyx
- Modal
- Nebius
- Parasail
- Novita AI
- DeepInfra

Provider adapters may differ only where their wire protocols require it: endpoint, model identifier, authentication headers, completion-token field name, reasoning syntax, top-k disabling, and streamed usage options. The logical request remains fixed.

The exact endpoints, model identifiers, documentation links, environment-variable names, and pricing inputs are recorded in [`assets/spec.yaml`](assets/spec.yaml) and enforced against [`providers.py`](providers.py). The bundled Modal arm targets the benchmark deployment recorded in that contract and requires its proxy credentials.

## Methodology and request schedule

Dataset selection and request order are deterministic and locked before execution.

1. Hash the frozen dataset release and deterministically shuffle each task family.
2. Select the first 200 items from each family.
3. Cycle through the three task families one item at a time.
4. For a question, prepare an identical logical request for every provider.
5. Rotate provider launch order by question to avoid consistently favoring one provider.
6. Release all provider calls for that question together.
7. Wait for every provider to complete, time out, or fail before starting the next question.

This is a paired comparison: providers see the same question at nearly the same time. Launch skew is recorded. Failed items remain in the denominator and are never replaced.

Every provider call has a hard 20-second client timeout. A timed-out call is recorded as a terminal failure, contributes to the timeout and failure-rate denominators, and releases that provider's place at the current question barrier. The runner still waits for every other provider in the round to reach a terminal outcome before advancing.

## Metrics

The workflow has no single headline metric. It reports latency, generation speed, task success, and operational reliability together so readers can select the measures relevant to their use case. Pooled results are accepted only when every task family contributes its required number of requests. Provider rows use balanced pooled E2E latency p99 as their deterministic display order, not as a designated headline metric.

- **TTFO p50/p95/p99:** request start to the first locally tokenized model output, whether reasoning or visible answer content.
- **TTFA p50/p95/p99:** request start to the first locally tokenized visible-answer token.
- **E2E latency p50/p95/p99:** request start to the final non-empty visible-answer token.
- **Pre-answer reasoning tokens p50/p95/p99:** locally tokenized reasoning emitted before or with the first answer-bearing event.
- **Reasoning emission rate:** share of responses with a TTFA sample that emitted reasoning before the answer.
- **Output tokens per second p50/p95/p99:** visible-answer tokens after the first token divided by time between the first and last answer tokens. Reasoning tokens are excluded.
- **Task success rate:** exact-correct, schema-valid answers divided by all submitted requests.
- **Failure rate:** timeouts, HTTP errors, and transport errors divided by all submitted requests.
- **Failure breakdown:** timeout, HTTP 429, other HTTP 4xx, HTTP 5xx, and transport-error rates.

Latency percentiles use nearest rank and higher percentiles are worse. Output generation speed is higher-is-better, so its public p95 and p99 values are lower-tail guarantees: 95% and 99% of eligible responses, respectively, generated at least that fast.

Latency populations exclude calls without the required timing event. Those calls still remain in submitted-request denominators for task success and failures, and the report discloses population coverage.

## Install and configure

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
```

Fill every credential in `.env`, then place the dataset at `data/micro-single-task/`. To use other paths, copy [`assets/spec.yaml`](assets/spec.yaml), update `dataset.source` and `dataset.manifest`, and pass that file with `--spec` on every applicable command.

## Run a smoke test

The smoke test runs one item from each family against all ten providers: 30 calls per attempt. It allows no retries, no more than two attempts, and a maximum combined smoke budget of $5.

```bash
openbench-inference smoke --workflow micro-single-task
```

The command creates a timestamped directory under `runs/micro-single-task/` and prints its path. To initialize and inspect the immutable schedule without making network calls:

```bash
openbench-inference init --workflow micro-single-task
```

## Run the full benchmark

A complete run makes 6,000 calls. The runner requires explicit authorization so it cannot be started accidentally:

```bash
openbench-inference run \
  --workflow micro-single-task \
  --full \
  --approve-full \
  --approval-id YOUR-AUDIT-ID
```

Resume an existing run by adding:

```bash
--run-dir runs/micro-single-task/run-YYYYMMDD-HHMMSS
```

Completed terminal request records are immutable and skipped on resume.

## Monitor a run

```bash
openbench-inference watch \
  --workflow micro-single-task \
  --run-dir runs/micro-single-task/run-YYYYMMDD-HHMMSS
```

Add `--once` for one JSON snapshot. The watcher reads durable local state only and makes no provider calls.

## Local artifacts

Each run directory contains:

- `run-lock.json`: hashes of the spec, dataset, selection, code, adapters, tokenizer, and resolved execution contract;
- `plan.json`: expected work and safety ceilings;
- `checkpoints/`: resumable attempt, round, and request state;
- `raw/`: terminal records and parsed streaming events;
- `artifacts/selection.json`: immutable selected IDs and item hashes;
- `artifacts/full-plan.json`: immutable paired request schedule;
- `artifacts/evaluations*.jsonl`: deterministic task evaluations;
- `artifacts/metrics.json`: task-family and balanced pooled metrics;
- `artifacts/paired-comparisons.json`: fastest-provider wins and pairwise latency differences;
- `artifacts/run-manifest*.json`: completeness, compatibility, coverage, and provenance;
- `report.md`: human-readable benchmark report;
- `artifacts/failures*.jsonl` and `events.jsonl`: operational diagnostics and lifecycle events.

Raw artifacts are the local source of truth. Derived evaluations, metrics, comparisons, and reports can be recomputed without issuing network requests. This OSS workflow does not publish data or write to a database.

## What this workflow does not claim

It does not measure provider saturation, high-concurrency throughput, dedicated endpoint performance, model intelligence, multi-turn agents, long-context behavior, or long-form generation. Results include the wait visible to the benchmark client: connection-pool waiting, network transit, provider queuing, prefill, and streamed generation.

## Implementation map

- [`config.py`](config.py): loads and validates the fixed workflow contract.
- [`dataset.py`](dataset.py): validates and hashes the frozen dataset.
- [`schedule.py`](schedule.py): deterministic selection and paired-round scheduling.
- [`providers.py`](providers.py): provider-specific wire mappings.
- [`transport.py`](transport.py): streaming HTTP transport and event timing.
- [`storage.py`](storage.py): immutable local records and checkpoints.
- [`evaluation.py`](evaluation.py): deterministic task scoring.
- [`metrics.py`](metrics.py): metric populations, percentiles, and paired analysis.
- [`runner.py`](runner.py): smoke/full execution and offline derivation.
- [`reporting.py`](reporting.py): Markdown report generation.
- [`watcher.py`](watcher.py): read-only progress snapshots.
