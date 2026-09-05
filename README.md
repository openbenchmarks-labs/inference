# Inference Benchmark

Open, independent benchmark of LLM inference providers: latency, tokens per second,
and exact task success across 10 APIs serving the same GLM 5.3 Flash model.
This repository contains the Python runner, provider adapters, deterministic
scoring, and local reports for reproducing the measurement procedure.

[Live leaderboard and methodology](https://openbenchmarks.com/inference) ·
[Results JSON](https://openbenchmarks.com/api/benchmarks/inference) ·
[Public Inference Microtasks dataset](https://huggingface.co/datasets/openbenchmarks/OB-Inference-Microtasks)

## GLM 5.3 Flash API providers compared: latency, tokens per second and accuracy

Snapshot from run `run-20260902-223626`, reviewed **3 September 2026 (UTC)**.
Each provider received 600 paired requests across lookup, classification, and
extraction, with streaming on, reasoning low, and concurrency one.
The [live results API](https://openbenchmarks.com/api/benchmarks/inference) is the
source for these rounded figures; the [leaderboard](https://openbenchmarks.com/inference)
is the source of truth for subsequent runs.

| Provider | E2E p50 (ms) | E2E p99 (ms) | TTFA p50 (ms) | Tokens/s p50 | Tokens/s p95 floor | Task success | Operational failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseten | 330.8 | 3,228.8 | 287.1 | 240.5 | 61.6 | 95.83% | 0.00% |
| DeepInfra | 1,826.9 | 17,070.7 | 1,151.7 | 69.4 | 9.0 | 95.50% | 4.00% |
| Fireworks AI | 1,954.2 | 10,487.6 | 1,061.1 | 58.2 | 27.9 | 99.50% | 0.00% |
| Modal | 574.1 | 3,278.9 | 470.0 | 222.2 | 84.7 | 99.33% | 0.00% |
| Nebius | 1,450.7 | 14,970.4 | 1,144.3 | 269.3 | 78.8 | 84.83% | 14.83% |
| Novita AI | 1,738.3 | 7,248.3 | 1,404.7 | 60.6 | 29.6 | 94.17% | 4.00% |
| Parasail | 1,590.8 | 11,512.3 | 1,108.9 | 52.4 | 22.8 | 98.33% | 1.33% |
| Telnyx | 669.4 | 3,478.4 | 573.5 | 184.3 | 67.5 | 99.50% | 0.00% |
| Together AI | 608.6 | 12,449.7 | 393.2 | 151.6 | 34.0 | 98.50% | 1.00% |
| Z.AI | 1,748.8 | 8,591.2 | 1,391.2 | 56.3 | 27.8 | 97.50% | 0.17% |

Providers are alphabetical; no single metric determines an overall winner.
Lower latency is better. Higher generation speed and task success are better.
TTFA measures the wait for the first visible answer token. The tokens/s p95
floor is the speed at least 95% of eligible responses achieved, not an upper-tail
speed. Latency and speed require valid timing samples; failed requests are
excluded from latency calculations but remain in the task-success denominator.
Percentiles describe the observed distribution, not confidence intervals.

## Lowest-latency LLM inference provider: p99 tail analysis

Baseten had the lowest E2E p99 in this snapshot at **3,228.8 ms**, followed by
Modal at **3,278.9 ms**. Modal returned exact-correct answers on **99.33%** of
submitted requests, compared with Baseten's **95.83%**. Tail latency and usable
answer yield therefore need to be read together. See the
[latency comparison](https://openbenchmarks.com/inference/lowest-latency-llm-inference-provider)
for the current run.

## Fastest LLM inference API by tokens per second

Nebius had the highest median visible-answer speed at **269.3 tokens/s**, with
**84.83%** task success and **14.83%** operational failures. Modal had the
highest p95 speed floor at **84.7 tokens/s**, with **99.33%** task success and
no observed operational failures. Median speed describes a different experience
from the slow tail. See the
[tokens per second comparison](https://openbenchmarks.com/inference/fastest-streaming-inference-api).

## Same model, different accuracy: LLM inference provider endpoint accuracy

Fireworks AI and Telnyx tied at **99.50%** task success, with no observed
operational failures in their 600 requests each. Their E2E p99 values were
**10,487.6 ms** and **3,478.4 ms**, respectively. Task success counts schema-valid,
exact-correct answers across all submitted requests, including timeouts, HTTP
errors, malformed output, and incorrect answers in the denominator. See the
[endpoint accuracy comparison](https://openbenchmarks.com/inference/most-accurate-inference-provider).

## Methodology

The benchmark measures short, deliberately easy, one-turn structured tasks on
serverless or shared endpoints. It holds GLM 5.3 Flash, low reasoning effort,
streaming JSON-object output, temperature 0, top-p 1, zero presence and frequency
penalties, and a 256-token completion ceiling fixed. Each request has a 20-second
client timeout and one attempt.

The same 200 questions per task family are selected deterministically and sent
to every provider in paired rounds. Provider launch order rotates; the next
round waits for every provider to finish or fail. Per-provider concurrency
stays at one. A pinned local `tiktoken o200k_base` tokenizer measures visible
answer tokens for all providers.

Meeting-note answers match the canonical answer or allowed aliases after case
and whitespace normalization. Ticket triage requires the exact category and
priority. Contract extraction requires all ten fields with exact typed values
and ISO dates. Task success receives no partial credit. Operational failures
are counted separately from incorrect HTTP-successful answers.

TTFO ends at the first reasoning or answer token; TTFA ends at the first visible
answer token; E2E ends at the final non-empty visible answer delta. Generation
speed is `(N - 1) / (last answer-token time - first answer-token time)` for
eligible visible answers, excluding reasoning tokens. Complete definitions and
scope are in the [benchmark methodology](https://openbenchmarks.com/inference#methodology)
and [workflow documentation](src/openbenchmarks_inference/workflows/micro_single_task/README.md).

## Public dataset and reproduction scope

The [Inference Microtasks public dataset](https://huggingface.co/datasets/openbenchmarks/OB-Inference-Microtasks)
contains 29 synthetic examples: 13 meeting-note lookups, 6 ticket-triage tasks,
and 10 contract extractions, with prompts and reference answers. It supports
inspection of the task format. The published benchmark uses 600 selected
questions from a separate private corpus; the public sample does not satisfy
the bundled full-run dataset requirements or reproduce leaderboard scores.
The rows record a generation budget of 100 tokens; the runner's benchmark
contract sets a 256-token completion ceiling.

The runner is public and the sample data is open for inspection. Dataset files,
credentials, raw historical runs, and database publication code are supplied
separately from this repository. See the [dataset contract](docs/dataset-format.md)
for the inputs required to execute the workflow.

## Workflows

### [`micro-single-task`](src/openbenchmarks_inference/workflows/micro_single_task/README.md)

Measures which inference provider makes one latency-sensitive user wait the least on small, easy tasks. Ten providers serve GLM 5.3 Flash with streaming enabled, reasoning effort set to low, and per-provider concurrency fixed at one.

The full benchmark selects 600 questions: 200 meeting-note lookups, 200 support-ticket classifications, and 200 contract-term extractions. For each question, the runner calls every provider concurrently with a 20-second per-request timeout, waits until every call reaches a terminal outcome, and only then advances to the next question. This produces 600 requests per provider and 6,000 provider calls in a complete run.

The workflow reports balanced pooled E2E latency, TTFO, TTFA, visible-answer token generation speed, task success, and operational failure rates. No single metric is designated as the headline metric. Use the relevant measures together to compare interactive responsiveness and reliability. This workflow does not measure high-concurrency throughput, model intelligence, long-form generation, or agent performance.

See the [workflow README](src/openbenchmarks_inference/workflows/micro_single_task/README.md) for the complete benchmark contract, provider list, metric definitions, dataset requirements, commands, and output layout.

Future workflows will appear here and as sibling packages in the workflow registry. List those installed in your checkout with:

```bash
openbench-inference list-workflows
```

## Install the inference benchmark runner

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
```

Place the frozen dataset at `data/micro-single-task/`, or copy the bundled spec and change `dataset.source` and `dataset.manifest`. See the [`micro-single-task` workflow README](src/openbenchmarks_inference/workflows/micro_single_task/README.md) and [dataset contract](docs/dataset-format.md).

## Run the inference benchmark

Validate and lock all local inputs without making network calls:

```bash
openbench-inference init --workflow micro-single-task
```

Run the 30-request smoke test:

```bash
openbench-inference smoke --workflow micro-single-task
```

The command prints the created run directory. Watch it from another terminal:

```bash
openbench-inference watch \
  --workflow micro-single-task \
  --run-dir runs/micro-single-task/run-YYYYMMDD-HHMMSS
```

Run the complete 6,000-call benchmark only after the smoke passes:

```bash
openbench-inference run \
  --workflow micro-single-task \
  --full \
  --approve-full \
  --approval-id YOUR-AUDIT-ID
```

Use `--spec path/to/spec.yaml` on `init`, `smoke`, or `run` to use an explicit contract. The path must be supplied consistently when resuming a run because the runner locks and verifies the spec, dataset, adapters, tokenizer, and code.

## Outputs

Every run is local and resumable. It writes immutable selection and schedule artifacts, request terminal records, parsed stream events, reconstructed answers, evaluations, aggregate metrics, paired comparisons, a manifest, checkpoints, and `report.md`. Secrets are never written. No command in this repository writes to a database or publishes results.

The primary metrics are TTFO, TTFA, E2E latency, visible-answer output speed, task success, and operational failure rates. Percentiles use nearest rank. Output-speed p95 and p99 are lower-tail guarantees: respectively, 95% and 99% of eligible responses generated at least that fast.

## Add another workflow

Workflows are intentionally isolated rather than forced through premature shared abstractions. Add a sibling package and register it in `workflows/registry.py`; see [Adding a workflow](docs/adding-a-workflow.md).

Run the test suite with:

```bash
pytest
```

No license file has been added yet; the repository owner should choose the project license before public release.

## Links

- [Inference benchmark and live leaderboard](https://openbenchmarks.com/inference)
- [Benchmark methodology](https://openbenchmarks.com/inference#methodology)
- [Machine-readable results JSON](https://openbenchmarks.com/api/benchmarks/inference)
- [Public Inference Microtasks dataset on Hugging Face](https://huggingface.co/datasets/openbenchmarks/OB-Inference-Microtasks)
- [OpenBenchmarks benchmark directory for LLM assistants](https://openbenchmarks.com/llms.txt)
