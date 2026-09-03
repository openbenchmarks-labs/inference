# OpenBenchmarks Inference

Open-source runners for reproducible inference-provider benchmarks. Each workflow owns its contract, provider adapters, scheduling, evaluation, metrics, documentation, and reporting under `src/openbenchmarks_inference/workflows/`.

This repository intentionally contains no benchmark dataset, credentials, historical runs, database publication code, migrations, or website code.

## Workflows

### [`micro-single-task`](src/openbenchmarks_inference/workflows/micro_single_task/README.md)

Measures which inference provider makes one latency-sensitive user wait the least on small, easy tasks. Ten providers serve GLM 5.3 Flash with streaming enabled, reasoning effort set to low, and per-provider concurrency fixed at one.

The dataset contains 600 selected questions: 200 meeting-note lookups, 200 support-ticket classifications, and 200 contract-term extractions. For each question, the runner calls every provider concurrently with a 20-second per-request timeout, waits until every call reaches a terminal outcome, and only then advances to the next question. This produces 600 requests per provider and 6,000 provider calls in a complete run.

The workflow reports balanced pooled E2E latency, TTFO, TTFA, visible-answer token generation speed, task success, and operational failure rates. No single metric is designated as the headline metric. Use the relevant measures together to compare interactive responsiveness and reliability—not high-concurrency throughput, model intelligence, long-form generation, or agent performance.

See the [workflow README](src/openbenchmarks_inference/workflows/micro_single_task/README.md) for the complete benchmark contract, provider list, metric definitions, dataset requirements, commands, and output layout.

Future workflows will appear here and as sibling packages in the workflow registry. List those installed in your checkout with:

```bash
openbench-inference list-workflows
```

## Install

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
```

Place the frozen dataset at `data/micro-single-task/`, or copy the bundled spec and change `dataset.source` and `dataset.manifest`. See the [`micro-single-task` workflow README](src/openbenchmarks_inference/workflows/micro_single_task/README.md) and [dataset contract](docs/dataset-format.md).

## Run

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
