# Adding an inference workflow

Create a new package beside `micro_single_task`:

```text
src/openbenchmarks_inference/workflows/<workflow_name>/
├── __init__.py
├── config.py
├── lifecycle.py
├── runner.py
└── ...workflow-owned modules and assets
```

Expose `WORKFLOW_NAME` and `load_config` from the package, then add one lazy import mapping to `workflows/registry.py`. The shared CLI discovers lifecycle, runner, and watcher modules only after a workflow is selected, so optional dependencies and provider code from one workflow do not affect another.

A workflow should own and test:

- its frozen configuration contract and input validation;
- dataset loading and deterministic selection;
- provider wire mappings and secret redaction;
- scheduling, concurrency, barriers, retries, and timeouts;
- durable checkpoints and resumability;
- evaluation, metric populations, percentile conventions, and reporting;
- smoke ceilings and full-run authorization guards.

Keep implementation inside the workflow package until at least two workflows have the same proven abstraction. Publication and application-specific database code belong outside this runner repository.
