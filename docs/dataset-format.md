# Micro-single-task dataset contract

The dataset itself is not distributed in this repository. The bundled spec expects:

```text
data/micro-single-task/
├── manifest.json
├── meeting-notes-lookup.jsonl
├── ticket-triage.jsonl
└── contract-terms-extraction.jsonl
```

`manifest.json` must identify a frozen, freeze-ready release and report the exact row count for each family:

```json
{
  "status": "frozen",
  "freeze_ready": true,
  "use_cases": {
    "meeting-notes-lookup": {"private_items": 237},
    "ticket-triage": {"private_items": 244},
    "contract-terms-extraction": {"private_items": 240}
  }
}
```

Each JSONL row must contain `id`, `use_case`, `split`, `prompt`, `output_schema`, `gold`, `aliases`, and `tokens`. IDs must be globally unique. `prompt` contains exactly `system` and `user`; the system text must equal the corresponding bundled prompt template. The loader validates task-specific schemas, answer types, input-token bands, counts, and hashes before a run is created.

The workflow deterministically shuffles each family using the benchmark slug, dataset release hash, task family, and schedule version. It freezes the first 200 IDs per family into the run lock. Failed requests are retained and are never replaced.

Do not commit private benchmark data. `data/private/` is ignored, and projects embedding this package should apply an equivalent rule to their configured dataset directory.

