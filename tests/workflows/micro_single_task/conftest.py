from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from openbenchmarks_inference.workflows.micro_single_task.config import DEFAULT_SPEC, load_config


def _contract_gold() -> dict[str, object]:
    return {
        "agreement_type": "saas_subscription",
        "effective_date": "2026-01-01",
        "initial_term_end_date": "2026-12-31",
        "auto_renews": True,
        "renewal_term_months": 12,
        "non_renewal_notice_days": 30,
        "payment_terms_days": 30,
        "termination_for_convenience": False,
        "termination_notice_days": None,
        "security_incident_notice_hours": 24,
    }


@pytest.fixture(scope="session")
def synthetic_config(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("micro-single-task")
    dataset_root = root / "data"
    dataset_root.mkdir()
    spec = yaml.safe_load(DEFAULT_SPEC.read_text(encoding="utf-8"))
    spec["runner"]["local_run_root"] = str(root / "runs")
    spec["dataset"]["source"] = str(dataset_root)
    spec["dataset"]["manifest"] = str(dataset_root / "manifest.json")

    for filename in DEFAULT_SPEC.parent.joinpath("prompt_templates").glob("*.txt"):
        target = root / "prompt_templates" / filename.name
        target.parent.mkdir(exist_ok=True)
        target.write_text(filename.read_text(encoding="utf-8"), encoding="utf-8")

    counts = {
        "meeting-notes-lookup": 237,
        "ticket-triage": 244,
        "contract-terms-extraction": 240,
    }
    for family, count in counts.items():
        template_name = spec["dataset"]["task_families"][family]["prompt_template"]
        system = (root / template_name).read_text(encoding="utf-8").removesuffix("\n")
        rows = []
        for index in range(count):
            if family == "meeting-notes-lookup":
                schema = {"answer": "string"}
                gold = {"type": "person", "value": "Ada"}
                tokens = {"input_total": 500}
            elif family == "ticket-triage":
                schema = {"category": "enum", "priority": "enum"}
                gold = {"category": "bug", "priority": "p2"}
                tokens = {"user": 50}
            else:
                schema = {
                    "agreement_type": "enum",
                    "effective_date": "string",
                    "initial_term_end_date": "string",
                    "auto_renews": "bool",
                    "renewal_term_months": "int | null",
                    "non_renewal_notice_days": "int | null",
                    "payment_terms_days": "int | null",
                    "termination_for_convenience": "bool",
                    "termination_notice_days": "int | null",
                    "security_incident_notice_hours": "int | null",
                }
                gold = _contract_gold()
                tokens = {"user": 500}
            rows.append(
                {
                    "id": f"{family}-{index:04d}",
                    "use_case": family,
                    "split": "private",
                    "prompt": {"system": system, "user": f"Synthetic case {index}"},
                    "output_schema": schema,
                    "gold": gold,
                    "aliases": [],
                    "tokens": tokens,
                }
            )
        (dataset_root / f"{family}.jsonl").write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )

    manifest = {
        "status": "frozen",
        "freeze_ready": True,
        "use_cases": {family: {"private_items": count} for family, count in counts.items()},
    }
    (dataset_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    spec_path = root / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return load_config(spec_path)

