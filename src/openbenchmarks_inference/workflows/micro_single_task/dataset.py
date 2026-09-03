"""Strict loader for a frozen benchmark dataset and referenced prompts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any

from .config import BenchmarkConfig


CONTRACT_FIELDS = (
    "agreement_type", "effective_date", "initial_term_end_date", "auto_renews",
    "renewal_term_months", "non_renewal_notice_days", "payment_terms_days",
    "termination_for_convenience", "termination_notice_days",
    "security_incident_notice_hours",
)


@dataclass(frozen=True)
class DatasetItem(Mapping[str, Any]):
    id: str
    task_family: str
    prompt: dict[str, str]
    output_schema: dict[str, str]
    gold: dict[str, Any]
    aliases: Any
    tokens: dict[str, Any]
    raw: dict[str, Any]
    sha256: str

    @property
    def messages(self) -> tuple[dict[str, str], dict[str, str]]:
        return ({"role": "system", "content": self.prompt["system"]}, {"role": "user", "content": self.prompt["user"]})

    def __getitem__(self, key: str) -> Any:
        if key == "messages":
            return self.messages
        return self.raw[key]

    def __iter__(self) -> Iterator[str]:
        return iter((*self.raw, "messages"))

    def __len__(self) -> int:
        return len(self.raw) + 1


@dataclass(frozen=True)
class DatasetRelease:
    by_family: dict[str, tuple[DatasetItem, ...]]
    file_hashes: dict[str, str]
    release_hash: str
    manifest: dict[str, Any]

    @property
    def total_items(self) -> int:
        return sum(len(rows) for rows in self.by_family.values())

    @property
    def hashes(self) -> dict[str, str]:
        return self.file_hashes


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _validate_gold(family: str, gold: Any, aliases: Any) -> None:
    if not isinstance(gold, dict):
        raise ValueError(f"{family}: gold must be an object")
    if family == "meeting-notes-lookup":
        if set(gold) != {"type", "value"} or not isinstance(gold["value"], str):
            raise ValueError("meeting gold must contain exact type/value")
        if not isinstance(aliases, list) or not all(isinstance(v, str) for v in aliases):
            raise ValueError("meeting aliases must be strings")
    elif family == "ticket-triage":
        if set(gold) != {"category", "priority"}:
            raise ValueError("ticket gold keys mismatch")
        if gold["category"] not in {"bug", "feature", "improvement", "customer_support"} or gold["priority"] not in {"p0", "p1", "p2", "p3"}:
            raise ValueError("ticket gold enum mismatch")
    else:
        if tuple(gold) != CONTRACT_FIELDS:
            raise ValueError("contract gold field order/schema mismatch")
        if gold["agreement_type"] not in {"saas_subscription", "software_license", "support_maintenance", "technology_services"}:
            raise ValueError("contract agreement_type mismatch")
        for field in ("effective_date", "initial_term_end_date"):
            if not isinstance(gold[field], str) or date.fromisoformat(gold[field]).isoformat() != gold[field]:
                raise ValueError(f"contract {field} is not ISO YYYY-MM-DD")
        for field in ("auto_renews", "termination_for_convenience"):
            if not isinstance(gold[field], bool):
                raise ValueError(f"contract {field} must be boolean")
        for field in CONTRACT_FIELDS[4:7] + CONTRACT_FIELDS[8:]:
            value = gold[field]
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise ValueError(f"contract {field} must be integer or null")


def load_dataset(config: BenchmarkConfig) -> DatasetRelease:
    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("status") != "frozen" or manifest.get("freeze_ready") is not True:
        raise ValueError("dataset manifest must be frozen and freeze-ready")
    use_cases = manifest.get("use_cases")
    if not isinstance(use_cases, dict) or set(use_cases) != set(config.task_families):
        raise ValueError("dataset manifest task families differ from the spec")

    hashes = {_relative(config.manifest_path): _file_hash(config.manifest_path)}
    by_family: dict[str, tuple[DatasetItem, ...]] = {}
    seen: set[str] = set()
    for family in config.task_families:
        family_spec = config.raw["dataset"]["task_families"][family]
        expected = int(family_spec["available_items"])
        manifest_row = use_cases[family]
        if manifest_row.get("private_items") != expected:
            raise ValueError(f"{family}: manifest private count differs from spec")
        template_path = (config.spec_path.parent / family_spec["prompt_template"]).resolve()
        template = template_path.read_text(encoding="utf-8").removesuffix("\n")
        hashes[_relative(template_path)] = _file_hash(template_path)
        path = config.dataset_root / f"{family}.jsonl"
        hashes[_relative(path)] = _file_hash(path)
        rows: list[DatasetItem] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
                required = {"id", "use_case", "split", "prompt", "output_schema", "gold", "aliases", "tokens"}
                if not isinstance(row, dict) or not required <= row.keys():
                    raise ValueError(f"{path}:{line_number}: missing required fields")
                item_id = row["id"]
                if not isinstance(item_id, str) or not item_id or item_id in seen:
                    raise ValueError(f"{path}:{line_number}: duplicate or invalid item ID")
                if row["use_case"] != family or row["split"] != "private":
                    raise ValueError(f"{path}:{line_number}: family/split mismatch")
                prompt = row["prompt"]
                if not isinstance(prompt, dict) or set(prompt) != {"system", "user"} or prompt["system"] != template or not prompt["user"]:
                    raise ValueError(f"{path}:{line_number}: prompt differs from frozen template")
                expected_schema = family_spec["output_schema"]
                normalized_schema = {
                    key: {"string": "string", "boolean": "bool", "number | null": "int | null"}[value]
                    for key, value in expected_schema.items()
                }
                if family == "ticket-triage":
                    normalized_schema = {"category": "enum", "priority": "enum"}
                if family == "contract-terms-extraction":
                    normalized_schema["agreement_type"] = "enum"
                if row["output_schema"] != normalized_schema:
                    raise ValueError(f"{path}:{line_number}: output schema mismatch")
                _validate_gold(family, row["gold"], row["aliases"])
                low, high = (int(v) for v in str(family_spec["input_tokens"]).split("-"))
                token_field = "input_total" if family == "meeting-notes-lookup" else "user"
                count = row["tokens"].get(token_field) if isinstance(row["tokens"], dict) else None
                if isinstance(count, bool) or not isinstance(count, int) or not low <= count <= high:
                    raise ValueError(f"{path}:{line_number}: input token band mismatch")
                seen.add(item_id)
                rows.append(DatasetItem(item_id, family, prompt, row["output_schema"], row["gold"], row["aliases"], row["tokens"], row, _canonical_hash(row)))
        if len(rows) != expected:
            raise ValueError(f"{family}: expected {expected} private items, found {len(rows)}")
        by_family[family] = tuple(rows)
    if len(seen) != int(config.raw["dataset"]["available_items"]):
        raise ValueError("dataset total differs from spec")
    release_hash = hashlib.sha256("\n".join(f"{key}:{hashes[key]}" for key in sorted(hashes)).encode()).hexdigest()
    return DatasetRelease(by_family, hashes, release_hash, manifest)
