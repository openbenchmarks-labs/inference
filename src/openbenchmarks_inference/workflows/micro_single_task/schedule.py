"""Deterministic selection, paired rounds, and rotating launch order."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import random
from typing import Any, Iterable

from .config import BenchmarkConfig
from .dataset import DatasetItem, DatasetRelease


SCHEDULE_VERSION = "micro-single-task-paired-round-robin-sha256-v1"


def selection_seed(slug: str, release_hash: str, family: str) -> int:
    material = f"{slug}\0{release_hash}\0{family}\0{SCHEDULE_VERSION}".encode()
    return int.from_bytes(hashlib.sha256(material).digest(), "big")


def family_order(config: BenchmarkConfig, dataset: DatasetRelease, family: str) -> tuple[str, ...]:
    ids = sorted(item.id for item in dataset.by_family[family])
    random.Random(selection_seed(config.benchmark_slug, dataset.release_hash, family)).shuffle(ids)
    return tuple(ids)


def selection_artifact(config: BenchmarkConfig, dataset: DatasetRelease) -> dict[str, Any]:
    families: dict[str, Any] = {}
    for family in config.task_families:
        order = family_order(config, dataset, family)
        item_map = {item.id: item for item in dataset.by_family[family]}
        selected = order[: config.selected_per_family]
        families[family] = {
            "seed": str(selection_seed(config.benchmark_slug, dataset.release_hash, family)),
            "full_family_order": list(order),
            "selected_prefix": list(selected),
            "selected_item_hashes": {item_id: item_map[item_id].sha256 for item_id in selected},
        }
    return {
        "algorithm_version": SCHEDULE_VERSION,
        "benchmark_slug": config.benchmark_slug,
        "dataset_release_hash": dataset.release_hash,
        "selected_items": config.selected_per_family * len(config.task_families),
        "families": families,
    }


@dataclass(frozen=True)
class WorkUnit:
    request_id: str
    round_id: str
    round_index: int
    provider: str
    provider_launch_index: int
    task_family: str
    item_id: str
    smoke_attempt: int | None = None

    @property
    def index(self) -> int:
        return self.round_index

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QuestionRound:
    round_id: str
    round_index: int
    task_family: str
    item_id: str
    units: tuple[WorkUnit, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "round_index": self.round_index,
            "task_family": self.task_family,
            "item_id": self.item_id,
            "provider_launch_order": [unit.provider for unit in self.units],
            "requests": [unit.to_dict() for unit in self.units],
        }


def _identity(kind: str, *parts: object) -> str:
    return kind + "-" + hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()[:24]


def rotated_providers(providers: tuple[str, ...], round_index: int) -> tuple[str, ...]:
    offset = round_index % len(providers)
    return providers[offset:] + providers[:offset]


def _build(config: BenchmarkConfig, dataset: DatasetRelease, *, smoke_attempt: int | None) -> tuple[QuestionRound, ...]:
    selected = selection_artifact(config, dataset)["families"]
    per_family = 1 if smoke_attempt is not None else config.selected_per_family
    rounds: list[QuestionRound] = []
    for item_index in range(per_family):
        for family in config.task_families:
            round_index = len(rounds)
            item_id = selected[family]["selected_prefix"][item_index]
            attempt_identity = smoke_attempt or 0
            round_id = _identity("round", SCHEDULE_VERSION, round_index, family, item_id, attempt_identity)
            providers = rotated_providers(config.providers, round_index)
            units = tuple(
                WorkUnit(
                    _identity("req", SCHEDULE_VERSION, provider, round_index, family, item_id, attempt_identity),
                    round_id, round_index, provider, launch_index, family, item_id, smoke_attempt,
                )
                for launch_index, provider in enumerate(providers)
            )
            rounds.append(QuestionRound(round_id, round_index, family, item_id, units))
    expected = 3 if smoke_attempt is not None else config.full_rounds
    if len(rounds) != expected or any(len(round_.units) != len(config.providers) for round_ in rounds):
        raise RuntimeError("resolved schedule differs from the declared execution shape")
    return tuple(rounds)


def build_smoke_schedule(config: BenchmarkConfig, dataset: DatasetRelease, attempt: int) -> tuple[QuestionRound, ...]:
    if attempt not in range(1, config.smoke_maximum_attempts + 1):
        raise ValueError("smoke attempt must be 1 or 2")
    return _build(config, dataset, smoke_attempt=attempt)


def build_full_schedule(config: BenchmarkConfig, dataset: DatasetRelease) -> tuple[QuestionRound, ...]:
    return _build(config, dataset, smoke_attempt=None)


def flatten(rounds: Iterable[QuestionRound]) -> tuple[WorkUnit, ...]:
    return tuple(unit for round_ in rounds for unit in round_.units)
