"""Shared dataset I/O adapters for SFT, RM, and PPO consumers.

This module intentionally keeps `dataset_builder.generate` outputs unchanged and
provides normalization utilities for downstream trainers.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Mapping, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset

DatasetFormat = Literal["auto", "meddialog", "preference_pairs"]
DetectedFormat = Literal["meddialog", "preference_pair"]


def _clean_text(text: str | None) -> str:
    if text is None:
        return ""
    return str(text).replace("\u200b", "").replace("\ufeff", "").strip()


def _doctor_from_dialogue_text(text: str | None) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    if "[Doctor]:" in cleaned:
        return cleaned.split("[Doctor]:", 1)[1].strip()
    return cleaned


def _normalize_prompt(prompt: str | None) -> str:
    prompt_text = _clean_text(prompt)
    if not prompt_text:
        raise ValueError("Prompt is empty or missing.")

    if "[Doctor]:" in prompt_text:
        patient_prefix = prompt_text.split("[Doctor]:", 1)[0].rstrip()
    else:
        patient_prefix = prompt_text

    if patient_prefix.startswith("[Patient]:"):
        return f"{patient_prefix}\n[Doctor]:"
    return f"[Patient]: {patient_prefix}\n[Doctor]:"


def detect_format(row: Mapping[str, Any]) -> DetectedFormat:
    """Detect schema type for a single row."""
    if any(k in row for k in ("chosen", "rejected", "chosen_doctor", "rejected_doctor")):
        return "preference_pair"
    if any(k in row for k in ("text", "patient", "doctor", "prompt")):
        return "meddialog"
    raise ValueError(
        "Unable to detect row format. Expected MedDialog keys "
        "(`text`/`patient`/`doctor`) or preference keys (`chosen`/`rejected`)."
    )


def _resolve_format(row: Mapping[str, Any], input_format: DatasetFormat) -> DetectedFormat:
    if input_format == "auto":
        return detect_format(row)
    if input_format == "meddialog":
        return "meddialog"
    if input_format == "preference_pairs":
        return "preference_pair"
    raise ValueError(f"Unsupported input_format: {input_format!r}")


def to_prompt(row: Mapping[str, Any]) -> str:
    """Return a canonical prompt ending in `\\n[Doctor]:`."""
    prompt = _clean_text(row.get("prompt"))
    if prompt:
        return _normalize_prompt(prompt)

    patient = _clean_text(row.get("patient"))
    if patient:
        return f"[Patient]: {patient}\n[Doctor]:"

    for key in ("chosen", "text"):
        maybe_dialogue = _clean_text(row.get(key))
        if "[Doctor]:" in maybe_dialogue:
            patient_prefix = maybe_dialogue.split("[Doctor]:", 1)[0].rstrip()
            return _normalize_prompt(patient_prefix)

    raise ValueError(
        "Unable to derive prompt. Expected `prompt`, `patient`, or a dialogue text containing `[Doctor]:`."
    )


def to_sft_text(
    row: Mapping[str, Any],
    *,
    pair_source: str = "chosen",
    input_format: DatasetFormat = "auto",
) -> str:
    """Normalize a row into single SFT text: `[Patient]... [Doctor]...`."""
    if pair_source != "chosen":
        raise ValueError(
            f"Unsupported pair_sft_source={pair_source!r}. Only `chosen` is supported."
        )

    resolved = _resolve_format(row, input_format)
    if resolved == "meddialog":
        text = _clean_text(row.get("text"))
        if text:
            return text
        patient = _clean_text(row.get("patient"))
        doctor = _clean_text(row.get("doctor"))
        if not patient or not doctor:
            raise ValueError(
                "MedDialog row requires either `text` or both `patient` and `doctor`."
            )
        return f"[Patient]: {patient}\n[Doctor]: {doctor}"

    prompt = to_prompt(row)
    doctor = _clean_text(row.get("chosen_doctor"))
    if not doctor:
        doctor = _doctor_from_dialogue_text(row.get("chosen"))
    if not doctor:
        raise ValueError(
            "Preference-pair row requires `chosen_doctor` or `chosen` with `[Doctor]:` text."
        )
    return f"{prompt} {doctor}".strip()


def to_pair_texts(
    row: Mapping[str, Any], *, input_format: DatasetFormat = "auto"
) -> tuple[str, str]:
    """Normalize a row into `(chosen_text, rejected_text)` dialogues."""
    resolved = _resolve_format(row, input_format)
    if resolved != "preference_pair":
        raise ValueError(
            "Pair conversion expects preference-pair rows (`chosen`/`rejected`)."
        )

    prompt = to_prompt(row)

    chosen = _clean_text(row.get("chosen"))
    if not chosen:
        chosen_doctor = _clean_text(row.get("chosen_doctor"))
        if chosen_doctor:
            chosen = f"{prompt} {chosen_doctor}".strip()

    rejected = _clean_text(row.get("rejected"))
    if not rejected:
        rejected_doctor = _clean_text(row.get("rejected_doctor"))
        if rejected_doctor:
            rejected = f"{prompt} {rejected_doctor}".strip()

    if not chosen or not rejected:
        raise ValueError(
            "Preference-pair row missing chosen/rejected content. "
            "Expected `chosen`+`rejected` or doctor-only fallbacks."
        )
    return chosen, rejected


def group_key(row: Mapping[str, Any], id_column: str = "conversation_id") -> str:
    """Stable group id used for split leakage prevention."""
    source_id = _clean_text(row.get("source_conversation_id"))
    if source_id:
        return source_id

    conv_id = _clean_text(row.get(id_column))
    if not conv_id:
        return "unknown"

    if "_" in conv_id:
        base, suffix = conv_id.rsplit("_", 1)
        if suffix.isdigit() and base:
            return base
    return conv_id


def group_split(
    dataset: "Dataset",
    *,
    split: str | None = None,  # kept for caller compatibility
    id_column: str = "conversation_id",
    test_size: float = 0.1,
    seed: int = 42,
) -> dict[str, Dataset]:
    """Group-aware train/test split on HF datasets."""
    if len(dataset) == 0:
        return {"train": dataset, "test": dataset}

    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}.")

    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(dataset):
        grouped[group_key(row, id_column=id_column)].append(idx)

    group_ids = list(grouped.keys())
    if len(group_ids) < 2:
        return {"train": dataset, "test": dataset.select([])}

    rng = random.Random(seed)
    rng.shuffle(group_ids)

    n_test_groups = max(1, int(round(len(group_ids) * test_size)))
    n_test_groups = min(n_test_groups, len(group_ids) - 1)
    test_groups = set(group_ids[:n_test_groups])

    train_indices: list[int] = []
    test_indices: list[int] = []
    for gid, indices in grouped.items():
        if gid in test_groups:
            test_indices.extend(indices)
        else:
            train_indices.extend(indices)

    return {
        "train": dataset.select(train_indices),
        "test": dataset.select(test_indices),
    }
