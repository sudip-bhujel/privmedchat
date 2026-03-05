"""De-identify medical dialogue data downloaded from Hugging Face.

The script loads a dataset split from Hugging Face, extracts patient/doctor
pairs, applies Presidio-based de-identification, and writes de-identified
train/val/test JSONL files under ``data/`` by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from tqdm import tqdm

log = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "UCSD26/medical_dialog"

FALLBACK_DATASETS = [
    ("UCSD26/medical_dialog", None),
    ("OpenMed/MedDialog", None),
    ("ruslanmv/ai-medical-chatbot", None),
    ("lighteval/med_dialog", "healthcaremagic"),
    ("lighteval/med_dialog", "icliniq"),
]

DEFAULT_ENTITIES = (
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "DATE_TIME",
    "LOCATION",
    "IP_ADDRESS",
    "US_SSN",
    "US_DRIVER_LICENSE",
    "US_PASSPORT",
    "MEDICAL_LICENSE",
)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """Normalize whitespace and strip lightweight artifacts."""
    text = (text or "").replace("\u200b", "").replace("\ufeff", "")
    return " ".join(text.split()).strip()


def format_dialogue(patient_text: str, doctor_text: str | None = None) -> str:
    """Format a single patient-doctor exchange."""
    patient_text = clean_text(patient_text)
    doctor_text = clean_text(doctor_text or "")
    if doctor_text:
        return f"[Patient]: {patient_text}\n[Doctor]: {doctor_text}"
    return f"[Patient]: {patient_text}\n[Doctor]:"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_dataset(
    name: str | None, subset: str | None
) -> tuple[Dataset, str, str | None]:
    """Load a Hugging Face dataset, falling back across known alternatives.

    `datasets>=4` does not support legacy script-based datasets. If the
    requested dataset fails (including script-based failures), we try a small
    ordered fallback list before failing.
    """
    candidates: list[tuple[str, str | None]] = []
    if name:
        candidates.append((name, subset))
    for fallback_name, fallback_subset in FALLBACK_DATASETS:
        if (fallback_name, fallback_subset) not in candidates:
            candidates.append((fallback_name, fallback_subset))

    errors: list[str] = []
    for dataset_name, dataset_subset in candidates:
        try:
            if dataset_subset:
                ds = load_dataset(dataset_name, dataset_subset, split="train")  # type: ignore[arg-type]
            else:
                ds = load_dataset(dataset_name, split="train")  # type: ignore[arg-type]

            if name and (dataset_name, dataset_subset) != (name, subset):
                print(
                    "WARNING: Requested dataset could not be loaded. "
                    f"Falling back to {dataset_name}"
                    f"{'/' + dataset_subset if dataset_subset else ''}."
                )
            return ds, dataset_name, dataset_subset
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).strip().splitlines()[0]
            if "Dataset scripts are no longer supported" in str(exc):
                msg = f"{msg} (script-based dataset blocked by installed `datasets` version)"
            errors.append(
                f"- {dataset_name}"
                f"{'/' + dataset_subset if dataset_subset else ''}: "
                f"{type(exc).__name__}: {msg}"
            )
            log.warning(
                "Failed loading dataset %s subset %s: %s",
                dataset_name,
                dataset_subset,
                msg,
            )

    raise RuntimeError(
        "Unable to load a supported medical dialogue dataset. Tried:\n"
        + "\n".join(errors)
    )


def _extract_pairs(record: dict[str, Any]) -> list[tuple[str, str]]:
    """Extract (patient, doctor) pairs from a single dataset record."""
    keys = {k.lower(): k for k in record}

    question_keys = {
        "query",
        "question",
        "patient",
        "patient_message",
        "input",
        "instruction",
        "ask",
        "symptoms",
        "user",
    }
    answer_keys = {
        "response",
        "answer",
        "doctor",
        "doctor_response",
        "output",
        "assistant",
        "reply",
    }

    q_key = next((keys[k] for k in keys if k in question_keys), None)
    a_key = next((keys[k] for k in keys if k in answer_keys), None)
    if q_key and a_key:
        patient = clean_text(str(record.get(q_key, "")))
        doctor = clean_text(str(record.get(a_key, "")))
        if patient and doctor:
            return [(patient, doctor)]

    dialogue_keys = {
        "dialog",
        "dialogue",
        "conversation",
        "utterances",
        "messages",
        "chat",
    }
    conv_key = next((keys[k] for k in keys if k in dialogue_keys), None)
    if conv_key:
        conv = record.get(conv_key)
        if not isinstance(conv, list) or not conv:
            return []

        # Role-annotated turns: [{"role": "...", "text": "..."}]
        if isinstance(conv[0], dict):
            pairs: list[tuple[str, str]] = []
            pending_patient: str | None = None
            for turn in conv:
                role = clean_text(
                    str(
                        turn.get("role")
                        or turn.get("speaker")
                        or turn.get("from")
                        or ""
                    )
                ).lower()
                text = clean_text(
                    str(
                        turn.get("text")
                        or turn.get("utterance")
                        or turn.get("content")
                        or turn.get("value")
                        or ""
                    )
                )
                if not text:
                    continue
                if role in {"patient", "user", "questioner", "human"}:
                    pending_patient = text
                elif (
                    role in {"doctor", "assistant", "answerer", "agent", "gpt"}
                    and pending_patient
                ):
                    pairs.append((pending_patient, text))
                    pending_patient = None
            return pairs

        # Alternating list format: patient at even indices, doctor at odd.
        pairs = []
        for i in range(0, len(conv) - 1, 2):
            patient = clean_text(str(conv[i]))
            doctor = clean_text(str(conv[i + 1]))
            if patient and doctor:
                pairs.append((patient, doctor))
        return pairs

    # Single text field with [Patient]/[Doctor] markers.
    text_key = next((keys[k] for k in keys if k in {"text", "dialogue_text"}), None)
    if text_key:
        text = clean_text(str(record.get(text_key, "")))
        if "[Patient]:" in text and "[Doctor]:" in text:
            patient = clean_text(
                text.split("[Doctor]:", 1)[0].replace("[Patient]:", "")
            )
            doctor = clean_text(text.split("[Doctor]:", 1)[1])
            if patient and doctor:
                return [(patient, doctor)]

    return []


def load_dialogues(
    dataset_name: str | None,
    dataset_subset: str | None,
    max_samples: int | None,
) -> tuple[list[dict[str, Any]], str, str | None]:
    """Load and normalize rows to patient/doctor format."""
    ds, chosen_name, chosen_subset = _load_dataset(dataset_name, dataset_subset)

    examples: list[dict[str, Any]] = []

    for idx, record in enumerate(ds):
        conv_id = str(record.get("conversation_id") or record.get("id") or idx)  # type: ignore[union-attr]
        for pair_idx, (patient, doctor) in enumerate(_extract_pairs(record)):  # type: ignore[assignment]
            examples.append(
                {
                    "conversation_id": f"{conv_id}_{pair_idx}",
                    "source_conversation_id": conv_id,
                    "patient": patient,
                    "doctor": doctor,
                }
            )
            if max_samples and len(examples) >= max_samples:
                break
        if max_samples and len(examples) >= max_samples:
            break

    return examples, chosen_name, chosen_subset


def deduplicate_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop exact duplicate patient/doctor pairs while preserving order."""
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for ex in examples:
        key = (
            clean_text(str(ex.get("patient", ""))).casefold(),
            clean_text(str(ex.get("doctor", ""))).casefold(),
        )
        if not key[0] or not key[1] or key in seen:
            continue
        seen.add(key)
        deduped.append(ex)
    return deduped


def split_data(
    examples: list[dict[str, Any]], seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Group-aware 80/10/10 split by source conversation id."""
    if not examples:
        return [], [], []

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        group_id = clean_text(str(ex.get("source_conversation_id", ""))) or clean_text(
            str(ex.get("conversation_id", ""))
        )
        grouped[group_id].append(ex)

    group_ids = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    n_groups = len(group_ids)
    n_train = int(0.8 * n_groups)
    n_val = int(0.1 * n_groups)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []

    for gid in group_ids[:n_train]:
        train.extend(grouped[gid])
    for gid in group_ids[n_train : n_train + n_val]:
        val.extend(grouped[gid])
    for gid in group_ids[n_train + n_val :]:
        test.extend(grouped[gid])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


# ---------------------------------------------------------------------------
# Presidio de-identification
# ---------------------------------------------------------------------------


def build_analyzer(nlp_model: str) -> AnalyzerEngine:
    """Create Presidio analyzer with a spaCy backend model."""
    nlp_configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": nlp_model}],
        "ner_model_configuration": {
            "labels_to_ignore": ["CARDINAL", "ORDINAL", "QUANTITY", "PRODUCT"]
        },
    }
    provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
    nlp_engine = provider.create_engine()
    return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])


class PresidioDeidentifier:
    """Presidio wrapper emitting deterministic placeholders per entity type."""

    def __init__(
        self,
        *,
        analyzer: AnalyzerEngine,
        entities: list[str],
        language: str = "en",
        score_threshold: float = 0.45,
    ) -> None:
        self.analyzer = analyzer
        self.anonymizer = AnonymizerEngine()
        self.entities = entities
        self.language = language
        self.score_threshold = score_threshold
        self.entity_counters: Counter[str] = Counter()

    def _next_placeholder(
        self,
        entity_type: str,
        span_value: str,
        row_cache: dict[tuple[str, str], str],
    ) -> str:
        key = (entity_type, clean_text(span_value).casefold())
        if key in row_cache:
            return row_cache[key]
        self.entity_counters[entity_type] += 1
        placeholder = f"<{entity_type}_{self.entity_counters[entity_type]:06d}>"
        row_cache[key] = placeholder
        return placeholder

    def deidentify_text(
        self, text: str, *, row_cache: dict[tuple[str, str], str]
    ) -> tuple[str, Counter[str]]:
        """Return de-identified text plus counts of applied entity replacements."""
        if not text:
            return text, Counter()

        results = self.analyzer.analyze(
            text=text,
            language=self.language,
            entities=self.entities,
            score_threshold=self.score_threshold,
        )
        if not results:
            return text, Counter()

        custom_results: list[RecognizerResult] = []
        operators: dict[str, OperatorConfig] = {}
        alias_to_entity: dict[str, str] = {}

        for idx, result in enumerate(results):
            span = text[result.start : result.end]
            if not clean_text(span):
                continue
            placeholder = self._next_placeholder(result.entity_type, span, row_cache)
            alias = f"{result.entity_type}__{idx}"
            alias_to_entity[alias] = result.entity_type
            custom_results.append(
                RecognizerResult(
                    entity_type=alias,
                    start=result.start,
                    end=result.end,
                    score=result.score,
                )
            )
            operators[alias] = OperatorConfig("replace", {"new_value": placeholder})

        if not custom_results:
            return text, Counter()

        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=custom_results,  # type: ignore[list-item]
            operators=operators,
        )

        applied: Counter[str] = Counter()
        for item in anonymized.items:
            ent = alias_to_entity.get(getattr(item, "entity_type", ""))
            if ent:
                applied[ent] += 1
        return anonymized.text, applied


# ---------------------------------------------------------------------------
# De-identification pipeline
# ---------------------------------------------------------------------------


@dataclass
class SplitStats:
    rows: int = 0
    changed_rows: int = 0
    changed_fields: int = 0
    entity_counts: Counter[str] = field(default_factory=Counter)


def _deidentify_str_field(
    row: dict[str, Any],
    key: str,
    deidentifier: PresidioDeidentifier,
    row_cache: dict[tuple[str, str], str],
    stats: SplitStats,
) -> None:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        return
    deid_value, entity_hits = deidentifier.deidentify_text(value, row_cache=row_cache)
    if deid_value != value:
        row[key] = deid_value
        stats.changed_fields += 1
    if entity_hits:
        stats.entity_counts.update(entity_hits)


def deidentify_examples(
    examples: list[dict[str, Any]],
    deidentifier: PresidioDeidentifier,
    *,
    split_name: str,
) -> tuple[list[dict[str, Any]], SplitStats]:
    """De-identify a list of normalized dialogue examples."""
    stats = SplitStats()
    out_rows: list[dict[str, Any]] = []

    for row in tqdm(examples, desc=f"De-identifying {split_name}", unit="row"):
        stats.rows += 1
        out = dict(row)
        row_cache: dict[tuple[str, str], str] = {}

        _deidentify_str_field(out, "patient", deidentifier, row_cache, stats)
        _deidentify_str_field(out, "doctor", deidentifier, row_cache, stats)

        patient = out.get("patient")
        doctor = out.get("doctor")
        if isinstance(patient, str) and isinstance(doctor, str) and patient and doctor:
            prompt = format_dialogue(patient)
            text = format_dialogue(patient, doctor)
            if out.get("prompt") != prompt:
                out["prompt"] = prompt
                stats.changed_fields += 1
            if out.get("text") != text:
                out["text"] = text
                stats.changed_fields += 1

        if out != row:
            stats.changed_rows += 1
        out_rows.append(out)

    return out_rows, stats


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_entities(raw: str) -> list[str]:
    entities = [e.strip() for e in raw.split(",") if e.strip()]
    if not entities:
        raise ValueError("No entities were provided.")
    return entities


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download, de-identify, and split medical dialogue data."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="Hugging Face dataset id.",
    )
    parser.add_argument(
        "--dataset_subset",
        type=str,
        default=None,
        help="Optional Hugging Face dataset subset/config name.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Directory to save de-identified split JSONL files.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="meddialog",
        help="Prefix for output filenames.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional max number of normalized dialogue examples.",
    )
    parser.add_argument(
        "--no_dedupe_exact",
        action="store_true",
        help="Disable exact patient+doctor deduplication before split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--entities",
        type=str,
        default=",".join(DEFAULT_ENTITIES),
        help="Comma-separated Presidio entity types to anonymize.",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.45,
        help="Minimum analyzer confidence score for replacement.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code for Presidio analyzer.",
    )
    parser.add_argument(
        "--nlp_model",
        type=str,
        default="en_core_web_lg",
        help="spaCy model name used by Presidio.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if not args.verbose:
        logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
        logging.getLogger("presidio-anonymizer").setLevel(logging.ERROR)

    output_dir = Path(args.output_dir)
    entities = parse_entities(args.entities)
    os.environ.setdefault(
        "TLDEXTRACT_CACHE", str((output_dir / ".tldextract").resolve())
    )

    print("=" * 60)
    print("Step 1: Initializing Presidio")
    print("=" * 60)
    print(f"NLP model: {args.nlp_model}")
    print(f"Entities: {', '.join(entities)}")
    print(f"Score threshold: {args.score_threshold:.2f}")
    analyzer = build_analyzer(args.nlp_model)
    deidentifier = PresidioDeidentifier(
        analyzer=analyzer,
        entities=entities,
        language=args.language,
        score_threshold=args.score_threshold,
    )

    print("\n" + "=" * 60)
    print("Step 2: Loading dataset from Hugging Face")
    print("=" * 60)
    print(f"Dataset: {args.dataset_name}")
    print(f"Subset:  {args.dataset_subset if args.dataset_subset else 'None'}")
    examples, loaded_name, loaded_subset = load_dialogues(
        args.dataset_name, args.dataset_subset, args.max_samples
    )
    print(f"Loaded source: {loaded_name}{'/' + loaded_subset if loaded_subset else ''}")
    print(f"Loaded normalized examples: {len(examples)}")
    if not examples:
        raise RuntimeError(
            "No patient/doctor pairs could be extracted from the dataset."
        )

    if not args.no_dedupe_exact:
        before = len(examples)
        examples = deduplicate_examples(examples)
        print(
            f"Deduplicated exact pairs: removed {before - len(examples)}, kept {len(examples)}"
        )

    print("\n" + "=" * 60)
    print("Step 3: Splitting data (80/10/10)")
    print("=" * 60)
    train, val, test = split_data(examples, seed=args.seed)
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

    print("\n" + "=" * 60)
    print("Step 4: De-identifying splits")
    print("=" * 60)
    train_deid, train_stats = deidentify_examples(
        train, deidentifier, split_name="train"
    )
    val_deid, val_stats = deidentify_examples(val, deidentifier, split_name="val")
    test_deid, test_stats = deidentify_examples(test, deidentifier, split_name="test")

    print("\n" + "=" * 60)
    print("Step 5: Writing outputs")
    print("=" * 60)
    train_out = output_dir / f"{args.output_prefix}_train_deidentified.jsonl"
    val_out = output_dir / f"{args.output_prefix}_val_deidentified.jsonl"
    test_out = output_dir / f"{args.output_prefix}_test_deidentified.jsonl"
    write_jsonl(train_out, train_deid)
    write_jsonl(val_out, val_deid)
    write_jsonl(test_out, test_deid)
    print(f"Wrote: {train_out}")
    print(f"Wrote: {val_out}")
    print(f"Wrote: {test_out}")

    total_stats = SplitStats()
    for part in (train_stats, val_stats, test_stats):
        total_stats.rows += part.rows
        total_stats.changed_rows += part.changed_rows
        total_stats.changed_fields += part.changed_fields
        total_stats.entity_counts.update(part.entity_counts)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"Rows processed: {total_stats.rows}")
    print(f"Rows changed:   {total_stats.changed_rows}")
    print(f"Fields changed: {total_stats.changed_fields}")
    if total_stats.entity_counts:
        print("Entity replacements:")
        for entity, count in total_stats.entity_counts.most_common():
            print(f"  {entity}: {count}")
    else:
        print("Entity replacements: none")


if __name__ == "__main__":
    main()
