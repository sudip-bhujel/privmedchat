"""Dataset builder package."""

__all__ = [
    "clean_text",
    "deduplicate_examples",
    "detect_format",
    "filter_by_judge",
    "format_dialogue",
    "group_key",
    "group_split",
    "load_dialogues",
    "generate_rejected_responses",
    "filter_by_similarity",
    "split_data",
    "to_pair_texts",
    "to_prompt",
    "to_sft_text",
    "write_jsonl",
]


def __getattr__(name: str):
    if name in {
        "detect_format",
        "group_key",
        "group_split",
        "to_pair_texts",
        "to_prompt",
        "to_sft_text",
    }:
        from dataset_builder import io_adapters as _adapters

        return getattr(_adapters, name)
    if name in __all__:
        from dataset_builder import generate as _generate

        return getattr(_generate, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
