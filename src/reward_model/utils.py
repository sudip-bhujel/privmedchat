"""Utility helpers for reward model training and checkpoint IO."""

from __future__ import annotations

import os

import torch
from peft import PeftModel


def determine_device():
    """Determine default available device."""
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def print_trainable_parameters(model) -> None:
    """Print trainable parameter count."""
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    pct = 100 * trainable_params / all_params if all_params > 0 else 0.0
    print(
        f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {pct:.4f}%"
    )


def save_reward_model(model, tokenizer, output_dir: str) -> None:
    """Save reward model adapter and score head."""
    os.makedirs(output_dir, exist_ok=True)

    unwrapped = model._module if hasattr(model, "_module") else model
    unwrapped.backbone.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(
        unwrapped.score_head.state_dict(), os.path.join(output_dir, "score_head.pt")
    )
    print(f"Model and tokenizer saved to {output_dir}")


def load_reward_model(model, tokenizer, input_dir: str) -> None:
    """Load reward model PEFT adapter and score head.

    The checkpoint is expected to contain PEFT adapter files
    (adapter_config.json + adapter_model.safetensors) produced by
    ``backbone.save_pretrained()`` and a ``score_head.pt`` state-dict.
    """
    unwrapped = model._module if hasattr(model, "_module") else model

    # Load PEFT adapter onto the backbone.  PeftModel.from_pretrained
    # returns a **new** PeftModel wrapping the base model, so we must
    # reassign.
    unwrapped.backbone = PeftModel.from_pretrained(unwrapped.backbone, input_dir)
    print(f"PEFT adapter loaded from {input_dir}")

    score_head_path = os.path.join(input_dir, "score_head.pt")
    if os.path.exists(score_head_path):
        unwrapped.score_head.load_state_dict(
            torch.load(score_head_path, map_location="cpu")
        )
        print(f"Score head loaded from {score_head_path}")
    else:
        print(
            f"Warning: score_head.pt not found in {input_dir}. Score head not loaded."
        )
