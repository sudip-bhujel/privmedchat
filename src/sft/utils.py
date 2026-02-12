"""Utility functions for SFT training."""

from __future__ import annotations

import os

import torch


def determine_device() -> torch.device:
    """Determine the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_trainable_parameters(model) -> None:
    """Print trainable parameter ratio."""
    trainable_params = 0
    all_params = 0

    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    ratio = 100 * trainable_params / all_params if all_params > 0 else 0.0
    print(
        f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {ratio:.4f}%"
    )


def save_sft_model(model, tokenizer, output_dir: str, epoch: int | None = None) -> str:
    """Save fine-tuned model and tokenizer."""
    save_dir = output_dir if epoch is None else os.path.join(output_dir, f"epoch_{epoch}")
    os.makedirs(save_dir, exist_ok=True)

    model_to_save = model._module if hasattr(model, "_module") else model
    model_to_save.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"Saved SFT checkpoint to {save_dir}")
    return save_dir
