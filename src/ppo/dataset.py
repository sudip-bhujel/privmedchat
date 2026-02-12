"""PPO dataset helpers."""

from __future__ import annotations

import torch
from dataset_builder.io_adapters import to_prompt
from torch.utils.data import Dataset
from tqdm import tqdm


class PPODataset(Dataset):
    """Simple in-memory rollout dataset."""

    def __init__(self, rollouts):
        self.rollouts = rollouts

    def __len__(self):
        return len(self.rollouts)

    def __getitem__(self, idx):
        return self.rollouts[idx]


def ppo_collate_fn(batch, pad_token_id: int = 0):
    """Pad variable-length rollout tensors."""
    max_len = max(item["input_ids"].size(0) for item in batch)

    padded = {
        "input_ids": [],
        "attention_mask": [],
        "action_mask": [],
        "old_log_probs": [],
        "ref_log_probs": [],
        "rewards": [],
    }

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len

        padded["input_ids"].append(
            torch.nn.functional.pad(item["input_ids"], (0, pad_len), value=pad_token_id)
        )
        padded["attention_mask"].append(
            torch.nn.functional.pad(item["attention_mask"], (0, pad_len), value=0)
        )
        padded["action_mask"].append(
            torch.nn.functional.pad(item["action_mask"], (0, pad_len), value=0)
        )
        padded["old_log_probs"].append(
            torch.nn.functional.pad(item["old_log_probs"], (0, pad_len), value=0.0)
        )
        padded["ref_log_probs"].append(
            torch.nn.functional.pad(item["ref_log_probs"], (0, pad_len), value=0.0)
        )
        padded["rewards"].append(item["rewards"])

    return {
        "input_ids": torch.stack(padded["input_ids"]),
        "attention_mask": torch.stack(padded["attention_mask"]),
        "action_mask": torch.stack(padded["action_mask"]),
        "old_log_probs": torch.stack(padded["old_log_probs"]),
        "ref_log_probs": torch.stack(padded["ref_log_probs"]),
        "rewards": torch.tensor(padded["rewards"]),
    }


def extract_prompts_from_dataset(dataset) -> list[str]:
    """Extract unique patient prompts from preference-pair dataset."""
    unique_prompts: dict[str, None] = {}

    for example in tqdm(dataset, desc="Extract prompts"):
        try:
            prompt = to_prompt(dict(example))
        except ValueError:
            continue
        unique_prompts[prompt] = None

    return list(unique_prompts.keys())
