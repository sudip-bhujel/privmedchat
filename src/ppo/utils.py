"""PPO utility functions for loss, checkpointing, and rollout generation."""

from __future__ import annotations

import gc
import os
from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from ppo.model import ActorCritic


def ppo_loss_fn(
    log_probs: torch.Tensor,
    values: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    action_mask: torch.Tensor,
    clip_epsilon: float = 0.2,
    beta_kl: float = 0.05,
) -> torch.Tensor:
    """Compute PPO objective with KL penalty and value loss."""
    kl_penalty = log_probs - ref_log_probs
    combined_rewards = rewards.unsqueeze(1) - beta_kl * kl_penalty

    advantages = combined_rewards - values.detach()

    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2)

    value_loss = 0.5 * (values - combined_rewards).pow(2)

    valid_tokens = action_mask.sum()
    assert valid_tokens > 0, "No valid tokens found in action_mask."

    policy_loss = (policy_loss * action_mask).sum() / valid_tokens
    value_loss = (value_loss * action_mask).sum() / valid_tokens
    return policy_loss + value_loss


def save_ppo_model(model, tokenizer, output_dir: str):
    """Save actor model and tokenizer."""
    os.makedirs(output_dir, exist_ok=True)

    actor_to_save = model._module if hasattr(model, "_module") else model
    actor_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def clean_memory() -> None:
    """Best-effort memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@torch.no_grad()
def collect_rollouts(
    model: ActorCritic,
    tokenizer,
    prompts,
    batch_size: int,
    max_new_tokens: int,
    device,
):
    """Collect policy rollouts with reward-model scoring."""
    model.eval()
    rollouts = []

    system_prompt = (
        "You are a careful medical assistant. Provide safe, concise, non-diagnostic guidance "
        "and recommend professional care when needed."
    )

    for i in tqdm(range(0, len(prompts), batch_size), desc="Rollouts"):
        batch_prompts = prompts[i : i + batch_size]

        formatted_prompts = []
        for user_content in batch_prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted_prompts.append(text)

        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=min(tokenizer.model_max_length, 8192),
        ).to(device)
        prompt_len = inputs.input_ids.shape[1]

        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id,
        )

        rm_scores = model.get_reward_score(
            input_ids=outputs, attention_mask=torch.ones_like(outputs)
        )
        ref_token_log_probs = model.get_ref_token_log_probs(outputs)
        old_token_log_probs = model.get_actor_token_log_probs(outputs)

        action_mask = torch.zeros_like(outputs, device=outputs.device)
        action_mask[:, prompt_len:] = 1

        if tokenizer.pad_token_id is not None:
            action_mask[outputs == tokenizer.pad_token_id] = 0

        for j in range(len(batch_prompts)):
            rollouts.append(
                {
                    "input_ids": outputs[j].cpu(),
                    "attention_mask": torch.ones_like(outputs[j]).cpu(),
                    "action_mask": action_mask[j].cpu(),
                    "old_log_probs": old_token_log_probs[j].cpu(),
                    "ref_log_probs": ref_token_log_probs[j].cpu(),
                    "rewards": rm_scores[j].item(),
                }
            )

        del (
            inputs,
            outputs,
            rm_scores,
            ref_token_log_probs,
            old_token_log_probs,
            action_mask,
        )
        clean_memory()

    return rollouts
