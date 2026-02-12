"""Differential Privacy Actor-Critic model used by PPO stage."""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModel, AutoModelForCausalLM

from ppo.utils import clean_memory
from reward_model.model import MedRewardModel


class ActorCritic(nn.Module):
    """Actor-Critic with frozen reference and reward models."""

    def __init__(
        self,
        model_name: str,
        reward_model_path: str,
        actor_lora_config: LoraConfig,
        critic_lora_config: LoraConfig,
        a_device: str = "auto",
        c_device: str = "auto",
        ref_device: str = "auto",
        rm_device: str = "auto",
    ):
        super().__init__()

        self.a_device = a_device
        self.c_device = c_device
        self.ref_device = ref_device
        self.rm_device = rm_device

        self.actor_base = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=a_device,
        )
        self.actor = get_peft_model(self.actor_base, actor_lora_config)
        self.actor.print_trainable_parameters()

        self.critic_base = AutoModel.from_pretrained(
            model_name,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=c_device,
        )
        self.critic_backbone = get_peft_model(self.critic_base, critic_lora_config)

        self.value_head = (
            nn.Linear(self.critic_base.config.hidden_size, 1, bias=False)
            .to(c_device)
            .float()
        )
        self.value_head.weight.requires_grad_(True)

        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=ref_device,
        )
        self.ref_model.eval()
        self.ref_model.requires_grad_(False)

        self.rm_base = AutoModel.from_pretrained(
            model_name,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=rm_device,
        )
        # Load the LoRA adapter trained during reward-model fine-tuning
        self.rm_base = PeftModel.from_pretrained(self.rm_base, reward_model_path)
        self.reward_model = MedRewardModel(self.rm_base).to(rm_device)

        # Load the trained score head
        score_head_path = os.path.join(reward_model_path, "score_head.pt")
        if os.path.exists(score_head_path):
            self.reward_model.score_head.load_state_dict(
                torch.load(score_head_path, map_location="cpu")
            )
            print(f"Reward model score head loaded from {score_head_path}")
        else:
            print(f"Warning: score_head.pt not found in {reward_model_path}")

        self.reward_model.eval()
        self.reward_model.requires_grad_(False)

    def forward_actor(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        input_ids = input_ids.to(self.a_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.a_device)
        return self.actor(input_ids=input_ids, attention_mask=attention_mask).logits

    def forward_critic(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        input_ids = input_ids.to(self.c_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.c_device)
        outputs = self.critic_backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden_states = outputs.last_hidden_state
        values = self.value_head(hidden_states.float()).squeeze(-1)
        return values

    @torch.no_grad()
    def forward_ref(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        input_ids = input_ids.to(self.ref_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.ref_device)
        return self.ref_model(input_ids=input_ids, attention_mask=attention_mask).logits

    @torch.no_grad()
    def get_reward_score(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        input_ids = input_ids.to(self.rm_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.rm_device)
        return self.reward_model(input_ids=input_ids, attention_mask=attention_mask)

    @torch.no_grad()
    def get_ref_token_log_probs(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        input_ids = input_ids.to(self.ref_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.ref_device)

        logits = self.ref_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
        token_logits = torch.gather(logits, -1, input_ids.unsqueeze(-1)).squeeze(-1)
        logsumexp = torch.logsumexp(logits, dim=-1)
        return token_logits - logsumexp

    @torch.no_grad()
    def get_actor_token_log_probs(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        input_ids = input_ids.to(self.a_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.a_device)

        logits = self.actor(input_ids=input_ids, attention_mask=attention_mask).logits
        token_logits = torch.gather(logits, -1, input_ids.unsqueeze(-1)).squeeze(-1)
        logsumexp = torch.logsumexp(logits, dim=-1)
        return token_logits - logsumexp

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        **generate_kwargs,
    ) -> torch.Tensor:
        input_ids = input_ids.to(self.a_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.a_device)

        actor = self.actor
        if hasattr(actor, "_module"):
            actor = actor._module

        return actor.generate(  # type: ignore[no-any-return]
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )

    def offload_inference_models(self) -> None:
        """Move frozen ref & reward models to CPU to free GPU memory for training."""
        self.ref_model.to("cpu")  # type: ignore
        self.reward_model.to("cpu")
        clean_memory()
        print("Inference models (ref, RM) offloaded to CPU.")

    def reload_inference_models(self) -> None:
        """Move frozen ref & reward models back to their GPUs for rollouts/eval."""
        self.ref_model.to(self.ref_device)  # type: ignore
        self.reward_model.to(self.rm_device)
        clean_memory()
        print("Inference models (ref, RM) reloaded to GPU.")
