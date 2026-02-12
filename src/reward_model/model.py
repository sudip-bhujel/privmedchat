"""MedDialog reward model."""

from __future__ import annotations

import torch
import torch.nn as nn


class MedRewardModel(nn.Module):
    """Backbone + scalar score head with last-token pooling."""

    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model
        self.score_head = nn.Linear(base_model.config.hidden_size, 1, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        last_hidden = outputs.last_hidden_state
        batch_size = input_ids.size(0)
        sequence_lengths = (attention_mask.sum(dim=1) - 1).clamp(min=0).long()
        batch_indices = torch.arange(batch_size, device=last_hidden.device)

        last_token_embeddings = last_hidden[batch_indices, sequence_lengths]
        scores = self.score_head(last_token_embeddings.to(self.score_head.weight.dtype))
        return scores
