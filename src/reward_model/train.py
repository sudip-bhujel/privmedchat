"""Train a reward model on preference pairs with optional differential privacy."""

from __future__ import annotations

import sys
import warnings
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from omegaconf import OmegaConf
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from dataset_builder.io_adapters import group_split, to_pair_texts
from reward_model.model import MedRewardModel
from reward_model.utils import (
    determine_device,
    print_trainable_parameters,
    save_reward_model,
)


class OpacusPairwiseLoss(nn.Module):
    """Pairwise loss compatible with Opacus ghost mode."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, all_rewards: torch.Tensor, _dummy_target=None):
        batch_size = all_rewards.size(0) // 2

        chosen_rewards = all_rewards[:batch_size]
        rejected_rewards = all_rewards[batch_size:]

        loss = -F.logsigmoid(chosen_rewards - rejected_rewards)
        per_sample_losses = torch.cat([loss, loss], dim=0)
        per_sample_losses = 0.5 * per_sample_losses

        if self.reduction == "none":
            return per_sample_losses
        if self.reduction == "sum":
            return per_sample_losses.sum()
        return per_sample_losses.mean()


def _process_batch(batch, model, device):
    input_ids = torch.cat(
        [batch["input_ids_chosen"], batch["input_ids_rejected"]], dim=0
    ).to(device)
    attention_mask = torch.cat(
        [batch["attention_mask_chosen"], batch["attention_mask_rejected"]],
        dim=0,
    ).to(device)

    all_rewards = model(input_ids=input_ids, attention_mask=attention_mask)
    return all_rewards


def _compute_metrics(all_rewards: torch.Tensor):
    batch_size = all_rewards.size(0) // 2
    rewards_chosen = all_rewards[:batch_size].squeeze(-1)
    rewards_rejected = all_rewards[batch_size:].squeeze(-1)

    accuracy = (rewards_chosen > rewards_rejected).float().mean().item()
    margin = (rewards_chosen - rewards_rejected).mean().item()
    mean_chosen = rewards_chosen.mean().item()
    mean_rejected = rewards_rejected.mean().item()

    return accuracy, margin, mean_chosen, mean_rejected


def _log_metrics(
    loss_value,
    accuracy,
    margin,
    mean_chosen,
    mean_rejected,
    optimizer,
    epoch_idx,
    step_counter,
):
    wandb.log(
        {
            "train/loss": loss_value,
            "train/accuracy": accuracy,
            "train/margin": margin,
            "train/score_chosen": mean_chosen,
            "train/score_rejected": mean_rejected,
            "train/lr": optimizer.param_groups[0]["lr"],
            "epoch": epoch_idx,
            "step": step_counter,
        }
    )


def split_batch(batch, chunk_size):
    keys = list(batch.keys())
    total_size = len(batch[keys[0]])
    chunks = []
    for i in range(0, total_size, chunk_size):
        chunk = {k: batch[k][i : i + chunk_size] for k in keys}
        chunks.append(chunk)
    return chunks


def _build_row(batch: dict[str, list[Any]], idx: int) -> dict[str, Any]:
    return {k: batch[k][idx] for k in batch}


def _tokenize_preference_batch(tokenizer, batch: dict[str, list[Any]], max_length: int):
    batch_size = len(next(iter(batch.values()))) if batch else 0
    chosen_texts: list[str] = []
    rejected_texts: list[str] = []

    for i in range(batch_size):
        row = _build_row(batch, i)
        chosen, rejected = to_pair_texts(row, input_format="preference_pairs")
        chosen_texts.append(chosen)
        rejected_texts.append(rejected)

    tok_chosen = tokenizer(
        chosen_texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    tok_rejected = tokenizer(
        rejected_texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    return {
        "input_ids_chosen": tok_chosen["input_ids"],
        "attention_mask_chosen": tok_chosen["attention_mask"],
        "input_ids_rejected": tok_rejected["input_ids"],
        "attention_mask_rejected": tok_rejected["attention_mask"],
    }


def train(
    model,
    optimizer,
    criterion,
    dataloader,
    device,
    epoch_idx: int,
    num_epochs: int,
    step_counter: int,
    max_physical_batch_size: int,
    lr_scheduler=None,
    enable_dp: bool = True,
):
    model.train()

    if enable_dp:
        with BatchMemoryManager(
            data_loader=dataloader,
            max_physical_batch_size=max_physical_batch_size,
            optimizer=optimizer,
        ) as memory_safe_dataloader:
            for batch in tqdm(
                memory_safe_dataloader, desc=f"Epoch {epoch_idx}/{num_epochs}"
            ):
                optimizer.zero_grad()

                all_rewards = _process_batch(batch, model, device)
                loss = criterion(all_rewards, None)

                loss.backward()
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()

                with torch.no_grad():
                    accuracy, margin, mean_chosen, mean_rejected = _compute_metrics(
                        all_rewards
                    )

                step_counter += 1
                _log_metrics(
                    loss.item(),
                    accuracy,
                    margin,
                    mean_chosen,
                    mean_rejected,
                    optimizer,
                    epoch_idx,
                    step_counter,
                )
    else:
        for batch in tqdm(dataloader, desc=f"Epoch {epoch_idx}/{num_epochs}"):
            mini_batches = split_batch(batch, max_physical_batch_size)

            optimizer.zero_grad()
            accumulated_loss = 0.0

            for mini_batch in mini_batches:
                all_rewards = _process_batch(mini_batch, model, device)
                loss = criterion(all_rewards, None).mean() / len(mini_batches)
                loss.backward()
                accumulated_loss += loss.item()

            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            with torch.no_grad():
                accuracy, margin, mean_chosen, mean_rejected = _compute_metrics(
                    all_rewards  # type: ignore[misc]
                )

            step_counter += 1
            _log_metrics(
                accumulated_loss * len(mini_batches),
                accuracy,
                margin,
                mean_chosen,
                mean_rejected,
                optimizer,
                epoch_idx,
                step_counter,
            )

    return step_counter


@torch.inference_mode()
def evaluate(model, dataloader, criterion, device, epoch: int):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_pairs = 0
    total_margin = 0.0

    all_chosen_scores = []
    all_rejected_scores = []

    for batch in tqdm(dataloader, desc=f"Evaluating {epoch}"):
        input_ids = torch.cat(
            [batch["input_ids_chosen"], batch["input_ids_rejected"]], dim=0
        ).to(device)
        attention_mask = torch.cat(
            [batch["attention_mask_chosen"], batch["attention_mask_rejected"]],
            dim=0,
        ).to(device)

        all_rewards = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = criterion(all_rewards, None).mean()

        batch_size = all_rewards.size(0) // 2
        rewards_chosen = all_rewards[:batch_size].squeeze(-1)
        rewards_rejected = all_rewards[batch_size:].squeeze(-1)

        total_loss += loss.item() * batch_size
        total_correct += (rewards_chosen > rewards_rejected).float().sum().item()
        total_margin += (rewards_chosen - rewards_rejected).sum().item()
        total_pairs += batch_size

        all_chosen_scores.extend(rewards_chosen.cpu().numpy().tolist())
        all_rejected_scores.extend(rewards_rejected.cpu().numpy().tolist())

    avg_loss = total_loss / total_pairs if total_pairs else 0.0
    avg_acc = total_correct / total_pairs if total_pairs else 0.0
    avg_margin = total_margin / total_pairs if total_pairs else 0.0

    print(f"\n[Epoch {epoch}] Eval Results:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Acc:  {avg_acc:.2%}")
    print(f"  Mrgn: {avg_margin:.4f}")

    wandb.log(
        {
            "eval/loss": avg_loss,
            "eval/accuracy": avg_acc,
            "eval/margin": avg_margin,
            "epoch": epoch,
            "eval/score_dist": wandb.plot.histogram(
                wandb.Table(
                    data=[[s] for s in all_chosen_scores]
                    + [[s] for s in all_rejected_scores],
                    columns=["score"],
                ),
                "score",
                title="Score Distribution (Combined)",
            ),
        }
    )

    model.train()
    return avg_acc


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    assert len(sys.argv) > 1, "Usage: uv run -m reward_model.train <config_file>"
    config = OmegaConf.load(sys.argv[1])

    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.wandb.name,
        config=OmegaConf.to_container(config, resolve=True),  # type: ignore[arg-type]
    )

    device_cfg = config.get("device", None)  # type: ignore[attr-defined]
    device = torch.device(device_cfg) if device_cfg else determine_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    raw_dataset = load_dataset(
        "json", data_files=config.preference_pairs_file, split="train"
    )
    eval_split_ratio = float(config.get("eval_split_ratio", 0.05))  # type: ignore[attr-defined]
    eval_split_seed = int(config.get("eval_split_seed", 42))  # type: ignore[attr-defined]
    split = group_split(
        raw_dataset,
        split="rm",
        id_column="conversation_id",
        test_size=eval_split_ratio,
        seed=eval_split_seed,
    )

    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")

    train_dataset = train_dataset.map(
        lambda ex, **kwargs: _tokenize_preference_batch(tokenizer, ex, **kwargs),
        batched=True,
        fn_kwargs={"max_length": config.max_seq_length},
    )
    eval_dataset = eval_dataset.map(
        lambda ex, **kwargs: _tokenize_preference_batch(tokenizer, ex, **kwargs),
        batched=True,
        fn_kwargs={"max_length": config.max_seq_length},
    )

    keep_cols = [
        "input_ids_chosen",
        "attention_mask_chosen",
        "input_ids_rejected",
        "attention_mask_rejected",
    ]

    remove_train_cols = [c for c in train_dataset.column_names if c not in keep_cols]
    remove_eval_cols = [c for c in eval_dataset.column_names if c not in keep_cols]
    if remove_train_cols:
        train_dataset = train_dataset.remove_columns(remove_train_cols)
    if remove_eval_cols:
        eval_dataset = eval_dataset.remove_columns(remove_eval_cols)

    train_dataset.set_format(type="torch", columns=keep_cols)
    eval_dataset.set_format(type="torch", columns=keep_cols)

    eval_batch_size = int(
        config.get("eval_batch_size", config.get("max_physical_batch_size", 1))  # type: ignore[attr-defined]
    )

    train_dataloader = DataLoader(
        train_dataset,  # type: ignore[attr-defined]
        batch_size=config.batch_size,
        shuffle=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,  # type: ignore[attr-defined]
        batch_size=eval_batch_size,
        shuffle=False,
    )

    dtype_cfg = str(config.get("torch_dtype", "auto")).lower()  # type: ignore[attr-defined]
    if dtype_cfg == "bfloat16":
        model_dtype = torch.bfloat16
    elif dtype_cfg == "float16":
        model_dtype = torch.float16
    elif dtype_cfg == "float32":
        model_dtype = torch.float32
    else:
        bf16_supported = bool(
            torch.cuda.is_available()
            and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        )
        model_dtype = (
            torch.bfloat16
            if bf16_supported
            else (torch.float16 if torch.cuda.is_available() else torch.float32)
        )

    try:
        base_model = AutoModel.from_pretrained(
            config.model_name,
            dtype=model_dtype,
        )
    except TypeError:
        # Backward-compat for older Transformers versions.
        base_model = AutoModel.from_pretrained(
            config.model_name,
            torch_dtype=model_dtype,
        )

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    base_model = get_peft_model(base_model, peft_config)

    model = MedRewardModel(base_model).to(device)
    model.train()

    model.score_head.weight.requires_grad = True
    model.score_head = model.score_head.float()

    try:
        model = ModuleValidator.fix(model)
    except Exception as exc:
        if "Can't pickle local object" not in str(exc):
            raise
        warnings.warn(
            "ModuleValidator.fix() failed due a non-picklable hook; continuing without fix."
        )
    ModuleValidator.validate(model, strict=False)

    enable_dp = bool(config.get("enable_dp", True))  # type: ignore[attr-defined]

    # Gradient checkpointing is incompatible with Opacus ghost clipping mode
    # Only enable it when DP is disabled
    use_gradient_checkpointing = bool(config.get("gradient_checkpointing", True))  # type: ignore[attr-defined]
    if use_gradient_checkpointing and not enable_dp:
        if hasattr(model.backbone, "gradient_checkpointing_enable"):
            model.backbone.gradient_checkpointing_enable()  # type: ignore[attr-defined]
        if hasattr(model.backbone, "enable_input_require_grads"):
            model.backbone.enable_input_require_grads()  # type: ignore[attr-defined]
        backbone_cfg = getattr(model.backbone, "config", None)
        if backbone_cfg is not None and hasattr(backbone_cfg, "use_cache"):
            backbone_cfg.use_cache = False
    elif use_gradient_checkpointing and enable_dp:
        warnings.warn(
            "Gradient checkpointing disabled: incompatible with Opacus ghost clipping mode."
        )

    print_trainable_parameters(model)

    train_criterion = OpacusPairwiseLoss(reduction="mean")
    eval_criterion = OpacusPairwiseLoss(reduction="none")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    privacy_engine = PrivacyEngine()

    if enable_dp:
        model, optimizer, criterion_gc, train_dataloader = (  # type: ignore[misc]
            privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_dataloader,
                criterion=train_criterion,  # type: ignore[misc]
                target_epsilon=config.target_epsilon,
                target_delta=config.target_delta,
                epochs=config.num_epochs,
                max_grad_norm=config.max_grad_norm,
                grad_sample_mode="ghost",
            )
        )
    else:
        criterion_gc = train_criterion

    steps_per_epoch_physical = max(
        1, len(train_dataset) // config.max_physical_batch_size
    )
    total_steps = steps_per_epoch_physical * config.num_epochs
    num_warmup_steps = int(0.08 * total_steps)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    if hasattr(optimizer, "noise_multiplier"):
        wandb.config.update({"actual_noise_multiplier": optimizer.noise_multiplier})  # type: ignore[attr-defined]
        print(
            f"DP training initiated with noise multiplier: {optimizer.noise_multiplier}"  # type: ignore[misc]
        )

    step_counter = 0

    for epoch in range(1, config.num_epochs + 1):
        step_counter = train(
            model,
            optimizer,
            criterion_gc,
            train_dataloader,
            device,
            epoch_idx=epoch,
            num_epochs=config.num_epochs,
            step_counter=step_counter,
            max_physical_batch_size=config.max_physical_batch_size,
            lr_scheduler=lr_scheduler,
            enable_dp=enable_dp,
        )

        epoch_dir = f"{config.output_dir}/epoch_{epoch}"
        save_reward_model(model, tokenizer, epoch_dir)

        acc = evaluate(model, eval_dataloader, eval_criterion, device, epoch)
        print(f"Evaluation Accuracy: {acc * 100:.2f}%")

    if enable_dp:
        print(
            f"Final privacy epsilon: {privacy_engine.get_epsilon(config.target_delta):.4f}"
        )

    wandb.finish()
    print("Done")


if __name__ == "__main__":
    main()
