"""SFT (and optional DP-SFT) training for MedDialog and preference-pair data.

Supports single-GPU, and multi-GPU via ``torch.distributed.run`` (torchrun)::

    uv run python -m torch.distributed.run \\
        --standalone --nproc_per_node=$NGPUS \\
        -m sft.train <config_file>

When DP is enabled the model is wrapped with Opacus's
``DifferentiallyPrivateDistributedDataParallel`` (DPDDP); otherwise standard
PyTorch ``DistributedDataParallel`` (DDP) is used.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from omegaconf import OmegaConf
from opacus import PrivacyEngine
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from dataset_builder.io_adapters import group_split, to_prompt, to_sft_text
from sft.utils import determine_device, print_trainable_parameters, save_sft_model

# ── Distributed helpers ──────────────────────────────────────────────


def _is_distributed() -> bool:
    """True when launched via ``torchrun`` / ``torch.distributed.run``."""
    if "LOCAL_RANK" not in os.environ:
        return False
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _global_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_main_process() -> bool:
    return _global_rank() == 0


def _setup_distributed(backend: str = "nccl") -> None:
    """Initialize the distributed process group.

    ``torchrun`` sets ``MASTER_ADDR``, ``MASTER_PORT``, ``RANK``,
    ``LOCAL_RANK``, and ``WORLD_SIZE`` automatically.
    """
    if _world_size() <= 1:
        return
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(_local_rank())


def _cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


class OpacusCausalLoss(nn.Module):
    """Per-sample causal language-modeling loss for Opacus ghost mode."""

    def __init__(self, ignore_index: int = -100, reduction: str = "mean"):
        super().__init__()
        self.ignore_index = ignore_index
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                f"Unsupported reduction={reduction!r}. Expected one of ['none', 'mean', 'sum']."
            )
        self.reduction = reduction

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor, _dummy_target=None
    ) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=self.ignore_index,
        ).view(shift_labels.size(0), -1)

        valid_mask = (shift_labels != self.ignore_index).float()
        per_sample = (token_loss * valid_mask).sum(dim=1) / valid_mask.sum(
            dim=1
        ).clamp_min(1.0)
        if self.reduction == "none":
            return per_sample
        if self.reduction == "sum":
            return per_sample.sum()
        return per_sample.mean()


def _build_row(batch: dict[str, list[Any]], idx: int) -> dict[str, Any]:
    return {k: batch[k][idx] for k in batch}


def _validate_sft_settings(
    input_format: str, label_mode: str, pair_sft_source: str
) -> None:
    valid_formats = {"auto", "meddialog", "preference_pairs"}
    valid_label_modes = {"doctor_only", "full_sequence"}
    if input_format not in valid_formats:
        raise ValueError(
            f"Unsupported input_format={input_format!r}. Expected one of {sorted(valid_formats)}."
        )
    if label_mode not in valid_label_modes:
        raise ValueError(
            f"Unsupported label_mode={label_mode!r}. Expected one of {sorted(valid_label_modes)}."
        )
    if pair_sft_source != "chosen":
        raise ValueError("Only pair_sft_source='chosen' is currently supported.")


def _tokenize_batch(
    tokenizer,
    batch: dict[str, list[Any]],
    *,
    max_length: int,
    input_format: str,
    label_mode: str,
    pair_sft_source: str,
):
    batch_size = len(next(iter(batch.values()))) if batch else 0
    texts: list[str] = []
    prompts: list[str] = []

    for i in range(batch_size):
        row = _build_row(batch, i)
        texts.append(
            to_sft_text(
                row,
                pair_source=pair_sft_source,
                input_format=input_format,  # type: ignore
            )
        )
        prompts.append(to_prompt(row))

    tok = tokenizer(texts, truncation=True, max_length=max_length, padding="max_length")
    pad_id = tokenizer.pad_token_id

    prompt_token_lens: list[int]
    if label_mode == "doctor_only":
        prompt_token_lens = [
            len(
                tokenizer(
                    prompt,
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=False,
                )["input_ids"]
            )
            for prompt in prompts
        ]
    else:
        prompt_token_lens = [0] * len(tok["input_ids"])

    labels: list[list[int]] = []
    for input_ids, prompt_len in zip(tok["input_ids"], prompt_token_lens):
        row_labels: list[int] = []
        for idx, token in enumerate(input_ids):
            if token == pad_id:
                row_labels.append(-100)
            elif label_mode == "doctor_only" and idx < prompt_len:
                row_labels.append(-100)
            else:
                row_labels.append(token)
        labels.append(row_labels)

    return {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "labels": labels,
    }


def split_batch(batch, chunk_size: int):
    keys = list(batch.keys())
    total_size = len(batch[keys[0]])
    chunks = []
    for i in range(0, total_size, chunk_size):
        chunk = {k: batch[k][i : i + chunk_size] for k in keys}
        chunks.append(chunk)
    return chunks


def _evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Eval"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            per_sample_loss = criterion(logits, labels, None)
            loss = per_sample_loss.mean()

            bsz = input_ids.size(0)
            total_loss += loss.item() * bsz
            total_count += bsz

    model.train()
    return total_loss / max(total_count, 1)


def train(
    model,
    optimizer,
    criterion,
    dataloader,
    device,
    max_physical_batch_size: int,
    enable_dp: bool,
    epoch_idx: int,
    num_epochs: int,
    step_counter: int,
    lr_scheduler=None,
):
    model.train()

    if enable_dp:
        with BatchMemoryManager(
            data_loader=dataloader,
            max_physical_batch_size=max_physical_batch_size,
            optimizer=optimizer,
        ) as memory_safe_dataloader:
            for batch in tqdm(
                memory_safe_dataloader,
                desc=f"Epoch {epoch_idx}/{num_epochs}",
                disable=not _is_main_process(),
            ):
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits
                loss = criterion(logits, labels, None)

                loss.backward()
                optimizer.step()

                if lr_scheduler is not None:
                    lr_scheduler.step()

                step_counter += 1
                if _is_main_process():
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "epoch": epoch_idx,
                            "step": step_counter,
                        }
                    )
    else:
        for batch in tqdm(
            dataloader,
            desc=f"Epoch {epoch_idx}/{num_epochs}",
            disable=not _is_main_process(),
        ):
            mini_batches = split_batch(batch, max_physical_batch_size)
            optimizer.zero_grad()
            total_loss = 0.0

            for mini in mini_batches:
                input_ids = mini["input_ids"].to(device)
                attention_mask = mini["attention_mask"].to(device)
                labels = mini["labels"].to(device)

                logits = model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits
                loss = criterion(logits, labels, None).mean() / len(mini_batches)
                loss.backward()
                total_loss += loss.item()

            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            step_counter += 1
            if _is_main_process():
                wandb.log(
                    {
                        "train/loss": total_loss * len(mini_batches),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch_idx,
                        "step": step_counter,
                    }
                )

    return step_counter


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    assert len(sys.argv) > 1, "Usage: uv run -m sft.train <config_file>"
    config = OmegaConf.load(sys.argv[1])

    # ── Distributed setup ────────────────────────────────────────────
    distributed = _is_distributed()
    if distributed:
        _setup_distributed()

    if distributed:
        device = torch.device(f"cuda:{_local_rank()}")
    else:
        device_cfg = config.get("device", None)  # type: ignore[attr-defined]
        device = torch.device(device_cfg) if device_cfg else determine_device()

    if _is_main_process():
        print(f"Using device: {device}  (world_size={_world_size()})")
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=config.wandb.name,
            group=config.wandb.get("group", None),
            config=OmegaConf.to_container(config, resolve=True),  # type: ignore[arg-type]
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        train_file = config.get("train_file") or config.get("data_file")  # type: ignore[attr-defined]
        eval_file = config.get("eval_file", None)  # type: ignore[attr-defined]
        input_format = str(config.get("input_format", "auto"))  # type: ignore[attr-defined]
        label_mode = str(config.get("label_mode", "doctor_only"))  # type: ignore[attr-defined]
        pair_sft_source = str(config.get("pair_sft_source", "chosen"))  # type: ignore[attr-defined]
        eval_split_ratio = float(config.get("eval_split_ratio", 0.05))  # type: ignore[attr-defined]
        eval_split_seed = int(config.get("eval_split_seed", 42))  # type: ignore[attr-defined]
        _validate_sft_settings(input_format, label_mode, pair_sft_source)

        if not train_file:
            raise ValueError(
                "Missing train file. Set `train_file` (or legacy `data_file`) in config."
            )

        raw_train_dataset = load_dataset("json", data_files=train_file, split="train")

        # Optional: limit training set size (for overfit baselines, ablations).
        max_train_samples = config.get("max_train_samples", None)  # type: ignore[attr-defined]
        if max_train_samples and max_train_samples < len(raw_train_dataset):
            raw_train_dataset = raw_train_dataset.shuffle(seed=eval_split_seed).select(
                range(int(max_train_samples))
            )
            if _is_main_process():
                print(f"Limiting training set to {max_train_samples} samples")

        if eval_file:
            train_dataset = raw_train_dataset
            eval_dataset = load_dataset("json", data_files=eval_file, split="train")
        else:
            split = group_split(
                raw_train_dataset,
                split="sft",
                id_column="conversation_id",
                test_size=eval_split_ratio,
                seed=eval_split_seed,
            )
            train_dataset = split["train"]
            eval_dataset = split["test"]

        train_dataset = train_dataset.map(
            lambda ex, **kwargs: _tokenize_batch(tokenizer, ex, **kwargs),
            batched=True,
            fn_kwargs={
                "max_length": config.max_seq_length,
                "input_format": input_format,
                "label_mode": label_mode,
                "pair_sft_source": pair_sft_source,
            },
        )
        eval_dataset = eval_dataset.map(
            lambda ex, **kwargs: _tokenize_batch(tokenizer, ex, **kwargs),
            batched=True,
            fn_kwargs={
                "max_length": config.max_seq_length,
                "input_format": input_format,
                "label_mode": label_mode,
                "pair_sft_source": pair_sft_source,
            },
        )

        keep_cols = ["input_ids", "attention_mask", "labels"]
        train_remove = [c for c in train_dataset.column_names if c not in keep_cols]
        eval_remove = [c for c in eval_dataset.column_names if c not in keep_cols]
        if train_remove:
            train_dataset = train_dataset.remove_columns(train_remove)
        if eval_remove:
            eval_dataset = eval_dataset.remove_columns(eval_remove)

        train_dataset.set_format(type="torch", columns=keep_cols)
        eval_dataset.set_format(type="torch", columns=keep_cols)

        # Optional: limit eval set size (speeds up per-epoch evaluation).
        max_eval_samples = config.get("max_eval_samples", None)  # type: ignore[attr-defined]
        if max_eval_samples and max_eval_samples < len(eval_dataset):
            eval_dataset = eval_dataset.select(range(int(max_eval_samples)))
            if _is_main_process():
                print(f"Limiting eval set to {max_eval_samples} samples")

        enable_dp = bool(config.get("enable_dp", False))  # type: ignore[attr-defined]

        # ── DataLoaders ──────────────────────────────────────────────
        # For DDP (with or without DP), use DistributedSampler to shard
        # data across ranks.  When DP is enabled we disable Opacus's
        # default Poisson sampling (which can give each rank a different
        # batch count, causing NCCL ALLREDUCE deadlocks) and rely on
        # uniform subsampling via DistributedSampler instead.
        train_sampler: DistributedSampler | None = None
        if distributed:
            train_sampler = DistributedSampler(
                train_dataset, num_replicas=_world_size(), rank=_global_rank()
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            drop_last=distributed,  # ensure equal batch count across ranks
        )  # type: ignore
        eval_loader = DataLoader(
            eval_dataset, batch_size=config.batch_size, shuffle=False
        )  # type: ignore

        # ── Model ────────────────────────────────────────────────────
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        )

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(base_model, lora_cfg).to(device)

        model = ModuleValidator.fix(model)
        ModuleValidator.validate(model, strict=False)

        if _is_main_process():
            print_trainable_parameters(model)

        # ── DDP / DPDDP wrapping (BEFORE PrivacyEngine) ─────────────
        if distributed:
            if enable_dp:
                model = DPDDP(model)
            else:
                model = DDP(model, device_ids=[_local_rank()])

        train_criterion = OpacusCausalLoss(ignore_index=-100, reduction="mean")
        eval_criterion = OpacusCausalLoss(ignore_index=-100, reduction="none")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        if enable_dp:
            privacy_engine = PrivacyEngine()
            model, optimizer, criterion_gc, train_loader = (  # type: ignore[misc]
                privacy_engine.make_private_with_epsilon(
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    criterion=train_criterion,  # type: ignore
                    target_epsilon=config.target_epsilon,
                    target_delta=config.target_delta,
                    epochs=config.num_epochs,
                    max_grad_norm=config.max_grad_norm,
                    grad_sample_mode="ghost",
                    poisson_sampling=not distributed,
                )
            )
        else:
            privacy_engine = None
            criterion_gc = train_criterion

        steps_per_epoch_physical = max(
            1, len(train_dataset) // config.max_physical_batch_size
        )
        total_steps = steps_per_epoch_physical * config.num_epochs
        warmup_steps = int(0.08 * total_steps)

        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        if _is_main_process() and hasattr(optimizer, "noise_multiplier"):
            wandb.config.update({"actual_noise_multiplier": optimizer.noise_multiplier})  # type: ignore[attr-defined]

        # ── Log DP accounting parameters ─────────────────────────────
        dp_accounting: dict[str, Any] | None = None
        accounting_path: Path | None = None
        if enable_dp and _is_main_process():
            n_samples = len(train_dataset)
            q = config.max_physical_batch_size / n_samples if n_samples > 0 else 0
            dp_accounting = {
                "dp_n_samples": n_samples,
                "dp_sampling_probability_q": q,
                "dp_total_steps_T": total_steps,
                "dp_epochs": config.num_epochs,
                "dp_max_grad_norm_C": config.max_grad_norm,
                "dp_target_epsilon": config.target_epsilon,
                "dp_target_delta": config.target_delta,
                "dp_noise_multiplier_sigma": getattr(
                    optimizer, "noise_multiplier", None
                ),
            }
            wandb.config.update(dp_accounting)
            accounting_path = Path(config.output_dir) / "dp_accounting.json"
            accounting_path.parent.mkdir(parents=True, exist_ok=True)
            accounting_path.write_text(
                json.dumps(dp_accounting, indent=2), encoding="utf-8"
            )
            print(f"DP accounting saved: {accounting_path}")

        # ── Training loop ────────────────────────────────────────────
        step_counter = 0

        for epoch in range(1, config.num_epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            step_counter = train(
                model=model,
                optimizer=optimizer,
                criterion=criterion_gc,
                dataloader=train_loader,
                device=device,
                max_physical_batch_size=config.max_physical_batch_size,
                enable_dp=enable_dp,
                epoch_idx=epoch,
                num_epochs=config.num_epochs,
                step_counter=step_counter,
                lr_scheduler=lr_scheduler,
            )

            # Synchronise ranks before eval to avoid training/eval overlap.
            if distributed:
                dist.barrier()

            # All ranks must run eval to keep DDP's NCCL ops in sync.
            eval_loss = _evaluate(model, eval_loader, eval_criterion, device)
            if _is_main_process():
                wandb.log({"eval/loss": eval_loss, "epoch": epoch})
                print(f"[Epoch {epoch}] Eval loss: {eval_loss:.4f}")
                save_sft_model(model, tokenizer, config.output_dir, epoch=epoch)

        if enable_dp and privacy_engine is not None and _is_main_process():
            realized_epsilon = privacy_engine.get_epsilon(config.target_delta)
            if dp_accounting is not None and accounting_path is not None:
                dp_accounting["dp_realized_epsilon"] = realized_epsilon
                accounting_path.write_text(
                    json.dumps(dp_accounting, indent=2), encoding="utf-8"
                )
            print(f"Final privacy epsilon: {realized_epsilon:.4f}")

        if _is_main_process():
            wandb.finish()
            print("Done")

    finally:
        if distributed:
            _cleanup_distributed()


if __name__ == "__main__":
    main()
