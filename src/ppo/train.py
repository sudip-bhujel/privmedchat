"""PPO training with differential privacy for medical dialogue alignment."""

from __future__ import annotations

import random
import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from omegaconf import OmegaConf
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from peft import LoraConfig, TaskType
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset_builder.io_adapters import group_split
from ppo.dataset import PPODataset, extract_prompts_from_dataset, ppo_collate_fn
from ppo.model import ActorCritic
from ppo.utils import clean_memory, collect_rollouts, save_ppo_model
from reward_model.utils import determine_device


@torch.inference_mode()
def evaluate(
    model: ActorCritic,
    tokenizer,
    prompts,
    batch_size: int,
    max_new_tokens: int,
    device,
    iteration: int,
):
    """Evaluate PPO model via greedy decoding on eval prompts."""
    model.eval()

    system_prompt = (
        "You are a careful medical assistant. Provide safe, concise, non-diagnostic guidance "
        "and recommend professional care when needed."
    )

    all_rewards = []
    all_kl_divs = []

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Evaluating {iteration}"):
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
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
        )

        rm_scores = model.get_reward_score(
            input_ids=outputs, attention_mask=torch.ones_like(outputs)
        )
        rm_scores_flat = rm_scores.squeeze().cpu().tolist()
        if not isinstance(rm_scores_flat, list):
            rm_scores_flat = [rm_scores_flat]
        all_rewards.extend(rm_scores_flat)

        ref_token_log_probs = model.get_ref_token_log_probs(outputs).to(device)
        actor_token_log_probs = model.get_actor_token_log_probs(outputs).to(device)

        action_mask = torch.zeros_like(outputs, device=device)
        action_mask[:, prompt_len:] = 1
        if tokenizer.pad_token_id is not None:
            outputs_on_device = outputs.to(device)
            action_mask[outputs_on_device == tokenizer.pad_token_id] = 0

        kl_div = actor_token_log_probs - ref_token_log_probs
        for j in range(len(batch_prompts)):
            valid_tokens = action_mask[j].sum()
            if valid_tokens > 0:
                seq_kl = (kl_div[j] * action_mask[j]).sum() / valid_tokens
                all_kl_divs.append(seq_kl.cpu().item())

        del inputs, outputs, rm_scores, ref_token_log_probs, actor_token_log_probs
        del action_mask, kl_div
        clean_memory()

    avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    avg_kl = sum(all_kl_divs) / len(all_kl_divs) if all_kl_divs else 0.0

    wandb.log(
        {
            "eval/reward": avg_reward,
            "eval/kl_divergence": avg_kl,
            "iteration": iteration,
        }
    )

    print(f"\n[Iteration {iteration}] Eval Results:")
    print(f"  Avg Reward: {avg_reward:.4f}")
    print(f"  Avg KL: {avg_kl:.4f}")

    model.train()
    return avg_reward


def _split_batch(batch, chunk_size: int):
    """Split a collated batch into smaller micro-batches."""
    keys = list(batch.keys())
    total = len(batch[keys[0]])
    chunks = []
    for i in range(0, total, chunk_size):
        chunks.append({k: batch[k][i : i + chunk_size] for k in keys})
    return chunks


def _actor_step(
    model, batch, device, clip_epsilon, beta_kl, old_log_probs_key="old_log_probs"
):
    """Compute actor (policy) loss for a single micro-batch.

    Returns (actor_loss, kl_mean) where kl_mean is the mean per-token KL
    divergence between the current policy and the reference model (detached).
    """
    input_ids = batch["input_ids"].to(device)
    action_mask = batch["action_mask"].to(device)
    old_log_probs = batch["old_log_probs"].to(device)
    ref_log_probs = batch["ref_log_probs"].to(device)
    rewards = batch["rewards"].float().to(device)

    actor_logits = model.forward_actor(input_ids=input_ids, attention_mask=None)
    # Use F.cross_entropy (fused kernel) to avoid materializing full [B,T,V]
    # log_softmax tensor — saves ~500MB+ GPU memory per step for large vocabs.
    current_taken_log_probs = -F.cross_entropy(
        actor_logits.view(-1, actor_logits.size(-1)),
        input_ids.view(-1),
        reduction="none",
    ).view(input_ids.shape)
    del actor_logits

    with torch.no_grad():
        values = model.forward_critic(input_ids, attention_mask=None).to(device)

    kl_penalty = current_taken_log_probs - ref_log_probs
    combined_rewards = rewards.unsqueeze(1) - beta_kl * kl_penalty
    advantages = combined_rewards - values.detach()

    ratio = torch.exp(current_taken_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    actor_loss = -torch.min(surr1, surr2)

    valid_tokens = action_mask.sum()
    actor_loss = (actor_loss * action_mask).sum() / valid_tokens

    # Mean KL for logging (detached)
    kl_mean = (kl_penalty.detach() * action_mask).sum() / valid_tokens
    return actor_loss, kl_mean.item()


def _critic_step(model, batch, device, beta_kl):
    """Compute critic (value) loss for a single micro-batch. Returns scalar loss."""
    input_ids = batch["input_ids"].to(device)
    action_mask = batch["action_mask"].to(device)
    ref_log_probs = batch["ref_log_probs"].to(device)
    rewards = batch["rewards"].float().to(device)

    with torch.no_grad():
        actor_logits = model.forward_actor(input_ids=input_ids, attention_mask=None)
        current_taken_log_probs = -F.cross_entropy(
            actor_logits.view(-1, actor_logits.size(-1)),
            input_ids.view(-1),
            reduction="none",
        ).view(input_ids.shape)
        del actor_logits

    values = model.forward_critic(input_ids, attention_mask=None)

    kl_penalty = current_taken_log_probs - ref_log_probs
    combined_rewards = rewards.unsqueeze(1) - beta_kl * kl_penalty
    combined_rewards = combined_rewards.to(values.device)

    critic_loss = 0.5 * (values - combined_rewards.detach()).pow(2)
    action_mask_critic = action_mask.to(values.device)
    valid_tokens = action_mask_critic.sum()
    critic_loss = (critic_loss * action_mask_critic).sum() / valid_tokens
    return critic_loss


def train_epoch(
    model,
    actor_optim,
    critic_optim,
    dataloader,
    max_physical_batch_size: int,
    clip_epsilon: float,
    beta_kl: float,
    device,
    iteration: int,
    epoch: int,
    num_epochs: int,
    step_counter: int,
    get_epsilon_fn,
    enable_dp: bool = True,
):
    """Train one PPO epoch with separate actor and critic passes."""
    model.train()
    total_actor_loss = 0.0
    total_critic_loss = 0.0
    total_kl = 0.0
    actor_steps = 0
    critic_steps = 0

    # ---- Actor pass ----
    if enable_dp:
        with BatchMemoryManager(
            data_loader=dataloader,
            max_physical_batch_size=max_physical_batch_size,
            optimizer=actor_optim,
        ) as actor_dataloader:
            for batch in tqdm(actor_dataloader, desc=f"Iteration {iteration} [Actor]"):
                actor_optim.zero_grad()
                actor_loss, kl_mean = _actor_step(
                    model, batch, device, clip_epsilon, beta_kl
                )
                actor_loss.backward()
                actor_optim.step()
                total_actor_loss += actor_loss.item()
                total_kl += kl_mean
                actor_steps += 1
                del actor_loss
                clean_memory()
    else:
        for batch in tqdm(dataloader, desc=f"Iteration {iteration} [Actor]"):
            mini_batches = _split_batch(batch, max_physical_batch_size)
            actor_optim.zero_grad()
            accumulated_loss = 0.0
            accumulated_kl = 0.0
            for mb in mini_batches:
                actor_loss, kl_mean = _actor_step(
                    model, mb, device, clip_epsilon, beta_kl
                )
                (actor_loss / len(mini_batches)).backward()
                accumulated_loss += actor_loss.item()
                accumulated_kl += kl_mean
                del actor_loss
            actor_optim.step()
            total_actor_loss += accumulated_loss
            total_kl += accumulated_kl / len(mini_batches)
            actor_steps += 1
            clean_memory()

    clean_memory()

    # ---- Critic pass ----
    if enable_dp:
        with BatchMemoryManager(
            data_loader=dataloader,
            max_physical_batch_size=max_physical_batch_size,
            optimizer=critic_optim,
        ) as critic_dataloader:
            for batch in tqdm(
                critic_dataloader, desc=f"Iteration {iteration} [Critic]"
            ):
                critic_optim.zero_grad()
                critic_loss = _critic_step(model, batch, device, beta_kl)
                critic_loss.backward()
                critic_optim.step()
                total_critic_loss += critic_loss.item()
                critic_steps += 1
                del critic_loss
                clean_memory()
    else:
        for batch in tqdm(dataloader, desc=f"Iteration {iteration} [Critic]"):
            mini_batches = _split_batch(batch, max_physical_batch_size)
            critic_optim.zero_grad()
            accumulated_loss = 0.0
            for mb in mini_batches:
                critic_loss = _critic_step(model, mb, device, beta_kl)
                (critic_loss / len(mini_batches)).backward()
                accumulated_loss += critic_loss.item()
                del critic_loss
            critic_optim.step()
            total_critic_loss += accumulated_loss
            critic_steps += 1
            clean_memory()

    avg_actor_loss = total_actor_loss / actor_steps if actor_steps else 0.0
    avg_critic_loss = total_critic_loss / critic_steps if critic_steps else 0.0
    avg_kl = total_kl / actor_steps if actor_steps else 0.0
    avg_loss = avg_actor_loss + avg_critic_loss

    log_dict: dict = {
        "train/actor_loss": avg_actor_loss,
        "train/critic_loss": avg_critic_loss,
        "train/loss": avg_loss,
        "train/kl_divergence": avg_kl,
        "iteration": iteration,
        "global_step": step_counter,
        "ppo_epoch": epoch,
    }
    if enable_dp:
        actor_epsilon, critic_epsilon = get_epsilon_fn()
        log_dict["train/actor_epsilon"] = actor_epsilon
        log_dict["train/critic_epsilon"] = critic_epsilon
    wandb.log(log_dict)

    clean_memory()
    return avg_loss


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    assert len(sys.argv) > 1, "Usage: uv run -m ppo.train <config_file>"
    config = OmegaConf.load(sys.argv[1])

    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.wandb.name,
        group=config.wandb.get("group", None),
        config=OmegaConf.to_container(config, resolve=True),  # type: ignore[arg-type],
    )

    wandb.define_metric("iteration")
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("rollout/*", step_metric="iteration")
    wandb.define_metric("eval/*", step_metric="iteration")
    wandb.define_metric("iteration/*", step_metric="iteration")

    device_cfg = config.get("device", None)  # type: ignore[attr-defined]
    device = torch.device(device_cfg) if device_cfg else determine_device()
    print(f"Using device: {device}")

    actor_lora = LoraConfig(
        r=config.actor.lora.r,
        lora_alpha=config.actor.lora.alpha,
        lora_dropout=config.actor.lora.dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],
    )
    critic_lora = LoraConfig(
        r=config.critic.lora.r,
        lora_alpha=config.critic.lora.alpha,
        lora_dropout=config.critic.lora.dropout,
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=["q_proj", "v_proj"],
    )

    ppo_system = ActorCritic(
        model_name=config.model_name,
        reward_model_path=config.rm.model_path,
        actor_lora_config=actor_lora,
        critic_lora_config=critic_lora,
        a_device=config.actor.device,
        c_device=config.critic.device,
        ref_device=config.ref.device,
        rm_device=config.rm.device,
        sft_checkpoint=config.get("sft_checkpoint", None),
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    raw_dataset = load_dataset(
        "json", data_files=config.preference_pairs_file, split="train"
    )
    eval_split_ratio = float(config.get("eval_split_ratio", 0.1))  # type: ignore[attr-defined]
    eval_split_seed = int(config.get("eval_split_seed", 42))  # type: ignore[attr-defined]
    split = group_split(
        raw_dataset,
        split="ppo",
        id_column="conversation_id",
        test_size=eval_split_ratio,
        seed=eval_split_seed,
    )
    ppo_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"Training samples: {len(ppo_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")

    train_prompts = extract_prompts_from_dataset(ppo_dataset)
    eval_prompts = extract_prompts_from_dataset(eval_dataset)
    print(f"Unique train prompts extracted: {len(train_prompts)}")
    print(f"Unique eval prompts extracted: {len(eval_prompts)}")

    max_rollout_prompts = getattr(config, "max_rollout_prompts", None)

    actor_optim = torch.optim.AdamW(ppo_system.actor.parameters(), lr=config.actor.lr)
    critic_optim = torch.optim.AdamW(
        list(ppo_system.critic_backbone.parameters())
        + list(ppo_system.value_head.parameters()),
        lr=config.critic.lr,
    )

    if max_rollout_prompts and max_rollout_prompts < len(train_prompts):
        sampled_prompts = random.sample(train_prompts, max_rollout_prompts)
    else:
        sampled_prompts = train_prompts

    initial_rollouts = collect_rollouts(
        model=ppo_system,
        tokenizer=tokenizer,
        prompts=sampled_prompts,
        batch_size=config.rollout_batch_size,
        max_new_tokens=config.max_new_tokens,
        device=device,
    )

    initial_dataset = PPODataset(initial_rollouts)
    collate_fn = lambda batch: ppo_collate_fn(
        batch, pad_token_id=tokenizer.pad_token_id
    )
    initial_dataloader = DataLoader(
        initial_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    ppo_system.train()
    enable_dp = bool(config.get("enable_dp", True))  # type: ignore[attr-defined]
    total_epochs = config.num_iterations * config.epochs_per_iteration

    privacy_engine_actor = None
    privacy_engine_critic = None

    if enable_dp:
        privacy_engine_actor = PrivacyEngine()
        actor_criterion = nn.MSELoss(reduction="mean")
        ppo_system.actor, actor_optim, _, _ = (  # type: ignore[misc]
            privacy_engine_actor.make_private_with_epsilon(
                module=ppo_system.actor,
                optimizer=actor_optim,
                data_loader=initial_dataloader,
                criterion=actor_criterion,  # type: ignore[misc]
                target_epsilon=config.dp.actor.target_epsilon,
                target_delta=config.dp.actor.target_delta,
                max_grad_norm=config.dp.actor.max_grad_norm,
                epochs=total_epochs,
                grad_sample_mode="ghost",
                loss_reduction="mean",
            )
        )

        critic_module = torch.nn.ModuleList(
            [ppo_system.critic_backbone, ppo_system.value_head]
        )
        privacy_engine_critic = PrivacyEngine()
        critic_criterion = nn.MSELoss(reduction="mean")
        critic_module, critic_optim, _, _ = (  # type: ignore[misc]
            privacy_engine_critic.make_private_with_epsilon(
                module=critic_module,
                optimizer=critic_optim,
                data_loader=initial_dataloader,
                criterion=critic_criterion,  # type: ignore[misc]
                target_epsilon=config.dp.critic.target_epsilon,
                target_delta=config.dp.critic.target_delta,
                max_grad_norm=config.dp.critic.max_grad_norm,
                epochs=total_epochs,
                grad_sample_mode="ghost",
                loss_reduction="mean",
            )
        )

        if hasattr(actor_optim, "noise_multiplier"):
            print(f"Actor noise multiplier: {actor_optim.noise_multiplier}")
        if hasattr(critic_optim, "noise_multiplier"):
            print(f"Critic noise multiplier: {critic_optim.noise_multiplier}")
    else:
        print("Differential privacy disabled.")

    actor_scheduler = CosineAnnealingLR(
        actor_optim, T_max=config.num_iterations, eta_min=config.actor.lr * 0.1
    )
    critic_scheduler = CosineAnnealingLR(
        critic_optim, T_max=config.num_iterations, eta_min=config.critic.lr * 0.1
    )

    for iteration in range(1, config.num_iterations + 1):
        if max_rollout_prompts and max_rollout_prompts < len(train_prompts):
            sampled_prompts = random.sample(train_prompts, max_rollout_prompts)
        else:
            sampled_prompts = train_prompts

        rollouts = collect_rollouts(
            model=ppo_system,
            tokenizer=tokenizer,
            prompts=sampled_prompts,
            batch_size=config.rollout_batch_size,
            max_new_tokens=config.max_new_tokens,
            device=device,
        )
        dataset = PPODataset(rollouts)

        all_rewards = [r["rewards"] for r in rollouts]
        avg_reward = sum(all_rewards) / len(all_rewards)
        std_reward = (
            torch.tensor(all_rewards).std().item() if len(all_rewards) > 1 else 0.0
        )
        wandb.log(
            {
                "rollout/reward": avg_reward,
                "rollout/reward_std": std_reward,
                "iteration": iteration,
            }
        )

        print(f"\n[Iteration {iteration}] Rollout Stats:")
        print(f"  Avg Reward: {avg_reward:.4f}")
        print(f"  Std Reward: {std_reward:.4f}")

        clean_memory()
        ppo_system.train()

        # Offload ref & RM to CPU — only actor+critic needed during training
        ppo_system.offload_inference_models()

        iteration_loss = 0.0

        for ppo_epoch in range(1, config.epochs_per_iteration + 1):
            dataloader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
            )
            global_step = (iteration - 1) * config.epochs_per_iteration + ppo_epoch

            epoch_loss = train_epoch(
                model=ppo_system,
                actor_optim=actor_optim,
                critic_optim=critic_optim,
                dataloader=dataloader,
                max_physical_batch_size=config.max_physical_batch_size,
                clip_epsilon=config.clip_epsilon,
                beta_kl=config.beta_kl,
                device=device,
                iteration=iteration,
                epoch=ppo_epoch,
                num_epochs=config.epochs_per_iteration,
                step_counter=global_step,
                get_epsilon_fn=(
                    (
                        lambda: (
                            privacy_engine_actor.get_epsilon(  # type: ignore[misc]
                                config.dp.actor.target_delta
                            ),
                            privacy_engine_critic.get_epsilon(  # type: ignore[misc]
                                config.dp.critic.target_delta
                            ),
                        )
                    )
                    if enable_dp
                    else lambda: (0.0, 0.0)
                ),
                enable_dp=enable_dp,
            )
            iteration_loss += epoch_loss

            del dataloader
            clean_memory()

        iteration_loss /= config.epochs_per_iteration

        # Reload ref & RM to GPU before eval / next rollout
        ppo_system.reload_inference_models()

        actor_scheduler.step()
        critic_scheduler.step()

        log_iter: dict = {
            "iteration/loss": iteration_loss,
            "train/lr": actor_optim.param_groups[0]["lr"],
            "iteration": iteration,
        }

        print(f"\n[Iteration {iteration}] Summary:")
        print(f"  Loss: {iteration_loss:.4f}")

        if enable_dp:
            actor_epsilon = privacy_engine_actor.get_epsilon(  # type: ignore[misc]
                config.dp.actor.target_delta
            )  # type: ignore[union-attr]
            critic_epsilon = privacy_engine_critic.get_epsilon(  # type: ignore[union-attr]
                config.dp.critic.target_delta
            )
            log_iter["iteration/actor_epsilon"] = actor_epsilon
            log_iter["iteration/critic_epsilon"] = critic_epsilon
            print(f"  Actor eps: {actor_epsilon:.2f}")
            print(f"  Critic eps: {critic_epsilon:.2f}")

        wandb.log(log_iter)

        if eval_prompts:
            eval_reward = evaluate(
                model=ppo_system,
                tokenizer=tokenizer,
                prompts=eval_prompts,
                batch_size=config.batch_size,
                max_new_tokens=config.max_new_tokens,
                device=device,
                iteration=iteration,
            )
            print(f"  Eval Avg Reward: {eval_reward:.4f}")
            clean_memory()

        save_dir = f"{config.output_dir}/ppo_iteration_{iteration}"
        save_ppo_model(ppo_system.actor, tokenizer, save_dir)
        print(f"  Saved checkpoint to {save_dir}")

        del rollouts, dataset, all_rewards
        clean_memory()

    wandb.finish()
    print("Done")


if __name__ == "__main__":
    main()
