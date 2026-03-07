# PrivMedChat

A privacy-preserving medical dialogue alignment project combining Supervised Fine-Tuning (SFT), Reward Modeling (RM), and Proximal Policy Optimization (PPO) using differentially private training.

## Example Runs

### 1. Generate Dataset

```bash
PYTHONPATH=src/ python -m dataset_builder.generate \
  --deidentified_dir data/deidentified \
  --deidentified_prefix meddialog \
  --base_model meta-llama/Llama-3.1-8B \
  --output_dir ./data/deidentified_output \
  --batch_size 64 \
  --temperature 0.7 \
  --sim_threshold 0.90 \
  --sim_keep_high_frac 0.05 \
  --enable_judge_filter \
  --judge_min_margin 0.20 \
  --judge_min_chosen_score 0.15 \
  --checkpoint_dir ./data/deidentified_output \
  --checkpoint_every 500
```

### 2. Supervised Fine-Tuning (SFT)

```bash
uv run -m sft.train configs/sft/llama3_8b_instruct.yaml
```

**Multi-GPU (DDP):**

```bash
uv run python -m torch.distributed.run --standalone --nproc_per_node=$NGPUS -m sft.train configs/sft/llama3_8b_instruct.yaml
```

### 3. Train Reward Model (RM)

```bash
uv run -m reward_model.train configs/reward_model/llama3_8b_instruct.yaml
```

### 4. Proximal Policy Optimization (PPO)

```bash
uv run -m ppo.train configs/ppo/llama3_8b_instruct_ep5.yaml
```
