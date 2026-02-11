# Z-Image-Turbo-DMDR  
**Speed-Preserved Fine-tuning via DMDR without Auxiliary LoRA**

This project provides a high-fidelity implementation of the **DMDR loss** tailored for **Z-Image-Turbo**.  
We introduce an **Aux-LoRA-Free** training pipeline that enables **Reinforcement Learning** and **SFT / LoRA fine-tuning** *without extra training overhead*, ensuring:

- ✅ crystal-clear image generation  
- ✅ zero inference speed penalty  
- ✅ no auxiliary LoRA branches during training  
- ✅ fully compatible with existing Z-Image-Turbo and Flow-Factory workflows  

---

## ✨ Key Features

- **DMDR loss for Z-Image-Turbo**
  - A drop-in loss module designed specifically for Z-Image-Turbo diffusion / flow models.
- **Aux-LoRA-Free RL & fine-tuning**
  - No extra LoRA heads or auxiliary branches for alignment / RL.
- **Speed-preserved inference**
  - The trained model runs with exactly the same inference graph as the original model.
- **Compatible with Flow-Factory**
  - Reinforcement learning is integrated by directly replacing the GRPO trainer.

---

## 📌 Supported Training Modes

This repository supports the following training modes:

| Training Type | Script | Speed Penalty |
-----|------|------
Reinforcement Learning (GRPO) | `grpo.py` (replacement) | ❌ No
LoRA fine-tuning | `train_zimage_turbo.py` | ❌ No
Full SFT fine-tuning | `train_zimage_turbo_full.py` | ❌ No

All modes are **Aux-LoRA-Free** and keep the original inference structure.

---

## 🧩 Reinforcement Learning with Flow-Factory (GRPO)

You can directly use the GRPO implementation in this repository to replace the official one in Flow-Factory.

### Step 1. Replace the trainer file

Replace the following file in the official [Flow-Factory](https://github.com/X-GenGroup/Flow-Factory) repository.

Target file:`./src/flow_factory/trainers/grpo.py`

Replace it with: `grpo.py` from this repository.


### Step 2. Train with Flow-Factory as usual

After the replacement, you can directly use `z_image_turbo.yaml` from this repository to launch reinforcement learning with Flow-Factory as usual.


## 🧠 LoRA Fine-Tuning (Speed-Preserved)

### Environment Setup

```bash
cd Z-image_train
```
```bash
python -m venv venv
```
```bash
source venv/bin/activate
```
```bash
pip install -r requirements.txt
```
---

You can perform LoRA training with:
```bash
accelerate launch \
  --multi_gpu \
  --num_processes=8 \
  train_zimage_turbo.py \
  --pretrained_model_name_or_path "/data/huggingface_cache/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots/your path" \
  --data_dir "/data/" \
  --output_dir "self-zimage-train/checkpoint" \
  --resolution 1024 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --max_train_steps 2000 \
  --mixed_precision bf16 \
  --lambda_sft 1.0 \
  --lambda_dmdr 0.1 \
  --use_wandb
```
---
## 🧠 Full SFT Fine-Tuning (Speed-Preserved)

For full-parameter supervised fine-tuning (SFT):
```bash
accelerate launch \
  --multi_gpu \
  --num_processes=8 \
  train_zimage_turbo_full.py \
  --pretrained_model_name_or_path "/data/huggingface_cache/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots/your path" \
  --data_dir "/data" \
  --output_dir "/data/self-zimage-train/checkpoint" \
  --resolution 1024 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --max_train_steps 2000 \
  --mixed_precision bf16 \
  --lambda_sft 1.0 \
  --lambda_dmdr 0.1 \
  --use_wandb
```





