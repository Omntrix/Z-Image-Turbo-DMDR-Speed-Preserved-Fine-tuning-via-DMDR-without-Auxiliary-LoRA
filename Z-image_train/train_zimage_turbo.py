
import argparse
import logging
import math
import os
from pathlib import Path

# Set env before importing torch components
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

# Hugging Face & Diffusers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModel
from diffusers import (
    AutoencoderKL, 
    FlowMatchEulerDiscreteScheduler, 
    ZImageTransformer2DModel, 
    ZImagePipeline
)
import wandb

logger = get_logger(__name__)

# ------------------------------------------------------------
# Dataset 
# ------------------------------------------------------------

class ZImageDataset(Dataset):
    def __init__(self, data_dir, tokenizer, resolution=1024, max_length=512):
        self.data_dir = Path(data_dir)
        self.image_paths = [
            p for p in self.data_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
        ]
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load Image
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)

        # Load Text
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            prompt = txt_path.read_text(encoding="utf-8").strip()
        else:
            prompt = "a high quality image"

        # Note: Using basic tokenization only, no chat template applied here.
        enc = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": enc.input_ids[0],
            "attention_mask": enc.attention_mask[0],
        }

def collate_fn(examples):
    return {
        "pixel_values": torch.stack([e["pixel_values"] for e in examples]),
        "input_ids": torch.stack([e["input_ids"] for e in examples]),
        "attention_mask": torch.stack([e["attention_mask"] for e in examples]),
    }

# ------------------------------------------------------------
# Arguments
# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Z-Image LoRA Training with SFT + DMD")

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./zimage_lora_out")
    
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=2000)
    
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    
    parser.add_argument("--lambda_sft", type=float, default=1.0, help="Weight for Supervised Fine-Tuning loss")
    parser.add_argument("--lambda_dmdr", type=float, default=1.0, help="Weight for DMD (Distillation) loss")
    parser.add_argument("--dmd_cfg", type=float, default=3.0, help="CFG scale used for DMD target calculation")

    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation_steps", type=int, default=50)
    parser.add_argument("--checkpointing_steps", type=int, default=100, help="Save weights every X steps")
    parser.add_argument("--use_wandb", action="store_true")

    return parser.parse_args()

# ------------------------------------------------------------
# Validation Logic
# ------------------------------------------------------------

def run_validation(accelerator, model, vae, text_encoder, tokenizer, scheduler, args, step):
    """Runs a validation inference and logs to WandB."""
    logger.info(f"Running validation at step {step}...")
    
    # Unwrap model for pipeline
    model = accelerator.unwrap_model(model)

    pipe = ZImagePipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=model,
        scheduler=scheduler
    )

    pipe = pipe.to(accelerator.device)
    pipe.set_progress_bar_config(disable=True)

    prompt = (
        "In a classical Suzhou-style courtyard, a Chinese woman in a dark green cheongsam "
        "sits on a stone bench, bamboo shadows swaying behind her, a gentle breeze ruffling "
        "her silk dress. Her expression is reserved and elegant, the sidelight illuminating "
        "her face. A close-up shot delicately reveals her gentle yet resilient nature."
    )

    # Use a fixed generator for consistency
    generator = torch.Generator(device=accelerator.device).manual_seed(42)

    with torch.no_grad():
        images = pipe(
            prompt=prompt,
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=9, # Turbo usually needs fewer steps
            guidance_scale=0.0,    # Guidance often 0 for Turbo/distilled models
            generator=generator,
            output_type="pil"
        ).images

    if args.use_wandb:
        wandb.log(
            {"validation": [wandb.Image(images[0], caption=f"step {step}")]}
        )
    
    # Cleanup to save VRAM
    del pipe
    torch.cuda.empty_cache()

# ------------------------------------------------------------
# Main Training Loop
# ------------------------------------------------------------

def main():
    args = parse_args()

    # 1. Accelerator Setup
    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs")
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=project_config,
        log_with="wandb" if args.use_wandb else None
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process and args.use_wandb:
        accelerator.init_trackers("zimage_lora_dmdr", vars(args))

    if args.seed is not None:
        set_seed(args.seed)

    # Determine weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 2. Load Models
    logger.info("Loading models...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", trust_remote_code=True
    )

    text_encoder = AutoModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", trust_remote_code=True, torch_dtype=weight_dtype
    )
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype
    )
    vae.requires_grad_(False)
    vae.eval()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # Teacher Model (Frozen)
    teacher = ZImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=weight_dtype
    )
    teacher.requires_grad_(False)
    teacher.eval()

    # Student Model (Trainable via LoRA)
    student = ZImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=weight_dtype
    )
    student.requires_grad_(False)

    # 3. LoRA Setup
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "w1", "w2", "w3"],
        init_lora_weights="gaussian",
    )
    student = get_peft_model(student, lora_config)
    
    if accelerator.is_main_process:
        student.print_trainable_parameters()

    # 4. Optimizer
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-2
    )

    # 5. Dataset & Dataloader
    dataset = ZImageDataset(
        args.data_dir,
        tokenizer=tokenizer,
        resolution=args.resolution
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # 6. Prepare with Accelerator
    student, optimizer, dataloader = accelerator.prepare(
        student, optimizer, dataloader
    )

    # Move frozen models to device
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    teacher.to(accelerator.device)

    # 7. Training Loop Setup
    global_step = 0
    total_epochs = math.ceil(args.max_train_steps / len(dataloader))
    
    vae_scaling_factor = vae.config.scaling_factor
    vae_shift_factor = getattr(vae.config, "shift_factor", 0.0)

    # Initialize TQDM
    progress_bar = tqdm(
        range(args.max_train_steps), 
        disable=not accelerator.is_local_main_process,
        desc="Training Steps"
    )

    for epoch in range(total_epochs):
        student.train()
        
        for batch in dataloader:
            with accelerator.accumulate(student):
                
                # --- A. Encode Images to Latents ---
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    # Apply VAE scaling/shift as per Z-Image logic
                    latents = (latents - vae_shift_factor) * vae_scaling_factor

                bsz, ch, h, w = latents.shape
                # Pixel unshuffle if using 4 channels (common in SD3-like/Z-Image architectures)
                if ch == 4:
                    latents = F.pixel_unshuffle(latents, downscale_factor=2)

                bsz = latents.shape[0]

                # --- B. Flow Matching Noise & Time Schedule ---
                noise = torch.randn_like(latents)
                
                # Sigmoid time schedule
                t = torch.sigmoid(torch.randn((bsz,), device=latents.device, dtype=weight_dtype))
                t_view = t.view(bsz, 1, 1, 1)

                # Flow Matching interpolation: t=0 is noise, t=1 is data
                x_t = (1.0 - t_view) * noise + t_view * latents

                # --- C. Encode Text ---
                with torch.no_grad():
                    te_out = text_encoder(
                        input_ids=batch["input_ids"].to(accelerator.device),
                        attention_mask=batch["attention_mask"].to(accelerator.device),
                        output_hidden_states=True,
                    )
                    # Take the second to last hidden layer
                    hidden = te_out.hidden_states[-2]

                prompt_embeds_list = []
                for i in range(bsz):
                    mask = batch["attention_mask"][i].bool()
                    prompt_embeds_list.append(hidden[i][mask].to(dtype=weight_dtype))

                # Note: Z-Image transformer expects latents as a list [B, C, H, W] -> list of [1, C, H, W]
                # actually it expects list of [1, C, H, W] for Multi-resolution support
                latent_list = [x_t[i].unsqueeze(1) for i in range(bsz)]

                # --- D. Student Forward Pass ---
                student_out = student(
                    latent_list,
                    t,
                    prompt_embeds_list,
                )[0]
                
                # Reshape student output: List[B] -> [B, C, H, W]
                student_pred = torch.stack(student_out, dim=0).squeeze(2)

                # --- E. Teacher Forward Pass (for DMD Target) ---
                with torch.no_grad():
                    # 1. Conditional Prediction
                    teacher_cond_out = teacher(latent_list, t, prompt_embeds_list)[0]
                    v_cond = torch.stack(teacher_cond_out, dim=0).squeeze(2)

                    # 2. Unconditional Prediction (Empty Prompt)
                    prompt_embeds_list_null = [torch.zeros_like(p) for p in prompt_embeds_list]
                    teacher_uncond_out = teacher(latent_list, t, prompt_embeds_list_null)[0]
                    v_uncond = torch.stack(teacher_uncond_out, dim=0).squeeze(2)

                    # 3. Construct DMD Target
                    # Formula: v_target = v_uncond + s * (v_cond - v_uncond)
                    v_target = v_uncond + args.dmd_cfg * (v_cond - v_uncond)

                # --- F. Loss Calculation ---
                
                # Ground Truth Velocity
                target_v = latents - noise

                # 1. SFT Loss (Regression to GT)
                loss_sft = F.mse_loss(student_pred.float(), target_v.float())

                # 2. DMD Loss (Regression to Teacher Distillation Target)
                loss_dmdr = F.mse_loss(student_pred.float(), v_target.detach().float())

                loss = args.lambda_sft * loss_sft + args.lambda_dmdr * loss_dmdr

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(student.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            # --- Step Update & Logging ---
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Update progress bar description
                progress_bar.set_postfix(
                    loss=loss.item(), 
                    sft=loss_sft.item(), 
                    dmd=loss_dmdr.item()
                )

                if accelerator.is_main_process:
                    accelerator.log(
                        {
                            "loss": loss.item(),
                            "loss_sft": loss_sft.item(),
                            "loss_dmdr": loss_dmdr.item(),
                        },
                        step=global_step
                    )

                # Save Checkpoint
                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    # Save only LoRA weights
                    unwrapped_student = accelerator.unwrap_model(student)
                    unwrapped_student.save_pretrained(save_path)
                    if accelerator.is_main_process:
                        logger.info(f"Saved checkpoint to {save_path}")

                # Run Validation
                if accelerator.is_main_process and global_step % args.validation_steps == 0:
                    run_validation(
                        accelerator, student, vae, text_encoder, 
                        tokenizer, scheduler, args, global_step
                    )

            if global_step >= args.max_train_steps:
                break
        
        if global_step >= args.max_train_steps:
            break

    # --- Training Finished ---
    if accelerator.is_main_process:
        logger.info("Training finished. Saving final model...")
        unwrapped = accelerator.unwrap_model(student)
        unwrapped.save_pretrained(args.output_dir)

    accelerator.end_training()

if __name__ == "__main__":
    main()