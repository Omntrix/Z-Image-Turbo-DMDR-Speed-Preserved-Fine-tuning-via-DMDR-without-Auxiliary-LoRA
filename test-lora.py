import torch
from diffusers import ZImagePipeline
from peft import PeftModel

# 1. Load the original pipeline
pipe = ZImagePipeline.from_pretrained(
    "/data/huggingface_cache/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots/your path",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")

# 2. Load LoRA weights
# LoRA weights are usually trained for the transformer module,
# so we wrap pipe.transformer with a PEFT model
lora_model_path = "/data/output"  # This directory should contain adapter_config.json and adapter_model.safetensors

print(f"Loading LoRA from: {lora_model_path}")

# Load the LoRA adapter from the pretrained path using PEFT
# This will automatically read the adapter config and handle key mapping
pipe.transformer = PeftModel.from_pretrained(
    pipe.transformer, 
    lora_model_path,
    adapter_name="default"
)

# If you want to merge the LoRA weights into the base model to speed up inference (optional)
# pipe.transformer = pipe.transformer.merge_and_unload()

# 3. Generate an image
prompt = "realistic photo, professional portrait photography of a man, young adult, natural skin texture, soft studio lighting, shallow depth of field, 85mm lens"  # Your text prompt

image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("example_with_lora.png")
