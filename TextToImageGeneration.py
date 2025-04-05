from diffusers import StableDiffusionPipeline
import torch

model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Or "cpu" if no GPU (but slower)

prompt = "A Sunrise"
image = pipe(prompt).images[0]
image.save("output.png")
