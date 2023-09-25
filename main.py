import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path()
Image_DIR = BASE_DIR / "images"

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

image_files = list(Image_DIR.glob("*.[jpg, png, jpeg]*"))

image = PIL.Image.open(image_files[0])
image = PIL.ImageOps.exif_transpose(image)
image = image.convert("RGB")

prompt = "turn him into animated cartoon for avatar"
images = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images
plt.imshow(images[0])
