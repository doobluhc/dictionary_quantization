from diffusion import stable_diffusion
from torchmetrics.functional.multimodal import clip_score
from torchmetrics.image.fid import FrechetInceptionDistance
from functools import partial
from torchvision.transforms import functional as F
import torch
import numpy as np
from zipfile import ZipFile
import requests
from PIL import Image
import os

def get_clip_score():
    prompts = [
        "a photo of an astronaut riding a horse on mars",
        "A high tech solarpunk utopia in the Amazon rainforest",
        "A pikachu fine dining with a view to the Eiffel Tower",
        "A mecha robot in a favela in expressionist style",
        "an insect robot preparing a delicious meal",
        "A small cabin on top of a snowy mountain in the style of Disney, artstation",
    ]
    images = np.array([])
    print(images.size)
    for prompt in prompts:
        if images.size == 0:
            images = stable_diffusion(prompt=prompt)
        else:
            images = np.vstack((images,stable_diffusion(prompt=prompt)))
        print(images.shape)
       

    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
    
    def calculate_clip_score(images, prompts):
        # images_int = (images * 255).astype("uint8")
        clip_score = clip_score_fn(torch.from_numpy(images).permute(0, 3, 1, 2), prompts).detach()
        return round(float(clip_score), 4)

    sd_clip_score = calculate_clip_score(images, prompts)
    print(f"CLIP score: {sd_clip_score}")

def get_fid():
    def preprocess_image(image):
        image = torch.tensor(image).unsqueeze(0)
        image = image.permute(0, 3, 1, 2) / 255.0
        return F.center_crop(image, (256, 256))
    
    dataset_path = "sample-imagenet-images"
    image_paths = sorted([os.path.join(dataset_path, x) for x in os.listdir(dataset_path)])
    real_images = [np.array(Image.open(path).convert("RGB")) for path in image_paths]
    real_images = torch.cat([preprocess_image(image) for image in real_images])
    print(real_images.shape)
    prompts = [
        "cassette player",
        "chainsaw"]
    # prompts = [
    #     "cassette player",
    #     "chainsaw",
    #     "chainsaw",
    #     "church",
    #     "gas pump",
    #     "gas pump",
    #     "gas pump",
    #     "parachute",
    #     "parachute",
    #     "tench",
    # ]

    fake_images = np.array([])
    for prompt in prompts:
        if fake_images.size == 0:
            fake_images = stable_diffusion(prompt=prompt)
        else:
            fake_images = np.vstack((fake_images,stable_diffusion(prompt=prompt)))
    fake_images = torch.from_numpy(fake_images).permute(0, 3, 1, 2)
    print(fake_images.shape)

    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)

    print(f"FID: {float(fid.compute())}")





if __name__ == "__main__":
    get_fid()

    
