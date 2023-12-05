from diffusion import stable_diffusion
from torchmetrics.functional.multimodal import clip_score
from torchmetrics.image.fid import FrechetInceptionDistance
from functools import partial
from torchvision.transforms import functional as F
import torch
import numpy as np
from skimage.transform import resize
import torchvision.datasets as dset
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
    img_idx = [3,500]
    
    def preprocess_image(img_idx):
        prompts = []
        real_images = np.array([])
        coco_val = dset.CocoCaptions(root = 'mscoco/val2017',
                        annFile = 'mscoco/annotations/captions_val2017.json')
        for idx in img_idx:
            prompts.append(coco_val[idx][1][2])
            img = resize(np.array(coco_val[idx][0]),(512,512,3))
            img = Image.fromarray((img * 255).astype(np.uint8))
            img = np.expand_dims(np.array(img),axis=0)
            if real_images.size == 0:
                real_images = img
            else:
                real_images = np.vstack((real_images,np.array(img)))

        real_images = torch.from_numpy(real_images).permute(0, 3, 1, 2)
        return prompts,real_images
    prompts,real_images = preprocess_image(img_idx)
    print(real_images.shape)
    fake_images = np.array([])
    for prompt in prompts:
        if fake_images.size == 0:
            fake_images = stable_diffusion(prompt=prompt)
        else:
    
            fake_images = np.vstack((fake_images,stable_diffusion(prompt=prompt)))
    for i in range(fake_images.shape[0]):
        image_data = fake_images[i]  
        image = Image.fromarray(image_data)
        image.save(f'img_by_original/image_{i + 1}.png')
            
    fake_images = torch.from_numpy(fake_images).permute(0, 3, 1, 2)
    

    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)

    print(f"FID: {float(fid.compute())}")





if __name__ == "__main__":
    get_fid()

    
