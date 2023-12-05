import torch
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler
from tqdm.auto import tqdm
from torch import autocast
import time

def load_quant_unet():
    codebooks = torch.load('codebooks.pth')
    unet = torch.load('quantized_model.pth')
    for name, param in unet.named_parameters():
        if 'weight' in name and param.dim() > 1:
            centroids = codebooks[name+'_codebook'].detach().numpy()
            indices = param.detach().numpy()
            # print(centroids)
            # print(centroids[indices].shape)
            quantized_tensor = torch.tensor(centroids[indices],dtype=torch.float32)
            param.data = (quantized_tensor.view_as(param))
            # print(param)
    return unet


def stable_diffusion(prompt):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  # 1. Load the autoencoder model which will be used to decode the latents into image space. 
  vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

  # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
  tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
  text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

  # 3. The UNet model for generating the latents.
  unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

  # unet = load_quant_unet()
  # unet = torch.load('gmm_quantized_unet_w_outlier_32.pth')  
  scheduler = EulerDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

   
  height = 512                        # default height of Stable Diffusion
  width = 512                         # default width of Stable Diffusion
  num_inference_steps = 50            # Number of denoising steps
  guidance_scale = 7.5                # Scale for classifier-free guidance
  generator = torch.manual_seed(32)   # Seed generator to create the inital latent noise
  batch_size = 1
  text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
  
  vae = vae.to(device)
  text_encoder = text_encoder.to(device)
  unet = unet.to(device)
  
#   embed_time0 = time.time()
  with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

  max_length = text_input.input_ids.shape[-1]
  uncond_input = tokenizer(
      [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
  )
  with torch.no_grad():
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]   
  text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
#   if torch.cuda.is_available(): 
#     torch.cuda.current_stream().synchronize()
#   embed_time1 = time.time()
#   print('embed_time',embed_time1-embed_time0)
  latents = torch.randn(
    (batch_size, unet.in_channels, height // 8, width // 8),
    generator=generator
  )
  latents = latents.to(device)

  scheduler.set_timesteps(num_inference_steps)

  latents = latents * scheduler.init_noise_sigma
#   unet_timesum = 0
  for t in tqdm(scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    # predict the noise residual
    with torch.no_grad():
    #   unet_time0 = time.time()
    #   print(latent_model_input.shape)
    #   print(t.shape)
    #   print(text_embeddings.shape)
      noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
    #   if torch.cuda.is_available():
    #     torch.cuda.current_stream().synchronize()
    #   unet_time1 = time.time()
    #   unet_timesum = unet_timesum + (unet_time1-unet_time0)
    

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample
#   print('unet_timesum',unet_timesum)
#   print('unet_avgtime',unet_timesum/num_inference_steps)
  # scale and decode the image latents with vae
  latents = 1 / 0.18215 * latents

  with torch.no_grad():
    # vae_time0 = time.time()
    image = vae.decode(latents).sample
    # torch.cuda.current_stream().synchronize()
    # vae_time1 = time.time()
    # print('vae_time',vae_time1-vae_time0)

  image = (image / 2 + 0.5).clamp(0, 1)
  image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
  images = (image * 255).round().astype("uint8")
  # pil_images = [Image.fromarray(image) for image in images]
  # img = pil_images[0]
  # img.save('test.png')

  return images


if __name__ == '__main__':
    stable_diffusion()
    # load_quant_unet()