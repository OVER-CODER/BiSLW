import torch
import torch.nn as nn
from diffusers import AutoencoderKL

class VAEWrapper(nn.Module):
    """
    Wrapper for a frozen VAE (Stable Diffusion compatible).
    Encodes images to latents and decodes latents to images.
    """
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", subfolder="vae"):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder=subfolder)
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.scaling_factor = self.vae.config.scaling_factor

    @torch.no_grad()
    def encode(self, images):
        """
        Encodes images to latents.
        Args:
            images: (B, C, H, W) tensor in range [-1, 1]
        Returns:
            latents: (B, 4, H/8, W/8)
        """
        dist = self.vae.encode(images).latent_dist
        latents = dist.sample() * self.scaling_factor
        return latents

    @torch.no_grad()
    def decode(self, latents):
        """
        Decodes latents to images.
        Args:
            latents: (B, 4, H/8, W/8)
        Returns:
            images: (B, C, H, W) in range [-1, 1]
        """
        latents = 1 / self.scaling_factor * latents
        images = self.vae.decode(latents).sample
        return images
