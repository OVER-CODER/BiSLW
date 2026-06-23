import torch
import torch.nn as nn
from diffusers import AutoencoderKL

class VAEWrapper(nn.Module):
    """Wrapper around a pre-trained frozen Variational Autoencoder (VAE) for Stable Diffusion.
    
    Handles image encoding to latent space and reconstruction decoding.
    
    Args:
        model_id (str): HuggingFace model repository ID.
        subfolder (str): Folder containing VAE configuration and weights.
    """
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", subfolder="vae"):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder=subfolder)
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.scaling_factor = self.vae.config.scaling_factor

    @torch.no_grad()
    def encode(self, images):
        """Encodes spatial images into VAE latents.
        
        Args:
            images (torch.Tensor): Images tensor of shape (B, 3, H, W) in range [-1, 1].
            
        Returns:
            torch.Tensor: Scaled latent representations of shape (B, 4, H/8, W/8).
        """
        dist = self.vae.encode(images).latent_dist
        latents = dist.sample() * self.scaling_factor
        return latents

    @torch.no_grad()
    def decode(self, latents):
        """Decodes latent representations back to reconstructed images.
        
        Args:
            latents (torch.Tensor): Latent tensor of shape (B, 4, H/8, W/8).
            
        Returns:
            torch.Tensor: Reconstructed images of shape (B, 3, H, W) in range [-1, 1].
        """
        latents = 1 / self.scaling_factor * latents
        images = self.vae.decode(latents).sample
        return images
