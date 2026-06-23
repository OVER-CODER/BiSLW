import torch
import numpy as np
from tqdm import tqdm

class DDIMSampler:
    """DDIM Sampler for generating images from latent representations.
    
    Args:
        unet (nn.Module): Frozen UNet wrapper model.
        scheduler (Scheduler): Stable Diffusion scheduler.
        num_inference_steps (int): Number of denoising inference steps.
    """
    def __init__(self, unet, scheduler, num_inference_steps=50):
        self.unet = unet
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps

    @torch.no_grad()
    def sample(self, latents, prompt_embeds, guidance_scale=7.5):
        """Denoises latents using classifier-free guided DDIM sampling.
        
        Args:
            latents (torch.Tensor): Initial starting latent tensor of shape (B, 4, H, W).
            prompt_embeds (torch.Tensor): Text prompt embeddings.
            guidance_scale (float): Scale factor for classifier-free guidance.
            
        Returns:
            torch.Tensor: Denoised latent tensor of shape (B, 4, H, W).
        """
        self.scheduler.set_timesteps(self.num_inference_steps)
        timesteps = self.scheduler.timesteps
        
        for t in tqdm(timesteps, desc="DDIM Denoising", leave=False):
            # Scale inputs for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Denoise noise residual estimation
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds)
            
            # Guidance update
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Denoise step towards t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
        return latents
