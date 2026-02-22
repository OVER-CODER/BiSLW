import torch
import numpy as np
from tqdm import tqdm

class DDIMSampler:
    """
    DDIM Sampler for Stable Diffusion.
    """
    def __init__(self, unet, scheduler, num_inference_steps=50):
        self.unet = unet
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps

    @torch.no_grad()
    def sample(self, latents, prompt_embeds, guidance_scale=7.5):
        """
        Generates images from latents using DDIM.
        Args:
            latents: (B, 4, H, W) Initial latents (watermarked)
            prompt_embeds: (B, L, D) Text embeddings
            guidance_scale: Classifier-free guidance scale
        Returns:
            final_latents: (B, 4, H, W)
        """
        self.scheduler.set_timesteps(self.num_inference_steps)
        timesteps = self.scheduler.timesteps
        
        # If latents are provided, we might be doing image-to-image or just starting generation from them.
        # "Watermarked latent must be fully generative" -> This implies we use the watermarked latent as the starting point (z_T) or as the structural guidance?
        # Usually, watermarking for diffusion means we watermark the *initial noise* or we watermark the *image* and then diffuse.
        # "Robust to: diffusion regeneration (DDIM / DDPM)"
        # If we watermark the latent z, and then use it as z_T for generation, the watermark must survive the denoising process.
        # However, standard generation starts from Gaussian noise.
        # If we watermark z_T, that IS the Gaussian noise (plus watermark).
        # So yes, we start with `latents` as the initial state.
        
        for t in tqdm(timesteps, desc="DDIM Sampling", leave=False):
            # Expand for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds)
            
            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous noisy sample x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
        return latents
