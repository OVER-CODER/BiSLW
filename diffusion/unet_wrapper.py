import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

class UNetWrapper(nn.Module):
    """Wrapper around a pre-trained frozen UNet for Stable Diffusion denoising.
    
    Args:
        model_id (str): HuggingFace model repository ID.
        subfolder (str): Folder containing UNet configuration and weights.
    """
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", subfolder="unet"):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder=subfolder)
        self.unet.eval()
        self.unet.requires_grad_(False)

    def forward(self, sample, timestep, encoder_hidden_states):
        """Predicts the noise residual at the given timestep.
        
        Args:
            sample (torch.Tensor): Noisy latent sample tensor.
            timestep (torch.Tensor): Current diffusion timestep tensor.
            encoder_hidden_states (torch.Tensor): Text embedding hidden states.
            
        Returns:
            torch.Tensor: Predicted noise residual tensor.
        """
        return self.unet(sample, timestep, encoder_hidden_states).sample
