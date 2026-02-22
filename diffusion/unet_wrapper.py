import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

class UNetWrapper(nn.Module):
    """
    Wrapper for frozen UNet.
    """
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", subfolder="unet"):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder=subfolder)
        self.unet.eval()
        self.unet.requires_grad_(False)

    def forward(self, sample, timestep, encoder_hidden_states):
        """
        Predicts noise residual.
        """
        return self.unet(sample, timestep, encoder_hidden_states).sample
