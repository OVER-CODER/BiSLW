"""Training coordinator module for BiSLW.

Manages execution of epoch loops, loss computation, model optimization,
and intermediate progress visualization exports.
"""

import os
from typing import Dict, List, Union

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """Trainer class for coordinating the watermark network training pipeline."""
    
    def __init__(
        self,
        config: Dict,
        models: Dict[str, torch.nn.Module],
        attacks: List[torch.nn.Module],
        losses: torch.nn.Module,
        dataloader: DataLoader,
        device: Union[str, torch.device]
    ):
        """Initializes the Trainer.

        Args:
            config (Dict): Configuration dictionary containing hyperparameters.
            models (Dict[str, torch.nn.Module]): Active neural networks dictionary.
            attacks (List[torch.nn.Module]): Suite of training attacks.
            losses (torch.nn.Module): Loss module for aggregate loss calculations.
            dataloader (DataLoader): Data loader providing image batch tensors.
            device (Union[str, torch.device]): Computation device target.
        """
        self.config = config
        self.models = models
        self.attacks = attacks
        self.losses = losses
        self.dataloader = dataloader
        self.device = device
        
        self.alpha_l = config.get('alpha_l', 0.3)
        self.alpha_h = config.get('alpha_h', 0.15)
        
        params = list(models['encoder_l'].parameters()) + \
                 list(models['encoder_h'].parameters()) + \
                 list(models['decoder_l'].parameters()) + \
                 list(models['decoder_h'].parameters())
                 
        if 'splitter' in models and isinstance(models['splitter'], torch.nn.Module):
             params += list(models['splitter'].parameters())
        if 'recombiner' in models and isinstance(models['recombiner'], torch.nn.Module):
             params += list(models['recombiner'].parameters())
             
        self.optimizer = optim.AdamW(params, lr=config['lr'])
        
    def set_train_mode(self):
        """Sets target training modules to active train mode."""
        self.models['encoder_l'].train()
        self.models['encoder_h'].train()
        self.models['decoder_l'].train()
        self.models['decoder_h'].train()
        self.models['vae'].eval()
        
    def train_epoch(self, epoch: int):
        """Runs optimization over a single epoch of data.

        Args:
            epoch (int): Index of the current training epoch.
        """
        self.set_train_mode()
        
        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            if isinstance(batch, (list, tuple)):
                images = batch[0].to(self.device)
            else:
                images = batch.to(self.device)
            B = images.shape[0]
            
            z = self.models['vae'].encode(images)
            z_low, z_high = self.models['splitter'](z)
            w = torch.randn(B, self.config['w_dim'], device=self.device)
            
            z_low_wm = self.models['encoder_l'](z_low, w, alpha=self.alpha_l)
            z_high_wm = self.models['encoder_h'](z_high, w, alpha=self.alpha_h)
            z_wm = self.models['recombiner'](z_low_wm, z_high_wm)
            
            z_wm_low, z_wm_high = self.models['splitter'](z_wm)
            w_pred_l = self.models['decoder_l'](z_wm_low)
            w_pred_h = self.models['decoder_h'](z_wm_high)
            
            attack = self.attacks[torch.randint(0, len(self.attacks), (1,)).item()]
            z_attacked = attack(z_wm)
            
            z_att_low, z_att_high = self.models['splitter'](z_attacked)
            w_pred_rob_l = self.models['decoder_l'](z_att_low)
            w_pred_rob_h = self.models['decoder_h'](z_att_high)
            
            loss, loss_dict = self.losses(w, w_pred_l, w_pred_h, z, z_wm, w_pred_rob_l, w_pred_rob_h)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            pbar.set_postfix(loss=loss.item(), **loss_dict)
            
            if pbar.n % 50 == 0:
                self.save_debug_images(z, z_wm, epoch, pbar.n)
                
    def save_debug_images(self, z: torch.Tensor, z_wm: torch.Tensor, epoch: int, step: int):
        """Decodes and saves side-by-side original vs watermarked images for visual checking.

        Args:
            z (torch.Tensor): Original latent tensor.
            z_wm (torch.Tensor): Watermarked latent tensor.
            epoch (int): Current training epoch.
            step (int): Current epoch step.
        """
        with torch.no_grad():
            z_sample = z[:4]
            z_wm_sample = z_wm[:4]
            
            img_orig = self.models['vae'].decode(z_sample)
            img_wm = self.models['vae'].decode(z_wm_sample)
            
            img_orig = (img_orig + 1) / 2
            img_wm = (img_wm + 1) / 2
            
            comparison = torch.cat([img_orig, img_wm], dim=0)
            os.makedirs('results', exist_ok=True)
            from torchvision.utils import save_image
            save_image(comparison, f'results/e{epoch}_s{step}.png', nrow=4)
        self.set_train_mode()
