import torch
import torch.optim as optim
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, config, models, attacks, losses, dataloader, device):
        self.config = config
        self.models = models # dict of models
        self.attacks = attacks # list of attacks
        self.losses = losses
        self.dataloader = dataloader
        self.device = device
        
        # Alpha values
        self.alpha_l = config.get('alpha_l', 0.3)
        self.alpha_h = config.get('alpha_h', 0.15)
        
        # Optimizers
        # We optimize Encoders and Decoders.
        # VAE and UNet are frozen.
        params = list(models['encoder_l'].parameters()) + \
                 list(models['encoder_h'].parameters()) + \
                 list(models['decoder_l'].parameters()) + \
                 list(models['decoder_h'].parameters())
                 
        if 'splitter' in models and isinstance(models['splitter'], torch.nn.Module):
             # If learned splitter has params
             params += list(models['splitter'].parameters())
        if 'recombiner' in models and isinstance(models['recombiner'], torch.nn.Module):
             params += list(models['recombiner'].parameters())
             
        self.optimizer = optim.AdamW(params, lr=config['lr'])
        
    def set_train_mode(self):
        """Set models to training mode."""
        self.models['encoder_l'].train()
        self.models['encoder_h'].train()
        self.models['decoder_l'].train()
        self.models['decoder_h'].train()
        # VAE stays in eval mode (frozen)
        self.models['vae'].eval()
        
    def train_epoch(self, epoch):
        self.set_train_mode()
        
        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            if isinstance(batch, (list, tuple)):
                images = batch[0].to(self.device)
            else:
                images = batch.to(self.device)
            B = images.shape[0]
            
            # 1. Encode to Latent
            z = self.models['vae'].encode(images)
            
            # 2. Split Latent
            z_low, z_high = self.models['splitter'](z)
            
            # 3. Generate Watermark
            w = torch.randn(B, self.config['w_dim'], device=self.device)
            # Normalize? "Use a continuous identity embedding".
            # Usually good to normalize to unit sphere or keep as gaussian.
            
            # 4. Inject Watermark
            z_low_wm = self.models['encoder_l'](z_low, w, alpha=self.alpha_l)
            z_high_wm = self.models['encoder_h'](z_high, w, alpha=self.alpha_h)
            # Note: Using same encoder network for both? Or separate?
            # Prompt says: "Implement two independent latent watermark encoders: Encoder_L ... Encoder_H"
            # My current code uses one class `WatermarkEncoder`. I should instantiate two instances.
            # I will fix this in `train.py` where I instantiate models.
            # Here I assume `models['encoder_l']` and `models['encoder_h']`.
            
            # Let's correct the assumption:
            z_low_wm = self.models['encoder_l'](z_low, w, alpha=self.alpha_l)
            z_high_wm = self.models['encoder_h'](z_high, w, alpha=self.alpha_h)
            
            # 5. Recombine
            z_wm = self.models['recombiner'](z_low_wm, z_high_wm)
            
            # 6. Decode Watermark (Clean)
            # Need to split again to decode?
            # "Two corresponding latent-frequency decoders recover the watermark from each band"
            # Yes, we split the watermarked latent.
            z_wm_low, z_wm_high = self.models['splitter'](z_wm)
            
            w_pred_l = self.models['decoder_l'](z_wm_low)
            w_pred_h = self.models['decoder_h'](z_wm_high)
            
            # 7. Attacks & Robustness
            # Apply random attack
            attack = self.attacks[torch.randint(0, len(self.attacks), (1,)).item()]
            z_attacked = attack(z_wm)
            
            z_att_low, z_att_high = self.models['splitter'](z_attacked)
            w_pred_rob_l = self.models['decoder_l'](z_att_low)
            w_pred_rob_h = self.models['decoder_h'](z_att_high)
            
            # 8. Loss
            loss, loss_dict = self.losses(w, w_pred_l, w_pred_h, z, z_wm, w_pred_rob_l, w_pred_rob_h)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            pbar.set_postfix(loss=loss.item(), **loss_dict)
            
            # Save images every 50 steps
            if pbar.n % 50 == 0:
                self.save_debug_images(z, z_wm, epoch, pbar.n)

    def save_debug_images(self, z, z_wm, epoch, step):
        with torch.no_grad():
            # Decode a few images
            # Take first 4
            z_sample = z[:4]
            z_wm_sample = z_wm[:4]
            
            img_orig = self.models['vae'].decode(z_sample)
            img_wm = self.models['vae'].decode(z_wm_sample)
            
            # Denormalize
            img_orig = (img_orig + 1) / 2
            img_wm = (img_wm + 1) / 2
            
            comparison = torch.cat([img_orig, img_wm], dim=0)
            os.makedirs('results', exist_ok=True)
            from torchvision.utils import save_image
            save_image(comparison, f'results/e{epoch}_s{step}.png', nrow=4)
        self.set_train_mode()  # Restore train mode after saving
