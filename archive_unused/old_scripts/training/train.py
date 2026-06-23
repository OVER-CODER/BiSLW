import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset
from latent_watermarking.models.vae_wrapper import VAEWrapper
from latent_watermarking.models.latent_split import LatentSplitter
from latent_watermarking.models.watermark_encoder import WatermarkEncoder
from latent_watermarking.models.watermark_decoder import WatermarkDecoder
from latent_watermarking.models.recombination import LatentRecombiner
from latent_watermarking.attacks.latent_noise import LatentNoiseAttack
from latent_watermarking.attacks.jpeg_sim import JpegSimAttack
from latent_watermarking.attacks.resize_crop import ResizeCropAttack
from latent_watermarking.training.losses import WatermarkLosses
from latent_watermarking.training.trainer import Trainer

from torchvision import transforms
from latent_watermarking.training.dataset import MirflickrDataset
from torchvision.utils import save_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='latent_watermarking/configs/default.yaml')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--limit', type=int, default=1500, help='Limit number of images')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        
    print(f"Using device: {device}")
    
    # 1. Initialize Models
    vae = VAEWrapper().to(device)
    splitter = LatentSplitter(mode=config['latent_split']).to(device)
    recombiner = LatentRecombiner(mode=config['latent_split']).to(device)
    
    encoder_l = WatermarkEncoder(watermark_dim=config['w_dim']).to(device)
    encoder_h = WatermarkEncoder(watermark_dim=config['w_dim']).to(device)
    
    decoder_l = WatermarkDecoder(watermark_dim=config['w_dim']).to(device)
    decoder_h = WatermarkDecoder(watermark_dim=config['w_dim']).to(device)
    
    models = {
        'vae': vae,
        'splitter': splitter,
        'recombiner': recombiner,
        'encoder_l': encoder_l,
        'encoder_h': encoder_h,
        'decoder_l': decoder_l,
        'decoder_h': decoder_h
    }
    
    # 2. Initialize Attacks
    attacks = [
        LatentNoiseAttack().to(device),
        JpegSimAttack(vae).to(device),
        ResizeCropAttack(vae).to(device)
    ]
    
    # 3. Initialize Losses
    losses = WatermarkLosses(
        lambda_w=config['lambda_w'],
        lambda_cons=config['lambda_cons'],
        lambda_latent=config['lambda_latent'],
        lambda_robust=config['lambda_robust']
    ).to(device)
    
    # 4. Data
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    dataset_path = 'latent_watermarking/mirflickr'
    if os.path.exists(dataset_path):
        print(f"Loading Mirflickr dataset from {dataset_path} with limit {args.limit}")
        dataset = MirflickrDataset(dataset_path, transform=transform, limit=args.limit)
    else:
        print("Mirflickr dataset not found, using random tensors for debug")
        dummy_data = torch.randn(100, 3, config['image_size'], config['image_size'])
        dataset = TensorDataset(dummy_data)
        
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
        
    # 5. Trainer
    if args.debug:
        config['epochs'] = 1
        
    trainer = Trainer(config, models, attacks, losses, dataloader, device)
    
    # 6. Train
    print("Starting training...")
    for epoch in range(config['epochs']):
        trainer.train_epoch(epoch)
        
        # Save sample watermarked images
        if epoch % 1 == 0: # Save every epoch
            with torch.no_grad():
                # Get a batch
                try:
                    batch = next(iter(dataloader))
                    if isinstance(batch, list):
                        images = batch[0].to(device)
                    else:
                        images = batch.to(device)
                    
                    if images.dim() == 3:
                        images = images.unsqueeze(0)
                    
                    # Encode and Watermark
                    z = vae.encode(images)
                    z_low, z_high = splitter(z)
                    w = torch.randn(images.shape[0], config['w_dim'], device=device)
                    z_low_wm = encoder_l(z_low, w, alpha=config['alpha_l'])
                    z_high_wm = encoder_h(z_high, w, alpha=config['alpha_h'])
                    z_wm = recombiner(z_low_wm, z_high_wm)
                    
                    # Decode to image
                    images_wm = vae.decode(z_wm)
                    
                    # Save comparison
                    comparison = torch.cat([images, images_wm], dim=0)
                    os.makedirs('results', exist_ok=True)
                    # Denormalize for saving
                    comparison = (comparison + 1) / 2
                    save_image(comparison, f'results/epoch_{epoch}.png')
                except Exception as e:
                    print(f"Error saving images: {e}")

    # Save models
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'encoder_l': encoder_l.state_dict(),
        'encoder_h': encoder_h.state_dict(),
        'decoder_l': decoder_l.state_dict(),
        'decoder_h': decoder_h.state_dict(),
    }, 'checkpoints/model.pth')
    print("Training complete. Model saved.")

if __name__ == '__main__':
    main()
