"""Inference script for BiSLW.

Loads a trained model checkpoint, embeds a random watermark in an input image,
applies optional attacks, and extracts the decoded watermark to compute recovery accuracy.
"""

import argparse
import os
import sys
from typing import Optional

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.latent_split import LatentSplitter
from models.recombination import LatentRecombiner
from models.vae_wrapper import VAEWrapper
from models.watermark_decoder import WatermarkDecoder
from models.watermark_encoder import WatermarkEncoder


def load_image(path: str, size: int = 256, device: str = 'cpu') -> torch.Tensor:
    """Loads, center-crops, and resizes an image to normalized tensor range [-1, 1].

    Args:
        path (str): File path to input image.
        size (int): Targeted square size of the image.
        device (str): Destination torch device.

    Returns:
        torch.Tensor: Normalized image tensor of shape (1, C, H, W).
    """
    img = Image.open(path).convert('RGB')
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim))
    img = img.resize((size, size), Image.LANCZOS)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor * 2 - 1
    return img_tensor.to(device)


def save_image(tensor: torch.Tensor, path: str):
    """Converts a normalized image tensor back to file format and saves it.

    Args:
        tensor (torch.Tensor): Image tensor of shape (1, C, H, W) in range [-1, 1].
        path (str): Destination output file path.
    """
    img_np = ((tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_np).save(path)


def jpeg_attack(images: torch.Tensor, quality: int = 70) -> torch.Tensor:
    """Applies a differentiable downsampling/upsampling JPEG approximation.

    Args:
        images (torch.Tensor): Input images of shape (B, C, H, W).
        quality (int): Simulated quality factor (1-100).

    Returns:
        torch.Tensor: Sim-attacked image tensor.
    """
    scale_factor = max(0.3, quality / 100)
    B, C, H, W = images.shape
    h_small = max(8, int(H * scale_factor))
    w_small = max(8, int(W * scale_factor))
    down = F.interpolate(images, size=(h_small, w_small), mode='bilinear', align_corners=False)
    up = F.interpolate(down, size=(H, W), mode='bilinear', align_corners=False)
    blend = quality / 100
    return blend * images + (1.0 - blend) * up


def main():
    """Main execution entrypoint for CLI inference run."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, default='models/BestModel/best.pt', help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='demo/output', help='Output directory')
    parser.add_argument('--attack', type=str, default='jpeg', choices=['none', 'jpeg', 'noise', 'blur', 'resize', 'crop', 'rotate'], help='Attack type')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load VAE
    vae = VAEWrapper().to(device)

    # 2. Load Checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    w_dim = ckpt.get('config', {}).get('w_dim', 32)
    alpha_l = ckpt.get('alpha_l', 0.02)
    alpha_h = ckpt.get('alpha_h', 0.01)

    splitter = LatentSplitter(mode='dct').to(device)
    recombiner = LatentRecombiner(mode='dct').to(device)
    encoder_l = WatermarkEncoder(watermark_dim=w_dim).to(device)
    encoder_h = WatermarkEncoder(watermark_dim=w_dim).to(device)
    decoder_l = WatermarkDecoder(watermark_dim=w_dim).to(device)
    decoder_h = WatermarkDecoder(watermark_dim=w_dim).to(device)

    encoder_l.load_state_dict(ckpt['encoder_l'])
    encoder_h.load_state_dict(ckpt['encoder_h'])
    decoder_l.load_state_dict(ckpt['decoder_l'])
    decoder_h.load_state_dict(ckpt['decoder_h'])

    for m in [encoder_l, encoder_h, decoder_l, decoder_h]:
        m.eval()

    # 3. Load & Encode Image
    img = load_image(args.image, size=256, device=device)
    z = vae.encode(img)

    # 4. Embed Watermark
    w_true = torch.randn(1, w_dim, device=device)
    w_true = (w_true > 0).float() * 2 - 1

    z_l, z_h = splitter(z)
    z_l_wm = encoder_l(z_l, w_true, alpha=alpha_l)
    z_h_wm = encoder_h(z_h, w_true, alpha=alpha_h)
    z_wm = recombiner(z_l_wm, z_h_wm)

    img_wm = vae.decode(z_wm)
    save_image(img_wm, os.path.join(args.output_dir, 'watermarked.png'))

    # Calculate image PSNR
    mse = F.mse_loss(img_wm, img).item()
    psnr = 10 * np.log10(4.0 / (mse + 1e-10))
    print(f"Embedding successful. PSNR: {psnr:.2f} dB")

    # 5. Apply Attack
    if args.attack == 'jpeg':
        img_att = jpeg_attack(img_wm, 70)
        print("Applied simulated JPEG-70 attack.")
    elif args.attack == 'noise':
        img_att = (img_wm + torch.randn_like(img_wm) * 0.05).clamp(-1, 1)
        print("Applied Gaussian noise attack (std=0.05).")
    elif args.attack == 'blur':
        img_att = F.avg_pool2d(img_wm, 5, stride=1, padding=2)
        print("Applied average blur attack.")
    elif args.attack == 'resize':
        # Resize to 0.5x and back
        B, C, H, W = img_wm.shape
        down = F.interpolate(img_wm, size=(H//2, W//2), mode='bilinear', align_corners=False)
        img_att = F.interpolate(down, size=(H, W), mode='bilinear', align_corners=False)
        print("Applied resize attack (0.5x).")
    elif args.attack == 'crop':
        # Crop 10% from edges and resize back
        B, C, H, W = img_wm.shape
        crop_h, crop_w = int(H * 0.1), int(W * 0.1)
        cropped = img_wm[:, :, crop_h:H-crop_h, crop_w:W-crop_w]
        img_att = F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)
        print("Applied random border crop (10%).")
    elif args.attack == 'rotate':
        # Rotate 10 degrees
        B, C, H, W = img_wm.shape
        angle_rad = 10 * np.pi / 180
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=img_wm.dtype, device=img_wm.device).unsqueeze(0).expand(B, -1, -1)
        grid = F.affine_grid(theta, img_wm.size(), align_corners=False)
        img_att = F.grid_sample(img_wm, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
        print("Applied rotation attack (10 degrees).")
    else:
        img_att = img_wm
        print("No attack applied.")

    save_image(img_att, os.path.join(args.output_dir, 'attacked.png'))

    # 6. Re-encode and Decode Watermark
    z_att = vae.encode(img_att)
    z_att_l, z_att_h = splitter(z_att)
    w_pred = (decoder_l(z_att_l) + decoder_h(z_att_h)) / 2

    # Calculate Bit Accuracy
    bits_true = (w_true > 0).float()
    bits_pred = (w_pred > 0).float()
    bit_acc = (bits_true == bits_pred).float().mean().item()
    print(f"Decoded Watermark Bit Accuracy: {bit_acc * 100:.2f}%")


if __name__ == '__main__':
    main()
