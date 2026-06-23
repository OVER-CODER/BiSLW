"""Streamlit web application demo for BiSLW.

Allows users to upload images, inject watermarks in the latent space, simulate various
attacks (JPEG compression, noise, blur), and extract the decoded watermark.
"""

import os
from typing import Optional, Tuple
import sys

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

from models.latent_split import LatentSplitter
from models.recombination import LatentRecombiner
from models.vae_wrapper import VAEWrapper
from models.watermark_decoder import WatermarkDecoder
from models.watermark_encoder import WatermarkEncoder


def load_image(uploaded_file, size: int = 256, device: str = 'cpu') -> Tuple[torch.Tensor, Image.Image]:
    """Loads, center-crops, and resizes an uploaded image to normalized tensor range.

    Args:
        uploaded_file: Streamlit file uploader buffer.
        size (int): Targeted square size of the image.
        device (str): Destination torch device.

    Returns:
        Tuple[torch.Tensor, Image.Image]: Normalized tensor of shape (1, C, H, W)
            and the loaded PIL Image object.
    """
    img = Image.open(uploaded_file).convert('RGB')
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim))
    img = img.resize((size, size), Image.LANCZOS)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor * 2 - 1
    return img_tensor.to(device), img


def get_image_download_link(tensor: torch.Tensor) -> Image.Image:
    """Converts a normalized image tensor back to a PIL image.

    Args:
        tensor (torch.Tensor): Image tensor of shape (1, C, H, W) in range [-1, 1].

    Returns:
        Image.Image: The resulting PIL image.
    """
    img_np = ((tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_np)


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


def main_streamlit():
    """Main Streamlit application interface execution."""
    st.set_page_config(page_title="BiSLW Demo", layout="wide")
    st.title("BiSLW: Bi-Spectral Latent Watermarking Demo")
    st.markdown("Encode and decode watermarks into Stable Diffusion v1.5 VAE latent space with high robustness.")

    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 'cpu')

    @st.cache_resource
    def load_models():
        vae = VAEWrapper().to(device)
        ckpt = torch.load('checkpoints/best.pt', map_location=device, weights_only=False)
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
            
        return vae, splitter, recombiner, encoder_l, encoder_h, decoder_l, decoder_h, w_dim, alpha_l, alpha_h

    try:
        vae, splitter, recombiner, encoder_l, encoder_h, decoder_l, decoder_h, w_dim, alpha_l, alpha_h = load_models()
    except Exception as e:
        st.error(f"Error loading models/checkpoints: {e}. Please run from repository root.")
        return

    st.sidebar.header("Parameters")
    attack_type = st.sidebar.selectbox("Attack Type", ["None", "JPEG Compression", "Gaussian Noise", "Blur"])
    
    if attack_type == "JPEG Compression":
        attack_severity = st.sidebar.slider("Attack Strength / Parameter", 10, 100, 70)
    elif attack_type == "Gaussian Noise":
        attack_severity = st.sidebar.slider("Attack Severity", 0.0, 0.2, 0.05)
    else:
        attack_severity = st.sidebar.slider("Blur Kernel Size", 3, 15, 5, step=2)

    st.header("Step 1: Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        img_tensor, orig_pil = load_image(uploaded_file, size=256, device=device)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(orig_pil, caption="Original Image (Resized to 256x256)", use_container_width=True)

        st.header("Step 2: Watermark Injection")
        w_true = torch.randn(1, w_dim, device=device)
        w_true = (w_true > 0).float() * 2 - 1
        
        z = vae.encode(img_tensor)
        z_l, z_h = splitter(z)
        z_l_wm = encoder_l(z_l, w_true, alpha=alpha_l)
        z_h_wm = encoder_h(z_h, w_true, alpha=alpha_h)
        z_wm = recombiner(z_l_wm, z_h_wm)

        with torch.no_grad():
            img_wm = vae.decode(z_wm)
            
        wm_pil = get_image_download_link(img_wm)
        with col2:
            st.image(wm_pil, caption="Watermarked Image", use_container_width=True)
            
        mse = F.mse_loss(img_wm, img_tensor).item()
        psnr = 10 * np.log10(4.0 / (mse + 1e-10))
        st.success(f"Watermark embedded. Image Quality PSNR: **{psnr:.2f} dB**")

        st.header("Step 3: Attack Simulation")
        if attack_type == "JPEG Compression":
            img_att = jpeg_attack(img_wm, attack_severity)
        elif attack_type == "Gaussian Noise":
            img_att = (img_wm + torch.randn_like(img_wm) * attack_severity).clamp(-1, 1)
        elif attack_type == "Blur":
            img_att = F.avg_pool2d(img_wm, attack_severity, stride=1, padding=attack_severity//2)
        else:
            img_att = img_wm
            
        att_pil = get_image_download_link(img_att)
        with col3:
            st.image(att_pil, caption=f"Attacked Image ({attack_type})", use_container_width=True)

        st.header("Step 4: Watermark Extraction")
        z_att = vae.encode(img_att)
        z_att_l, z_att_h = splitter(z_att)
        w_pred = (decoder_l(z_att_l) + decoder_h(z_att_h)) / 2

        bits_true = (w_true > 0).float()
        bits_pred = (w_pred > 0).float()
        bit_acc = (bits_true == bits_pred).float().mean().item()

        st.metric("Decoded Watermark Bit Accuracy", f"{bit_acc * 100:.2f}%")
        if bit_acc >= 0.75:
            st.balloons()
            st.success("Watermark successfully verified!")
        else:
            st.warning("Watermark could not be verified (Accuracy below 75%).")


def main_cli():
    """Fallback interactive instructions if Streamlit is missing."""
    print("Streamlit is not installed. To run the visual demo, run: pip install streamlit && streamlit run demo/app.py")
    print("Starting CLI interactive demo...")
    print("To run, please execute: python demo/inference.py --image <path_to_image>")


if __name__ == '__main__':
    if HAS_STREAMLIT:
        main_streamlit()
    else:
        main_cli()
