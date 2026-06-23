"""Upgraded Streamlit Web Application Demo for BiSLW.

Allows users to upload custom images or select preloaded examples, inject watermarks
into Stable Diffusion v1.5 VAE latent space using candidate checkpoints, apply
simulated image-space attacks (JPEG, noise, blur, crop, resize, rotate), and recover
the watermark with real-time visual accuracy verification.
"""

import os
import sys
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
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
from tools.evaluation.metrics import SSIM, PSNR


def load_image_from_path(path: str, size: int = 256, device: str = 'cpu') -> Tuple[torch.Tensor, Image.Image]:
    """Loads and normalizes an image from a local path.

    Args:
        path (str): Local file path.
        size (int): Square size target.
        device (str): Destination torch device.

    Returns:
        Tuple[torch.Tensor, Image.Image]: Normalized tensor [-1, 1] and PIL image.
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
    return img_tensor.to(device), img


def load_image_from_upload(uploaded_file, size: int = 256, device: str = 'cpu') -> Tuple[torch.Tensor, Image.Image]:
    """Loads and normalizes an uploaded file buffer.

    Args:
        uploaded_file: Streamlit file uploader buffer.
        size (int): Square size target.
        device (str): Destination torch device.

    Returns:
        Tuple[torch.Tensor, Image.Image]: Normalized tensor [-1, 1] and PIL image.
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


def get_pil_image(tensor: torch.Tensor) -> Image.Image:
    """Converts a normalized image tensor back to a PIL Image.

    Args:
        tensor (torch.Tensor): Normalized image tensor in range [-1, 1].

    Returns:
        Image.Image: The restored PIL Image.
    """
    img_np = ((tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_np)


def apply_attacks(images: torch.Tensor, attack_type: str, severity: float) -> torch.Tensor:
    """Simulates various image-space attacks for robustness evaluations.

    Args:
        images (torch.Tensor): Watermarked image tensor.
        attack_type (str): Type of attack to simulate.
        severity (float): Parameter intensity of the attack.

    Returns:
        torch.Tensor: Attacked image tensor.
    """
    if attack_type == "JPEG Compression":
        quality = int(severity)
        scale_factor = max(0.3, quality / 100)
        B, C, H, W = images.shape
        h_small = max(8, int(H * scale_factor))
        w_small = max(8, int(W * scale_factor))
        down = F.interpolate(images, size=(h_small, w_small), mode='bilinear', align_corners=False)
        up = F.interpolate(down, size=(H, W), mode='bilinear', align_corners=False)
        blend = quality / 100
        return blend * images + (1.0 - blend) * up
        
    elif attack_type == "Gaussian Noise":
        noise = torch.randn_like(images) * severity
        return torch.clamp(images + noise, -1, 1)
        
    elif attack_type == "Blur":
        kernel_size = int(severity)
        return F.avg_pool2d(images, kernel_size, stride=1, padding=kernel_size // 2)
        
    elif attack_type == "Resize":
        B, C, H, W = images.shape
        h_small = max(8, int(H * severity))
        w_small = max(8, int(W * severity))
        down = F.interpolate(images, size=(h_small, w_small), mode='bilinear', align_corners=False)
        return F.interpolate(down, size=(H, W), mode='bilinear', align_corners=False)
        
    elif attack_type == "Crop":
        B, C, H, W = images.shape
        crop_ratio = severity
        crop_h = int(H * crop_ratio)
        crop_w = int(W * crop_ratio)
        cropped = images[:, :, crop_h:H-crop_h, crop_w:W-crop_w]
        return F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)
        
    elif attack_type == "Rotation":
        B, C, H, W = images.shape
        angle_rad = severity * np.pi / 180
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=images.dtype, device=images.device).unsqueeze(0).expand(B, -1, -1)
        grid = F.affine_grid(theta, images.size(), align_corners=False)
        return F.grid_sample(images, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
        
    return images


def main_streamlit():
    """Main Streamlit showcase application logic."""
    st.set_page_config(page_title="BiSLW Interactive Demo", layout="wide")
    
    st.markdown("""
        <style>
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
            text-align: center;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #1f77b4;
        }
        .metric-label {
            font-size: 14px;
            color: #6c757d;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🛡️ BiSLW: Bi-Spectral Latent Watermarking Showcase")
    st.markdown("Inject and decode robust watermarks into Stable Diffusion v1.5 VAE latent space with dual-band spectral decomposition.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Checkpoint mapping
    checkpoints_map = {
        "Finetuned Efficient (Default)": "models/BestModel/best.pt",
        "Efficient Baseline (High Quality)": "resultmodels/efficient_20260221_164606/best_model.pth",
        "Lightweight (High Robustness)": "resultmodels/attack_fast_20260224_193821/best.pt"
    }
    
    # Sidebar parameter selectors
    st.sidebar.header("⚙️ Configuration")
    ckpt_name = st.sidebar.selectbox("Select Model Checkpoint", list(checkpoints_map.keys()))
    ckpt_path = checkpoints_map[ckpt_name]
    
    # Load model resources dynamically
    @st.cache_resource
    def load_selected_model(path: str):
        vae = VAEWrapper().to(device)
        ckpt = torch.load(path, map_location=device, weights_only=False)
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
        vae, splitter, recombiner, encoder_l, encoder_h, decoder_l, decoder_h, w_dim, alpha_l, alpha_h = load_selected_model(ckpt_path)
    except Exception as e:
        st.error(f"Error loading model checkpoint: {e}. Ensure checkpoints are downloaded or placed correctly.")
        return

    # Image source selector
    st.header("📸 Step 1: Select or Upload Image")
    source_choice = st.radio("Image Source", ["Preloaded Examples", "Upload Your Own"], horizontal=True)
    
    img_tensor = None
    orig_pil = None
    
    if source_choice == "Preloaded Examples":
        examples = {
            "Portrait (Face details)": "demo/examples/portrait.jpg",
            "Landscape (Natural textures)": "demo/examples/landscape.jpg",
            "Animal (Fine hair)": "demo/examples/cat.jpg",
            "City (High frequency structures)": "demo/examples/city.jpg",
            "Graphics/Text (Sharp edges)": "demo/examples/graphics.jpg"
        }
        selected_ex = st.selectbox("Choose a sample image", list(examples.keys()))
        img_tensor, orig_pil = load_image_from_path(examples[selected_ex], size=256, device=device)
    else:
        uploaded_file = st.file_uploader("Upload custom image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            img_tensor, orig_pil = load_image_from_upload(uploaded_file, size=256, device=device)

    # Proceed if image is ready
    if img_tensor is not None:
        st.subheader("Selected Base Image")
        st.image(orig_pil, width=256)
        
        # Injection configuration
        st.header("🔏 Step 2: Watermark Injection")
        st.markdown("Generates a pseudo-random 32-bit signature key vector and embeds it into VAE sub-bands.")
        
        # Setup static signature payload
        if 'payload' not in st.session_state:
            payload = torch.randn(1, w_dim, device=device)
            st.session_state.payload = (payload > 0).float() * 2 - 1
            
        w_true = st.session_state.payload
        
        st.write("Watermark Payload Key (Binary):")
        # Visual representation of binary payload
        payload_bits = ["1" if val > 0 else "0" for val in w_true[0].tolist()]
        st.code(" | ".join(payload_bits), language="text")
        
        # Embed watermark
        z = vae.encode(img_tensor)
        z_l, z_h = splitter(z)
        z_l_wm = encoder_l(z_l, w_true, alpha=alpha_l)
        z_h_wm = encoder_h(z_h, w_true, alpha=alpha_h)
        z_wm = recombiner(z_l_wm, z_h_wm)
        
        with torch.no_grad():
            img_wm = vae.decode(z_wm)
            
        wm_pil = get_pil_image(img_wm)
        
        # Compute metrics
        psnr_metric = PSNR(data_range=2.0)
        ssim_metric = SSIM(data_range=2.0).to(device)
        
        psnr_val = psnr_metric(img_tensor, img_wm).item()
        ssim_val = ssim_metric(img_tensor, img_wm).item()
        
        col_embed_orig, col_embed_wm, col_embed_diff = st.columns(3)
        with col_embed_orig:
            st.image(orig_pil, caption="Original base image", use_container_width=True)
        with col_embed_wm:
            st.image(wm_pil, caption="Embedded (Watermarked)", use_container_width=True)
        with col_embed_diff:
            # Difference heatmap visualization
            diff = np.abs(np.array(orig_pil).astype(float) - np.array(wm_pil).astype(float))
            diff_enhanced = np.clip(diff * 15, 0, 255).astype(np.uint8)
            
            fig, ax = plt.subplots()
            cax = ax.imshow(np.mean(diff_enhanced, axis=2), cmap='hot')
            ax.axis('off')
            fig.colorbar(cax, orientation='horizontal', pad=0.05, label='Perturbation severity (enhanced 15x)')
            st.pyplot(fig)
            plt.close()

        # Attack configuration
        st.header("💥 Step 3: Attack Simulation")
        st.markdown("Select an image manipulation attack below to evaluate how resilient the watermark is.")
        
        attack_type = st.selectbox("Select Attack Pattern", [
            "None", "JPEG Compression", "Gaussian Noise", "Blur", "Resize", "Crop", "Rotation"
        ])
        
        if attack_type == "JPEG Compression":
            severity = st.slider("JPEG Quality Factor", 10, 100, 70)
        elif attack_type == "Gaussian Noise":
            severity = st.slider("Gaussian Noise standard deviation", 0.0, 0.2, 0.05)
        elif attack_type == "Blur":
            severity = st.slider("Average Blur Kernel Size", 3, 15, 5, step=2)
        elif attack_type == "Resize":
            severity = st.slider("Resize Downsample Ratio", 0.25, 1.0, 0.5)
        elif attack_type == "Crop":
            severity = st.slider("Random Border Crop Edge Ratio", 0.0, 0.3, 0.1)
        elif attack_type == "Rotation":
            severity = st.slider("Rotation Degrees", -45, 45, 10)
        else:
            severity = 0.0
            
        img_att = apply_attacks(img_wm, attack_type, severity)
        att_pil = get_pil_image(img_att)
        
        col_att, col_recovery = st.columns(2)
        with col_att:
            st.image(att_pil, caption=f"Attacked Image ({attack_type})", use_container_width=True)

        # Extraction phase
        with col_recovery:
            st.markdown("### Recovered Signature Key")
            z_att = vae.encode(img_att)
            z_att_l, z_att_h = splitter(z_att)
            w_pred = (decoder_l(z_att_l) + decoder_h(z_att_h)) / 2
            
            bits_true = (w_true > 0).float()
            bits_pred = (w_pred > 0).float()
            bit_acc = (bits_true == bits_pred).float().mean().item()
            
            # Show color-coded bit verification
            bit_checks = []
            for t_val, p_val in zip(bits_true[0].tolist(), bits_pred[0].tolist()):
                if t_val == p_val:
                    bit_checks.append("🟩")
                else:
                    bit_checks.append("🟥")
            
            st.write("Bit Match Comparison (🟩 = Match, 🟥 = Mismatch):")
            st.write(" ".join(bit_checks))
            
            st.markdown(f"**Extraction Confidence Levels:**")
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.bar(range(w_dim), w_pred[0].cpu().numpy(), color=['green' if t==p else 'red' for t, p in zip(bits_true[0].tolist(), bits_pred[0].tolist())])
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
            ax.set_ylim(-3, 3)
            ax.set_ylabel("Confidence")
            ax.set_xlabel("Bit Index")
            st.pyplot(fig)
            plt.close()

        # Premium metrics row display
        st.header("📊 Evaluation Summary Metrics")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Embedding Imperceptibility (PSNR)</div>
                    <div class="metric-value">{psnr_val:.2f} dB</div>
                    <div style="font-size: 12px; color: {'green' if psnr_val>=35 else 'orange'};">
                        {'Target (>=35dB) Met' if psnr_val>=35 else 'Noticeable modifications'}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Structural Similarity (SSIM)</div>
                    <div class="metric-value">{ssim_val:.4f}</div>
                    <div style="font-size: 12px; color: {'green' if ssim_val>=0.90 else 'orange'};">
                        {'Target (>=0.90) Met' if ssim_val>=0.90 else 'Slight structural changes'}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        with m_col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Extraction Accuracy</div>
                    <div class="metric-value">{bit_acc*100:.1f}%</div>
                    <div style="font-size: 12px; color: {'green' if bit_acc>=0.75 else 'red'};">
                        {'Verified Robust Signature' if bit_acc>=0.75 else 'Extraction failed (unreliable)'}
                    </div>
                </div>
            """, unsafe_allow_html=True)


def main_cli():
    """Alternative CLI launcher warning if Streamlit is not initialized."""
    print("Streamlit is not installed. To launch the interactive showcase app, run:")
    print("  pip install streamlit")
    print("  PYTHONPATH=. streamlit run demo/app.py")


if __name__ == '__main__':
    if HAS_STREAMLIT:
        main_streamlit()
    else:
        main_cli()
