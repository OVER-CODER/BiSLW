import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from models.latent_split import LatentSplitter
from models.recombination import LatentRecombiner

def load_vae(device):
    """Load VAE for encoding/decoding images."""
    print("Loading VAE...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        subfolder='vae',
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae

def encode_image(vae, img, scaling_factor=0.18215):
    """Encode image to latent."""
    with torch.no_grad():
        latent = vae.encode(img).latent_dist.mean
        return latent * scaling_factor

def decode_image(vae, latent, scaling_factor=0.18215):
    """Decode latent back to image."""
    with torch.no_grad():
        latent = latent / scaling_factor
        image = vae.decode(latent).sample
        # Convert back to [0, 1] range
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

def load_sample_image(sample_dir, size=512, device='cpu'):
    """Load a sample image."""
    extensions = ['*.png', '*.jpg', '*.jpeg']
    image_files = []
    for ext in extensions:
        image_files.extend(list(Path(sample_dir).glob(ext)))
    
    if not image_files:
        raise ValueError(f"No images found in {sample_dir}")
    
    img_path = sorted(image_files)[0] # Taking the first image
    print(f"Loading: {img_path.name}")
    
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim))
    img = img.resize((size, size), Image.LANCZOS)
    
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
    img_tensor = img_tensor * 2 - 1  # [-1, 1]
    
    return img_tensor.unsqueeze(0).to(device)

def tensor_to_pil(tensor):
    """Convert tensor image to PIL Image."""
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

def visualize_frequency_perturbation(output_dir='results/frequency_perturbation', seed=42):
    """Generate frequency perturbation visualization."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load VAE and tools
    vae = load_vae(device)
    splitter = LatentSplitter(mode='dct').to(device)
    recombiner = LatentRecombiner(mode='dct').to(device)
    
    # Load sample image
    sample_path = Path(PROJECT_ROOT) / 'sample_images/hq'
    image = load_sample_image(sample_path, device=device)
    
    # Process original
    z = encode_image(vae, image)
    z_low, z_high = splitter(z)
    
    # Create mask to know where low/high regions are
    mask_low = (z_low != 0).float()
    mask_high = (z_high != 0).float()
    
    # Generate random noise perturbations
    # Low frequency changes need large perturbations to be visible
    # High frequency changes can be smaller but need to be visible as texture/noise
    noise = torch.randn_like(z)
    
    # Calculate std of components to scale noise appropriately
    std_low = z_low[z_low != 0].std()
    std_high = z_high[z_high != 0].std()
    
    # Scale factors: We want visible but not completely destroying perturbations
    # Use SMALL controlled perturbations to keep images clean and recognizable
    alpha_low = std_low * 0.1  # Small perturbation for low freq
    alpha_high = std_high * 0.5 # Reduced small perturbation for high freq
    
    # Apply perturbations
    z_low_perturbed = z_low + noise * mask_low * alpha_low
    z_high_perturbed = z_high + noise * mask_high * alpha_high
    
    # Recombine and decode
    z_full_orig = recombiner(z_low, z_high) # Should be identical to z
    z_full_low_pert = recombiner(z_low_perturbed, z_high)
    z_full_high_pert = recombiner(z_low, z_high_perturbed)
    
    img_orig_recon = decode_image(vae, z_full_orig)
    img_low_pert = decode_image(vae, z_full_low_pert)
    img_high_pert = decode_image(vae, z_full_high_pert)
    
    # Convert to PIL for saving/plotting
    pil_orig = tensor_to_pil(img_orig_recon)
    pil_low = tensor_to_pil(img_low_pert)
    pil_high = tensor_to_pil(img_high_pert)
    
    # Save individual images
    pil_orig.save(os.path.join(output_dir, 'original.png'))
    pil_low.save(os.path.join(output_dir, 'low_freq_perturbed.png'))
    pil_high.save(os.path.join(output_dir, 'high_freq_perturbed.png'))
    
    # Create side-by-side plot
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 14})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(pil_orig)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(pil_low)
    axes[1].set_title('Low-freq perturbed\n(semantic/layout changes)')
    axes[1].axis('off')
    
    axes[2].imshow(pil_high)
    axes[2].set_title('High-freq perturbed\n(texture/noise/detail changes)')
    axes[2].axis('off')
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'frequency_perturbation_comparison.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'frequency_perturbation_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization to {output_dir}")

if __name__ == '__main__':
    visualize_frequency_perturbation()
