# BiSLW Interactive Demo

This directory contains the code to run the interactive web-based and command-line demo of the Bi-Spectral Latent Watermarking (BiSLW) system.

## 1. Requirements

Install Streamlit and other dependencies:
```bash
pip install streamlit torch torchvision pillow diffusers
```

## 2. Running the Interactive Web App

Launch the visual web demo using Streamlit:
```bash
PYTHONPATH=. streamlit run demo/app.py
```
This launches a browser interface where you can:
1. Upload any custom image.
2. Embed a random 32-bit watermark in the SD v1.5 VAE latent space.
3. Apply on-the-fly simulated attacks (JPEG, Gaussian noise, blur) with sliders.
4. Extract the watermark and display the exact bit recovery accuracy metrics.

## 3. Running the CLI Inference Script

Alternatively, you can run single-image inference directly from the command line:
```bash
PYTHONPATH=. python demo/inference.py --image sample_images/hq/hq_market.jpg --attack jpeg
```
Options:
*   `--image`: Path to input image (required)
*   `--checkpoint`: Path to model checkpoint (default: `checkpoints/best.pt`)
*   `--attack`: Type of attack simulation (`none`, `jpeg`, `noise`, `blur`)
*   `--output-dir`: Output directory for generated figures (default: `demo/output`)
