# BiSLW Interactive Demo & Showcase

This directory contains the code to run the interactive graphical demonstration and CLI inference scripts for the Bi-Spectral Latent Watermarking (BiSLW) system.

---

## A. Demo Capabilities

The Streamlit web demo is designed to visually demonstrate the entire embedding, attack, and extraction lifecycle:
1. **Input Image**: Select or upload an image to undergo watermarking.
2. **Watermarked Image**: View the high-fidelity watermarked reconstruction output by the VAE decoder.
3. **Residual Map**: Inspect the pixel-space residual map showing the exact difference (multiplied for visibility) between the original and watermarked image.
4. **Attacked Image**: Apply simulated image distortions on the watermarked image in real time.
5. **Recovered Watermark**: View the comparison between the original 32-bit signature and the reconstructed signature from the low-frequency and high-frequency components.
6. **Metrics**: Read live structural similarity and recovery statistics.

---

## B. Supported Attacks

The demo includes adjustable controls to apply simulated, non-differentiable attacks:
*   **None**: Zero modification baseline.
*   **JPEG Compression**: Reduces visual bandwidth and introduces block artifacts (quality factors from 10 to 100).
*   **Gaussian Noise**: Introduces random high-frequency pixel deviations.
*   **Average / Gaussian Blur**: Dampens high-frequency components.
*   **Bilinear Resize**: Simulates downscaling to a fraction of the native size, then restoring the dimensions.
*   **Center Crop**: Spatial cutting of outer boundary percentages.
*   **Rotation**: Spatial angle rotations (both clockwise and counter-clockwise).

---

## C. Run Instructions

### Launch the Streamlit Web Application

Ensure that all dependencies are installed, then start the web interface:
```bash
PYTHONPATH=. streamlit run demo/app.py
```
By default, the server will start at `http://localhost:8501`.

---

## D. CLI Inference

For headless environments, you can run batch or single-image inference via the command-line interface:
```bash
PYTHONPATH=. python demo/inference.py --image demo/examples/portrait.jpg --attack jpeg --output-dir demo/output
```

### CLI Arguments:
*   `--image`: Absolute or relative path to the target input image (required).
*   `--checkpoint`: Path to the trained model checkpoint (default: `models/BestModel/best.pt`).
*   `--attack`: Type of simulated attack to apply (`none`, `jpeg`, `noise`, `blur`, `crop`, `rotate`).
*   `--output-dir`: Path to save the output watermarked and attacked images (default: `demo/output`).

---

## E. Example Inputs

Preloaded benchmark assets are located in the [demo/examples/](file:///Users/overcoder/Code/ag_dw/latent_watermarking/demo/examples/) folder:
1. `portrait.jpg`: Headshot portrait tailored for checking fine skin details and high-frequency JPEG compression.
2. `landscape.jpg`: Natural scene landscape with detailed tree textures, used for testing Gaussian noise.
3. `cat.jpg`: Animal subject with complex fur patterns, ideal for testing blur attacks.
4. `city.jpg`: Structured architectural city view, ideal for evaluating spatial crop robustness.
5. `graphics.jpg`: Vector art and sharp edges, ideal for evaluating spatial rotation robustness.

---

## F. Expected Outputs

Running the default checkpoint `models/BestModel/best.pt` yields the following performance targets:
*   **Image-Space PSNR**: $\ge 37.40\text{ dB}$ (virtually indistinguishable from VAE baseline).
*   **SSIM**: $\ge 0.91$
*   **Extraction Bit Accuracy**:
    *   No attack: $\sim 100.0\%$
    *   Low-Frequency Extractor (JPEG/Blur): $\ge 90.0\%$
    *   High-Frequency Extractor (Noise/Resize): $\ge 85.0\%$

---

## G. Demo Pipeline

The demo executes the following sequence:
```
Input Image 
   │
   ▼
[VAE Encoder] ──► Original Latent (z)
                     │
                     ▼
              [2D DCT-II Split]
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
   Low-Freq Latent          High-Freq Latent
        │                         │
[Watermark Encoder]       [Watermark Encoder]
  (w-dim = 32)              (w-dim = 32)
        │                         │
        ▼                         ▼
 Watermarked Low          Watermarked High
        └────────────┬────────────┘
                     ▼
            [Inverse DCT-II] ──► Watermarked Latent (z_w)
                                    │
                                    ▼
                             [VAE Decoder] ──► Watermarked Image
                                                  │
                                                  ▼
                                            [Apply Attack]
                                                  │
                                                  ▼
                                            [VAE Encoder]
                                                  │
                                                  ▼
                                            [2D DCT-II]
                                                  │
                                     ┌────────────┴────────────┐
                                     ▼                         ▼
                                Low-Freq Band             High-Freq Band
                                     │                         │
                            [Watermark Decoder]       [Watermark Decoder]
                                     │                         │
                                     ▼                         ▼
                              Low-Band Estimate         High-Band Estimate
                                     └────────────┬────────────┘
                                                  ▼
                                         [Extract Signature]
```

---

## H. Limitations

*   **Rotation and Crop Attacks**: Spatial transformations like rotation and crop alter the grid alignment of latent coefficients. While BiSLW retains moderate detection, severe crops ($&gt;15\%$) or large rotation angles ($&gt;15^\circ$) degrade bit accuracy due to the spatial sensitivity of the VAE encoder.
*   **VAE CEILING**: Visual distortion and extraction performance are inherently bound by the reconstruction capability of the underlying Stable Diffusion v1.5 VAE.
