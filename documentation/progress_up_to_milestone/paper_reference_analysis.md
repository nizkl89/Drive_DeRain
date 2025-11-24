# De-raining Natural Scenes using Diffusion Models

**Authors**  
Emil Biju, Sidharth Tadeparti, Sneha Jayaganthan  
Stanford University (Electrical & Mechanical Engineering)  

## Project Overview  
This Stanford project systematically benchmarks modern generative models for **single-image deraining** on real-world raindrop-affected outdoor scenes (not synthetic streaks). The core focus is diffusion models, with careful exploration of compute efficiency (via rain segmentation) and data efficiency (style transfer, unpaired training on limited/synthetic/unpaired data).

## Key Contributions
- First thorough comparison of diffusion-based, GAN-based, and style-transfer methods for **real-world raindrop removal** (most prior work focuses on synthetic streak rain).
- U-Net + Diffusion pipeline that applies diffusion **only to rain-affected patches** → removes patch artifacts and speeds up inference.
- Fine-tuning of a pretrained WeatherDiffusion model (originally for multiple weather types) specifically for deraining → achieves the **best visual quality** despite lower PSNR/SSIM.
- Extensive ablation on data/compute trade-offs: paired vs synthetic vs unpaired training, fine-tuned vs zero-shot diffusion, etc.

## Methods Compared

| Model Type                  | Model                                    | Data Type       | Notes                                                                 |
|-----------------------------|------------------------------------------|-----------------|-----------------------------------------------------------------------|
| Baseline                   | Stable Diffusion Inpainting + U-Net mask | Paired real     | Very poor (PSNR 10.73)                                                    |
| Compute-efficient           | Diffusion + U-Net segmentation             | Paired real     | Applies diffusion only on rainy patches → cleaner than full-patch diffusion |
| Data-efficient              | VGG Style Transfer                        | Paired real     | Decent but adds unwanted style artifacts                                       |
|                             | CycleGAN                                 | Unpaired real   | Good metrics but fails to fully remove raindrops                          |
|                             | Pix2Pix on synthetic data (Rain100H)  | Synthetic streaks | High PSNR but poor generalization to real raindrops                        |
| End-to-end (paired real)   | Pix2Pix GAN                             | Paired real     | **Highest PSNR (70.96)** but visible artifacts and unnatural look       |
|                             | Diffusion (pretrained, no fine-tuning)   | Paired real     | Patchy when run patch-wise                                             |
|                             | **Diffusion (fine-tuned)**               | Paired real     | **Best visual quality** (closest to ground truth, natural textures)       |

## Results Summary (from Table 1)

| Model                                      | PSNR (dB) | SSIM  |
|--------------------------------------------|-------------|-------|
| Stable Diffusion Inpainting (baseline)    | 10.73       | 0.20  |
| Diffusion + U-Net                          | 21.74       | 0.78  |
| VGG Style Transfer                          | 60.19       | 0.70  |
| CycleGAN                                   | 70.17       | 0.83  |
| Pix2Pix (synthetic data)                | 69.53       | 0.78  |
| Pix2Pix (real paired data)               | **70.96**  | **0.83** |
| Diffusion (pretrained)                     | 19.87       | 0.78  |
| **Diffusion (fine-tuned)**                 | 24.51       | 0.80  |

**Key Insight**: PSNR/SSIM **do not correlate perfectly** with perceived quality in this task. GANs optimize L1/MSE → sky-high PSNR but blurry/artifact-prone outputs. The fine-tuned diffusion model produces **significantly more natural and sharp results**, especially in textures, signs, and fine details — despite "only" 24.51 dB PSNR.

Visual comparisons (Figures 3–5 in the paper) clearly show that the **fine-tuned diffusion model** is the only one that produces images almost indistinguishable from ground truth.

## Main Findings

1. **Fine-tuned diffusion models** outperform GANs **qualitatively** for real-world deraining, even if GANs win on PSNR/SSIM.

2. **Patch-wise diffusion** suffers from contrast inconsistencies between patches; using **U-Net segmentation** to process only rainy patches eliminates this artifact and reduces compute.

3. **Style transfer** (both VGG and CycleGAN) can work surprisingly well even with very little or unpaired data, but they tend to alter image style/appearance in unwanted ways.

4. **Synthetic streak datasets** (e.g., Rain100H) **do not transfer** well to real raindrop removal.

5. **Stable Diffusion inpainting** is **not suitable** for this task when masked by a U-Net; it hallucinates rain back in to maintain continuity.

## Conclusion  
For real-world single-image deraining with raindrops, **a fine-tuned conditional diffusion model fine-tuned on a modest real-world paired dataset** (~1000 images) currently gives the **best visual results**. The authors show a practical pipeline (U-Net segmentation + selective diffusion patching) that mitigates the main drawbacks of diffusion models (patch artifacts and high inference cost).

The project is exceptionally well-executed, with clear ablations, honest reporting of metric vs perceptual quality mismatch, and thoughtful discussion of compute/data trade-offs. The fine-tuned diffusion model stands out as the clear winner for anyone wanting the highest-quality derained images today.