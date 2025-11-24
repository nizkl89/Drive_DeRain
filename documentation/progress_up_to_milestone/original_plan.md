### Project Summary: Deblurring Rainy Driving Images Using Diffusion Models for Robust Object Detection in Autonomous Vehicles

**Authors**: Lam King Cheuk, Yau Ho Yin, Chan Chun Hugo (HKUST)\*\*  
**Course**: COMP4471 (Milestone Report)

**Core Problem**  
Rainy weather causes severe degradations (rain streaks, water droplets, refractive blur, motion blur) that drastically reduce object detection performance in autonomous driving. According to the NHTSA 2025 report, mAP can drop 20–40% for critical objects (vehicles, license plates) in rain, creating serious safety risks. The team originally planned a general-purpose deblurring project with DeblurGAN-v2 but pivoted to a more focused and impactful problem: **rain-specific deblurring as a preprocessing step to improve downstream object detection robustness in real-world driving scenarios**.

**Objective**  
Develop an efficient deblurring model that restores rainy/blurry driving images with ≤100 ms inference latency on edge devices (e.g., NVIDIA Jetson), achieving **at least 20% mAP uplift** on YOLOv8 (targeting ≥75% mAP@0.5 in heavy rain vs. ~55% on raw rainy images).

**Dataset**  
BDD100K-Subsets repository (derived from BDD100K), which provides **paired clear and rainy images** with annotations in BDD100K, COCO, and YOLO formats.

- Subsets: train_clear, train_gen-rainy, train_rainy, test_clear, test_rainy
- Images resized to 512×512, 80/10/10 split
- Additional synthetic rain augmentation using RainyGAN to increase intensity variety

**Proposed Technical Approach** (Two-Stage Pipeline)

1. **Stage 1 – Rain-Aware Segmentation**  
   Fine-tune a U-Net (ResNet50 backbone) to generate binary masks of rain-affected regions (streaks + droplets).  
   Input: RGB rainy image + depth map from BDD100K annotations  
   Loss: Dice + AdamW (lr=1e-3), 50 epochs  
   Goal: Localize rain artifacts → reduce computation by 50–70% in the next stage.

2. **Stage 2 – Patch-wise Conditional Diffusion Deblurring**  
   Apply conditional diffusion only on masked regions using a fine-tuned Stable Diffusion model.  
   Conditioning: masked rainy patch + depth map  
   Losses: L1 reconstruction + VGG perceptual + multi-scale  
   Text prompt during inference: “Clear driving scene without rain or blur”  
   Target inference: <80 ms on Jetson NX  
   Patches are merged back into the original image.

**Baselines**

- DeblurGAN-v2
- Pix2Pix

**Expected Performance (Targets)**

- Baseline (DeblurGAN-v2): ~60% mAP, ~25 dB PSNR
- Proposed diffusion pipeline: **≥75% mAP, >28 dB PSNR**  
  Evaluation will use PSNR/SSIM + downstream YOLOv8 mAP@0.5 (with statistical significance testing) + qualitative visuals (especially license plate readability).

**Current Progress (as of Milestone)**  
-stage

- Dataset fully prepared and preprocessed
- U-Net segmentation implemented and trained
- DeblurGAN-v2 baseline set up
- Diffusion model fine-tuning in progress (early epochs completed)
- All code running on PyTorch + Google Colab A100

**Future Work**  
Full diffusion fine-tuning on the complete dataset, latency optimization for edge deployment, possible integration of Swin-Transformer or latent kernel prediction modules.

**Summary in One Sentence**  
The project proposes a novel two-stage pipeline that first segments rain-affected areas with a U-Net and then applies patch-wise conditional Stable Diffusion to achieve high-quality, efficient deblurring of rainy driving images, significantly boosting YOLOv8 object detection performance in adverse weather for safer autonomous vehicles.
