# DriveDeRain Training Pipeline - Phase 2 Plan

## 🎯 Overall Objective

Train a two-stage derain model (U-Net + Diffusion fine-tuning) using the **completed V40 windshield rain dataset** (1,550 images) to remove windshield rain effects from driving images and generalize to real-world rainy conditions.

---

## 📋 Current Status

**Status:** ✅ **READY TO START TRAINING** - All synthetic data complete

**Completed:**
- ✅ 1,550 windshield rainy images (V40) generated
- ✅ 1,550 clear ground truth images prepared
- ✅ 1,550 real rainy images (for testing) organized
- ✅ Train/Val/Test splits configured (1200/150/200)
- ✅ Dataset synced to Google Drive
- ✅ Quality validation passed (user approved V40)

**Next:** Implement training pipeline for U-Net derain model

---

## 🏗️ Phase 2A: Training Infrastructure Setup (NEXT)


### Objective
Set up training environment, data loaders, and infrastructure for U-Net derain training.

### Tasks:
1. **Create training notebook: `train_unet_derain.ipynb`**
   - Google Colab compatible
   - GPU-enabled (T4 or better)
   - Sections: Setup → Data Loading → Model → Training → Evaluation
   - Estimated time: 1 day

2. **Implement PyTorch Dataset class**
   - Load paired images (clear, windshield_rainy_v40)
   - Handle train/val/test splits
   - Image preprocessing:
     - Resize to 256×256 (for memory efficiency) or 512×512 (for quality)
     - Normalize to [-1, 1] for model input
     - ToTensor conversion
   - Optional augmentation:
     - Random horizontal flip (p=0.5)
     - Random crop (if using 512×512)
     - Color jitter (slight variation)
   - Estimated time: 0.5 day

3. **Create DataLoaders**
   - Batch size: 8-16 (depending on GPU memory)
   - Num workers: 2-4 (Colab limitation)
   - Shuffle: True for train, False for val/test
   - Pin memory: True for faster GPU transfer
   - Estimated time: 0.5 day

4. **Visualization utilities**
   - Display random training samples (clear vs rainy)
   - Show model predictions during training
   - Plot training curves (loss, PSNR, SSIM)
   - Side-by-side comparisons (rainy → derained → ground truth)
   - Estimated time: 0.5 day

**Total Phase 2A Time:** 2.5 days

---

## 🧠 Phase 2B: U-Net Model Architecture

### Objective
Implement U-Net derain model with appropriate architecture for windshield rain removal.

### Model Architecture:
```
U-Net Encoder-Decoder:
├─ Input: 3 channels (RGB rainy image) → 256×256 or 512×512
├─ Encoder (Downsampling):
│   ├─ Conv Block 1: 64 channels, 3×3 conv + ReLU + BN
│   ├─ Conv Block 2: 128 channels, 3×3 conv + ReLU + BN
│   ├─ Conv Block 3: 256 channels, 3×3 conv + ReLU + BN
│   ├─ Conv Block 4: 512 channels, 3×3 conv + ReLU + BN
│   └─ Bottleneck: 1024 channels, 3×3 conv + ReLU + BN
├─ Decoder (Upsampling):
│   ├─ UpConv Block 4: 512 channels + skip connection
│   ├─ UpConv Block 3: 256 channels + skip connection
│   ├─ UpConv Block 2: 128 channels + skip connection
│   └─ UpConv Block 1: 64 channels + skip connection
└─ Output: 3 channels (RGB clean image), tanh activation
```

### Implementation Details:
- **Conv Block:** Conv2d → BatchNorm2d → ReLU (inplace)
- **Downsampling:** MaxPool2d (2×2) or Conv2d stride 2
- **Upsampling:** ConvTranspose2d (2×2 stride 2) or Upsample + Conv
- **Skip Connections:** Concatenate encoder features to decoder
- **Output Activation:** Tanh (outputs in [-1, 1])

### Alternative: Pretrained U-Net
**Option:** Use pretrained U-Net from `segmentation_models_pytorch`
```python
import segmentation_models_pytorch as smp
model = smp.Unet(
    encoder_name="resnet34",        # Pretrained ResNet backbone
    encoder_weights="imagenet",     # ImageNet weights
    in_channels=3,                  # RGB input
    classes=3,                      # RGB output
    activation=None                 # Custom activation
)
```

**Pros:**
- ✅ Faster convergence (pretrained encoder)
- ✅ Better feature extraction
- ✅ Proven architecture

**Cons:**
- ❌ Less control over architecture
- ❌ Larger model size

**Recommendation:** Start with pretrained U-Net for faster results, implement custom U-Net if needed.

**Estimated Time:** 1 day (pretrained) or 2 days (custom)

---

## 📊 Phase 2C: Loss Functions & Metrics

### Objective
Define loss functions to train U-Net for high-quality derain output.

### Multi-Component Loss Function:
```
Total Loss = α₁ × L1_Loss + α₂ × Perceptual_Loss + α₃ × SSIM_Loss

Where:
- L1_Loss: Pixel-wise absolute difference (sharp reconstruction)
- Perceptual_Loss: VGG feature matching (perceptual quality)
- SSIM_Loss: Structural similarity (preserve textures)
- α₁, α₂, α₃: Loss weights (tunable hyperparameters)
```

### Loss Component Details:

**1. L1 Loss (Pixel Reconstruction)**
```python
l1_loss = nn.L1Loss()
loss_l1 = l1_loss(output, target)
```
- **Purpose:** Ensure pixel-level accuracy
- **Weight (α₁):** 1.0 (baseline)

**2. Perceptual Loss (VGG Features)**
```python
# Use pretrained VGG16 features
vgg = torchvision.models.vgg16(pretrained=True).features[:16]
vgg.eval()
for param in vgg.parameters():
    param.requires_grad = False

# Compute feature difference
features_output = vgg(output)
features_target = vgg(target)
loss_perceptual = nn.MSELoss()(features_output, features_target)
```
- **Purpose:** Match perceptual quality (textures, structures)
- **Weight (α₂):** 0.1-0.5 (tunable)

**3. SSIM Loss (Structural Similarity)**
```python
from pytorch_msssim import ssim
loss_ssim = 1 - ssim(output, target, data_range=1.0)
```
- **Purpose:** Preserve structural information
- **Weight (α₃):** 0.1-0.3 (tunable)

### Evaluation Metrics:
**Quantitative Metrics:**
- **PSNR (Peak Signal-to-Noise Ratio):** Higher = better (target: >28dB)
- **SSIM (Structural Similarity):** Higher = better (target: >0.85)
- **LPIPS (Learned Perceptual Image Patch Similarity):** Lower = better
- **MSE (Mean Squared Error):** Lower = better

**Qualitative Metrics:**
- Visual inspection of derained outputs
- Comparison with real clear images
- Artifact detection (blurriness, color shifts)

**Estimated Time:** 1 day (implement + integrate)

---

## 🚀 Phase 2D: Training Loop Implementation

### Objective
Implement robust training loop with checkpointing, logging, and validation.

### Training Configuration:
```python
# Hyperparameters
EPOCHS = 100
BATCH_SIZE = 8  # or 16 if GPU memory allows
LEARNING_RATE = 1e-4
IMAGE_SIZE = 256  # or 512 for higher quality
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)
# or CosineAnnealingLR for smooth decay
```

### Training Loop Structure:
```python
for epoch in range(EPOCHS):
    # Training phase
    model.train()
    for batch in train_loader:
        rainy_images, clear_images = batch
        
        # Forward pass
        outputs = model(rainy_images)
        
        # Compute loss
        loss = compute_total_loss(outputs, clear_images)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            rainy_images, clear_images = batch
            outputs = model(rainy_images)
            
            # Compute metrics (PSNR, SSIM)
            val_psnr = compute_psnr(outputs, clear_images)
            val_ssim = compute_ssim(outputs, clear_images)
    
    # Scheduler step
    scheduler.step(val_loss)
    
    # Save checkpoint
    if val_psnr > best_psnr:
        save_checkpoint(model, optimizer, epoch, val_psnr)
```

### Checkpointing Strategy:
- **Save best model:** When validation PSNR improves
- **Save periodic checkpoints:** Every 10 epochs
- **Save to Google Drive:** `/content/drive/MyDrive/DriveDeRain/checkpoints/`
- **Checkpoint contents:**
  - Model state_dict
  - Optimizer state_dict
  - Epoch number
  - Best metrics (PSNR, SSIM)
  - Loss weights (for reproducibility)

### Logging & Monitoring:
- **Progress bars:** `tqdm` for batch/epoch progress
- **TensorBoard (optional):** Log loss curves, images, metrics
- **Console output:** Print epoch summary (loss, PSNR, SSIM)
- **Visual checkpoints:** Save sample derained images every 5 epochs

**Estimated Time:** 1.5 days (implementation + debugging)

---

## 🧪 Phase 2E: Training Execution & Monitoring

### Objective
Train U-Net model on V40 windshield rain dataset and monitor convergence.

### Training Splits:
- **Train:** 1,200 paired images (clear + windshield_rainy_v40)
- **Val:** 150 paired images (for hyperparameter tuning)
- **Test:** 200 real rainy images (for final evaluation)

### Expected Training Time:
- **Per epoch:** ~5-10 minutes (1,200 images, batch size 8)
- **Total (100 epochs):** ~8-16 hours
- **Recommendation:** Run overnight or use multiple sessions

### Training Phases:
**Phase 1: Initial Convergence (Epochs 1-20)**
- Monitor loss decrease
- Check for instability (NaN, exploding gradients)
- Validate learning rate (adjust if needed)
- **Expected:** Loss drops rapidly, PSNR increases to ~20dB

**Phase 2: Quality Improvement (Epochs 21-60)**
- Loss stabilizes, PSNR improves slowly
- Start seeing visually acceptable derained outputs
- **Expected:** PSNR reaches 25-28dB, SSIM ~0.80-0.85

**Phase 3: Fine-tuning (Epochs 61-100)**
- Diminishing returns on metrics
- Focus on artifact removal
- **Expected:** PSNR 28-30dB, SSIM ~0.85-0.90

### Stopping Criteria:
- **Early stopping:** If val loss doesn't improve for 15 epochs
- **Target metrics reached:** PSNR > 28dB AND SSIM > 0.85
- **Maximum epochs:** 100 epochs
- **Overfitting detected:** Val loss increases while train loss decreases

### Monitoring Checklist:
- [ ] Training loss decreasing smoothly
- [ ] Validation loss tracking training loss (no overfitting)
- [ ] PSNR improving over epochs
- [ ] SSIM improving over epochs
- [ ] Derained images visually acceptable (no blur, color shifts, artifacts)
- [ ] Model generalizes to validation set

**Estimated Time:** 1-2 days (training + monitoring)

---

## 📈 Phase 2F: Evaluation on Real Rainy Images

### Objective
Test trained U-Net on real BDD100K rainy images to validate generalization.

### Test Set Evaluation:
**Dataset:** 200 real rainy images from BDD100K (test split)

**Evaluation Protocol:**
1. **Load best checkpoint** (highest validation PSNR)
2. **Inference on test set:**
   - Process all 200 real rainy images
   - Generate derained outputs
   - Save results for visual inspection
3. **Compute metrics (if clear ground truth available):**
   - PSNR, SSIM (if paired clear images exist)
   - LPIPS (perceptual quality)
4. **Qualitative analysis:**
   - Visual inspection (random sample)
   - Check for artifacts (blur, color distortion, incomplete rain removal)
   - Compare with original rainy images

### Expected Challenges:
**Domain Gap (Synthetic → Real):**
- V40 uses windshield droplets + trails (synthetic)
- Real rain may include falling streaks, atmospheric fog, complex reflections
- **Mitigation:** U-Net may partially generalize; diffusion refinement (Stage 2) will help

**Artifact Detection:**
- **Residual rain:** Trails/droplets not fully removed
- **Over-smoothing:** Loss of texture/details
- **Color shifts:** Incorrect white balance after derain
- **Blurriness:** Perceptual loss may blur fine details

### Success Criteria:
- ✅ U-Net removes most V40-style windshield rain (droplets + trails)
- ✅ Derained images look visually cleaner than input
- ✅ No major artifacts (color shifts, blur, distortion)
- ✅ PSNR > 25dB on synthetic val set (if ground truth available)
- ⚠️ May not fully generalize to real rain (expected, will improve in Stage 2)

**Estimated Time:** 0.5 day (inference + analysis)

---

## 🔧 Phase 2G: Hyperparameter Tuning (Optional)

### Objective
Optimize training hyperparameters if initial results are suboptimal.

### Tunable Hyperparameters:
**Model Architecture:**
- [ ] Image resolution (256×256 vs 512×512)
- [ ] Encoder depth (4 vs 5 levels)
- [ ] Channel counts (64/128/256/512 vs 32/64/128/256)
- [ ] Pretrained encoder (ResNet34 vs ResNet50 vs EfficientNet)

**Loss Weights:**
- [ ] L1 weight (α₁): 1.0 → tune
- [ ] Perceptual weight (α₂): 0.1-0.5 → tune
- [ ] SSIM weight (α₃): 0.1-0.3 → tune

**Training Configuration:**
- [ ] Learning rate (1e-4 vs 5e-5 vs 1e-3)
- [ ] Batch size (8 vs 16 vs 32)
- [ ] Optimizer (Adam vs AdamW vs SGD)
- [ ] Scheduler (ReduceLROnPlateau vs CosineAnnealing vs StepLR)

**Data Augmentation:**
- [ ] Horizontal flip probability (0.5 vs 0.3)
- [ ] Color jitter strength (0.1 vs 0.2)
- [ ] Random crop (enabled vs disabled)

### Tuning Strategy:
**Approach:** One variable at a time (OVAT)
1. Start with baseline configuration
2. Identify bottleneck (loss not decreasing? Poor PSNR? Artifacts?)
3. Adjust one hyperparameter
4. Train for 20-30 epochs (quick validation)
5. Compare metrics with baseline
6. Keep if improvement, revert if worse
7. Repeat for next hyperparameter

**Estimated Time:** 2-4 days (if needed)

---

## 🌟 Phase 3: Diffusion Model Fine-Tuning (Future)

### Objective
Fine-tune Stable Diffusion InstructPix2Pix on U-Net derained outputs to remove residual artifacts and enhance quality.

### Training Data:
- **Input:** U-Net derained images (may have artifacts)
- **Target:** Original clear images (ground truth)
- **Prompt:** "remove rain artifacts, enhance image quality"

### Model:
- **Base Model:** `timbrooks/instruct-pix2pix`
- **Fine-tuning:** LoRA (Low-Rank Adaptation) for efficient training
- **Epochs:** 20-50 (diffusion models converge faster)

### Expected Outcome:
- Remove residual rain effects U-Net missed
- Enhance perceptual quality (sharper, more realistic)
- Handle real rainy images better (diffusion has better generalization)

### Estimated Time:
- **Implementation:** 2 days
- **Training:** 1-2 days (20-50 epochs)
- **Evaluation:** 0.5 day

**Total Phase 3 Time:** 3.5-4.5 days

---

## 📊 Phase 4: Final Evaluation & Comparison (Future)

### Objective
Comprehensive evaluation of full pipeline (U-Net + Diffusion) against baselines.

### Evaluation Protocol:
**Test Set:** 200 real rainy BDD100K images

**Metrics:**
- Quantitative: PSNR, SSIM, LPIPS
- Qualitative: Visual inspection, user study
- Robustness: Performance across intensities (light/medium/heavy rain)

**Baseline Comparisons:**
- Traditional methods: DerainNet, RESCAN
- Recent methods: MPRNet, Restormer
- Our pipeline: U-Net → Diffusion

**Success Criteria:**
- ✅ Outperforms baselines on PSNR/SSIM
- ✅ Visually superior derained outputs
- ✅ Generalizes to real-world rain
- ✅ No artifacts or quality degradation

**Estimated Time:** 1-2 days

---

## 📅 Overall Timeline

### Phase 2: U-Net Training (Current)
```
Phase 2A: Infrastructure Setup        [2.5 days]
Phase 2B: U-Net Architecture          [1 day]
Phase 2C: Loss Functions & Metrics    [1 day]
Phase 2D: Training Loop               [1.5 days]
Phase 2E: Training Execution          [1-2 days]
Phase 2F: Evaluation on Real Images   [0.5 day]
Phase 2G: Hyperparameter Tuning       [2-4 days, optional]

Total: 7.5-13 days (without tuning: 7.5-8.5 days)
```

### Phase 3: Diffusion Fine-Tuning (Future)
```
Implementation + Training + Evaluation: 3.5-4.5 days
```

### Phase 4: Final Evaluation (Future)
```
Comprehensive evaluation: 1-2 days
```

**Overall Estimated Time:** 12-20 days (end-to-end pipeline)

---

## 🚧 Risks & Mitigations

### Risk 1: U-Net Overfits to V40 Synthetic Patterns
**Issue:** Model learns V40-specific droplet/trail patterns, fails on real rain

**Mitigation:**
- Use strong data augmentation (flip, crop, jitter)
- Add dropout layers in U-Net (0.2-0.3)
- Early stopping based on validation loss
- Test on real rainy images frequently
- Rely on diffusion stage (Phase 3) to generalize better

---

### Risk 2: Domain Gap (Synthetic → Real)
**Issue:** V40 windshield rain != real rain (different phenomena)

**Mitigation:**
- Accept U-Net limitations (Stage 1 baseline)
- Diffusion fine-tuning (Stage 2) will bridge gap
- Consider mixing synthetic + real rainy images (semi-supervised)
- Use perceptual loss (VGG) to match real image statistics

---

### Risk 3: Training Instability (NaN loss, gradient explosion)
**Issue:** Loss becomes NaN or model doesn't converge

**Mitigation:**
- Use gradient clipping (`torch.nn.utils.clip_grad_norm_`)
- Lower learning rate (1e-5 instead of 1e-4)
- Check for data normalization issues
- Ensure loss weights are balanced (α₁, α₂, α₃)
- Use mixed precision training (torch.cuda.amp)

---

### Risk 4: Poor Generalization to Real Rainy Images
**Issue:** U-Net performs well on synthetic val set but fails on real rain

**Mitigation:**
- Validate on real rainy images early (every 10 epochs)
- Adjust loss weights (increase perceptual loss α₂)
- Use pretrained encoder (ResNet34) for better features
- Proceed to diffusion stage (better generalization expected)

---

### Risk 5: Insufficient GPU Resources (Colab limitations)
**Issue:** Training too slow or OOM (Out of Memory) errors

**Mitigation:**
- Reduce image resolution (256×256 instead of 512×512)
- Use smaller batch size (4 instead of 8)
- Use Colab Pro (more GPU time, better GPUs)
- Split training into multiple sessions (checkpoint resume)
- Gradient accumulation (simulate larger batches)

---

## 🎯 Success Criteria

### Phase 2 (U-Net Training):
- ✅ Model converges without instability
- ✅ Validation PSNR > 28dB, SSIM > 0.85 (on synthetic val set)
- ✅ Derained images visually acceptable (no major artifacts)
- ✅ Training completes within 2 weeks
- ⚠️ Partial generalization to real rain (expected, will improve in Phase 3)

### Phase 3 (Diffusion Fine-Tuning - Future):
- ✅ Diffusion model enhances U-Net outputs
- ✅ PSNR improvement of +2-3dB over U-Net alone
- ✅ Removes residual artifacts
- ✅ Generalizes well to real rainy images

### Phase 4 (Final Evaluation - Future):
- ✅ Outperforms baseline derain methods
- ✅ Qualitatively superior on real test set
- ✅ Robust across rain intensities
- ✅ Ready for publication/deployment

---

## 📞 Next Actions

### Immediate (This Week):
1. **Create `train_unet_derain.ipynb`** notebook with 10 sections:
   - Section 1: Mount Drive & Setup
   - Section 2: Configuration (hyperparameters, paths)
   - Section 3: Import Libraries
   - Section 4: Dataset & DataLoader
   - Section 5: U-Net Model Architecture
   - Section 6: Loss Functions & Metrics
   - Section 7: Training Loop
   - Section 8: Visualization Utilities
   - Section 9: Train Model (execute training)
   - Section 10: Evaluate on Real Images

2. **Implement Dataset class** for loading clear + windshield_rainy_v40 pairs

3. **Set up U-Net model** (pretrained ResNet34 encoder)

4. **Implement multi-component loss** (L1 + Perceptual + SSIM)

5. **Start training** (initial run with default hyperparameters)

### Next Week:
6. **Monitor training** (check convergence, metrics)

7. **Evaluate on real rainy images** (test set)

8. **Tune hyperparameters** (if needed)

9. **Document results** (update progress.md)

### Future (Phase 3):
10. **Implement diffusion fine-tuning** pipeline

11. **Train diffusion model** on U-Net outputs

12. **Final evaluation** and comparison

---

## 📚 Resources & References

### Models & Libraries:
- **PyTorch:** https://pytorch.org/
- **Segmentation Models PyTorch:** https://github.com/qubvel/segmentation_models.pytorch
- **PyTorch SSIM:** https://github.com/VainF/pytorch-msssim
- **LPIPS:** https://github.com/richzhang/PerceptualSimilarity
- **Stable Diffusion InstructPix2Pix:** https://huggingface.co/timbrooks/instruct-pix2pix

### Datasets:
- **BDD100K:** https://www.bdd100k.com/
- **V40 Windshield Rain Dataset:** `/content/drive/MyDrive/DriveDeRain/data/`

### Related Papers:
- **U-Net:** Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- **Perceptual Loss:** Johnson et al., "Perceptual Losses for Real-Time Style Transfer" (2016)
- **InstructPix2Pix:** Brooks et al., "InstructPix2Pix: Learning to Follow Image Editing Instructions" (2023)
- **Image Derain:** Survey paper on single image deraining techniques

---

**Last Updated:** 2025-12-04  
**Status:** ✅ Ready to start Phase 2 (U-Net Training)  
**Next Milestone:** Train U-Net derain model on V40 dataset
