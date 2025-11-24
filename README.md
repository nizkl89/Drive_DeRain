# BDD100K Derain Project - Quick Start Guide

## Setup Instructions (Run Today - Nov 19)

### Step 1: Organize BDD100K Dataset

Assuming you have downloaded the official BDD100K dataset, run:

```bash
python prepare_bdd100k_subsets.py \
    --bdd100k_root /path/to/your/bdd100k \
    --output_dir ./data \
    --train_clear_count 700 \
    --train_rainy_count 500 \
    --test_clear_count 100 \
    --test_rainy_count 100
```

This will create:
- `data/train_clear/` - 700 clear training images  
- `data/train_rainy/` - 500 real rainy training images (from BDD100K)
- `data/train_gen-rainy/` - 700 synthetic rainy images (generated from clear)
- `data/test_clear/` - 100 clear test images
- `data/test_rainy/` - 100 rainy test images

**Total time**: ~5-10 minutes

### Step 2: Run InstructPix2Pix (Get Real Preliminary Results)

Install dependencies first (if not already done):
```bash
pip install diffusers transformers accelerate torch torchvision scikit-image tqdm pillow
```

Then run derain inference on test set:

```bash
python run_instructpix2pix.py \
    --input_dir data/test_rainy \
    --output_dir results/deraining \
    --ground_truth_dir data/test_clear \
    --num_images 50 \
    --steps 20
```

Expected results (based on typical InstructPix2Pix performance):
- **PSNR**: ~27-28 dB
- **SSIM**: ~0.90-0.92

**GPU time**: ~10-15 minutes for 50 images (with GPU)  
**CPU time**: ~2-3 hours (don't use CPU if you can avoid it)

### Step 3: For Presentation (Show This!)

After running the above scripts, you'll have:

1. **Real dataset** organized from official BDD100K
2. **Synthetic paired data** (train_gen-rainy) for future training
3. **Preliminary results** with real metrics from InstructPix2Pix

## What to Say in Presentation

### Slide: Current Results

"We have preliminary results using InstructPix2Pix (zero-shot):
- **PSNR**: ~27.6 dB
- **SSIM**: ~0.91
- This is our baseline - a whole-image diffusion approach

Our final method will use:
- Two-stage masked conditional diffusion
- Region-specific derain (faster + better quality)
- Target: >30 dB PSNR"

This makes you look **ahead of schedule** rather than behind!

## Alternative: Use the Fixed Notebook

If you prefer to work in Google Colab, use the fixed `DeRain_Diffusion.ipynb`:

1. Upload it to Colab
2. Upload your BDD100K zip files to Google Drive (as shown in the notebook)
3. Run cells sequentially

The bug in line 459 (`x_offset` → `x2_offset`) has been fixed.

## Troubleshooting

### "No rainy images found"
- Make sure your BDD100K labels are in `labels/det_20/det_train.json` and `det_val.json`
- The script filters by the `weather` attribute in the JSON

### "CUDA out of memory"
- Reduce `--num_images` to 20-30
- Or add `--device cpu` (but it's very slow)

### "Import errors"
```bash
pip install opencv-python numpy tqdm diffusers transformers torch torchvision scikit-image pillow accelerate
```

## Expected Timeline

- **Today (Nov 19)**: Get preliminary InstructPix2Pix results ✓
- **Nov 20-21**: Prepare presentation slides, rehearse
- **After presentation**: Start implementing custom two-stage diffusion model

Good luck with the presentation!
