# Google Colab Data Preparation Guide

## 📋 Quick Start

### File Location
- Script: `test.py`
- Project Directory: `/content/drive/MyDrive/com4471/project`

### How to Use in Google Colab

1. **Create new Colab notebook** or open existing one
2. **Copy sections from `test.py`** into separate cells
3. **Run cells sequentially** (or Run All)

## ✅ Key Features

### Smart Skip Logic (No Re-download on "Run All")

The script automatically detects:
- ✓ Already installed packages (skips reinstall)
- ✓ Already downloaded dataset (skips download)
- ✓ Already processed splits (skips processing)
- ✓ Already generated synthetic images (skips generation)

**This means you can safely "Run All" multiple times without wasting time!**

## 📂 Output Structure

```
/content/drive/MyDrive/com4471/project/
├── bdd100k_raw/              # Downloaded BDD100K dataset
│   ├── ds0/                  # Supervisely format images
│   └── ann/                  # Annotations/labels
│
├── image_lists/              # Cached image lists
│   ├── rainy_images.pkl
│   ├── clear_images.pkl
│   └── overcast_images.pkl
│
├── data/                     # Final organized dataset
│   ├── train/
│   │   ├── rainy/           (1,200 images @ 512x512)
│   │   ├── clear/           (1,200 images @ 512x512)
│   │   └── synthetic_rainy/ (1,200 images @ 512x512)
│   ├── val/
│   │   ├── rainy/           (150 images)
│   │   └── clear/           (150 images)
│   ├── test/
│   │   ├── rainy/           (200 images)
│   │   └── clear/           (200 images)
│   ├── dataset_info.json    # Metadata
│   └── dataset_samples.png  # Visualization
│
├── config.json              # Training configuration
└── COLAB_GUIDE.md          # This file
```

## 🎯 Sections Breakdown

### SECTION 1: Mount Drive
- Mounts Google Drive
- Creates project directory
- Sets working directory

### SECTION 2: Install Dependencies (Smart)
- **NEW:** Only installs if not already present
- Checks each package before installing
- Safe for "Run All"

### SECTION 3: Download Dataset (Smart)
- **NEW:** Checks if already downloaded (>10k images)
- Skips download if found
- Downloads 5.39 GB from dataset-tools

### SECTION 4: Verify Download
- Explores directory structure
- Counts total images
- Shows sample file paths

### SECTION 5: Alternative Manual Download
- Instructions if dataset-tools fails
- Use official BDD100K website
- Manual zip extraction code

### SECTION 6: Parse Structure
- Finds images directory
- Finds labels directory
- Handles both formats (Supervisely & BDD100K official)

### SECTION 7: Extract Weather Metadata
- Parses all JSON labels
- Extracts weather attributes
- Shows distribution statistics

### SECTION 8: Filter by Weather
- Separates rainy images
- Separates clear images
- Saves filtered lists to pickle files

### SECTION 9: Create Splits (Smart)
- **NEW:** Checks if already processed
- Skips if >1000 training images found
- Resizes to 512x512
- Creates train/val/test directories

### SECTION 10: Generate Synthetic Rain (Smart)
- **NEW:** Checks if already generated
- Skips if >1000 synthetic images found
- Adds realistic rain effects
- Multiple intensity levels

### SECTION 11: Create Metadata
- Saves dataset statistics
- Counts all splits
- Calculates training pairs

### SECTION 12: Visualize Samples
- Creates 3x6 grid visualization
- Shows rainy vs clear samples
- Saves to PNG file

### SECTION 13: Verify Integrity
- Checks all images are readable
- Verifies dimensions (512x512)
- Reports any corrupted files

### SECTION 14: Save Configuration
- Creates config.json for training
- Sets paths, hyperparameters
- Ready for model training

### SECTION 15: Final Summary
- Prints complete statistics
- Shows next steps
- Lists all output files

## ⏱️ Estimated Time

| Step | First Run | Subsequent Runs |
|------|-----------|-----------------|
| Mount Drive | 5s | 5s |
| Install Dependencies | 2-3 min | **5s** (skipped) |
| Download Dataset | 15-30 min | **5s** (skipped) |
| Parse & Filter | 5-10 min | 5-10 min |
| Create Splits | 10-15 min | **5s** (skipped) |
| Generate Synthetic | 5-10 min | **5s** (skipped) |
| Metadata & Verify | 2-3 min | 2-3 min |
| **TOTAL** | **40-70 min** | **~15 min** |

## 💾 Storage Requirements

- BDD100K Raw: ~5.4 GB
- Processed Dataset: ~4-5 GB
- Temporary Files: ~1 GB
- **Total: ~11-12 GB in Google Drive**

## 🚨 Troubleshooting

### "Dataset-tools download fails"
- Use SECTION 5 (manual download)
- Download from https://bdd-data.berkeley.edu/
- Upload zips to Google Drive
- Run extraction code

### "Not enough rainy images"
- Check weather distribution in SECTION 7
- May need to adjust split sizes in SECTION 9
- Or use overcast images as additional data

### "Out of memory during processing"
- Reduce batch processing in image resizing
- Process splits separately
- Restart runtime and clear cache

### "Colab disconnects during download"
- Enable "Run All" before stepping away
- Or use Colab Pro for longer sessions
- Download will resume from checkpoint

## ✅ Verification Checklist

After running all sections, verify:

- [ ] `data/train/rainy/` has ~1,200 images
- [ ] `data/train/clear/` has ~1,200 images
- [ ] `data/train/synthetic_rainy/` has ~1,200 images
- [ ] `data/val/` has rainy + clear folders
- [ ] `data/test/` has rainy + clear folders
- [ ] `config.json` exists
- [ ] `dataset_info.json` exists
- [ ] All images are 512x512 pixels
- [ ] Sample visualization looks good

## 🎯 Next Steps After Data Prep

1. **Train Pix2Pix GAN baseline**
   - Use `data/train/synthetic_rainy` → `data/train/clear`
   - Target: Improve from 19.58 dB

2. **Generate rain masks for segmentation**
   - Use synthetic pairs (rainy - clear = mask)
   - Train U-Net for rain detection

3. **Fine-tune diffusion model**
   - Use masks + rainy images
   - Target: 24-28 dB PSNR

4. **Evaluate on test set**
   - Compare all methods
   - Run YOLOv8 detection benchmark

## 📞 Support

If you encounter issues:
1. Check the output messages carefully
2. Verify paths are correct (`com4471/project`)
3. Ensure enough Google Drive storage
4. Check Colab GPU/RAM status

---

**Ready to start! Just copy-paste sections from `test.py` into Colab! 🚀**
