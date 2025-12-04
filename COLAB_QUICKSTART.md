# Colab Quick Start - Using the Fixed Rain Generator

## Step-by-Step Instructions for Google Colab

### 1. Upload the Fixed Generator

In your Colab notebook, add a new cell:

```python
# Upload fixed_rain_generator.py from your local machine
from google.colab import files
uploaded = files.upload()

# Or if it's in your Drive:
# (Make sure the file is in your project folder first)
import sys
sys.path.insert(0, '/content/drive/MyDrive/comp4471/project')
```

### 2. Test on One Image First

```python
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from fixed_rain_generator import FixedModularRainGenerator

# Load a test image
img_path = "/content/drive/MyDrive/comp4471/project/data/val/clear/5d206f57-418ddad7.jpg"
img = cv2.imread(img_path)

# Generate rain
generator = FixedModularRainGenerator()
rainy, depth = generator.generate(
    img,
    rain_streaks="medium",
    wet_roads="medium",
    light_bloom="medium"
)

# Display comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original (Clear)", fontsize=14)
axes[0].axis("off")

axes[1].imshow(cv2.cvtColor(rainy, cv2.COLOR_BGR2RGB))
axes[1].set_title("Fixed Synthetic Rain", fontsize=14, color="green")
axes[1].axis("off")

plt.tight_layout()
plt.show()

print("✓ If you see NO cyan blobs, the fix is working!")
```

### 3. If Results Look Good, Update Your Notebook

**Method A - Replace the class in Section 5:**

1. Open `Modular_Rain_Generation_real.ipynb`
2. Go to **Section 5** (the cell with `class ModularRainGenerator:`)
3. Copy the entire `FixedModularRainGenerator` class from `fixed_rain_generator.py`
4. Paste it, replacing the old class
5. Rename `FixedModularRainGenerator` to `ModularRainGenerator`
6. Re-run the notebook from Section 5 onwards

**Method B - Import at the top of Section 8:**

Add this to the beginning of Section 8 (before generation starts):

```python
# Use fixed generator instead
from fixed_rain_generator import FixedModularRainGenerator as ModularRainGenerator
print("✓ Using fixed rain generator (no artifacts)")
```

### 4. Run Full Generation

Continue with your existing workflow:
- Section 8 will now use the fixed generator
- All images will be processed without cyan artifacts
- Everything else remains the same

## Key Differences You'll Notice

| Aspect | Before (Broken) | After (Fixed) |
|--------|-----------------|---------------|
| Objects | Cyan blobs, halos | Clean, natural |
| Wet roads | Artificial blue tint | Natural reflection |
| Rain visibility | Obscured by artifacts | Clear streaks |
| Overall quality | Unrealistic | Photorealistic |

## Troubleshooting

**Q: I still see artifacts**
- Make sure you're using `FixedModularRainGenerator`, not the old class
- Check that the import worked: `print(generator.__class__.__name__)`

**Q: Import error**
- Verify the file path in `sys.path.insert()`
- Make sure `fixed_rain_generator.py` is in the correct folder

**Q: Comparison with original**
To see before/after, generate one image with both:

```python
from fixed_rain_generator import FixedModularRainGenerator

# Original (broken)
# ... run Section 5 first to define the old class ...
old_gen = ModularRainGenerator()
old_rainy, _ = old_gen.generate(img, rain_streaks="medium", wet_roads="medium", light_bloom="medium")

# Fixed
new_gen = FixedModularRainGenerator()
new_rainy, _ = new_gen.generate(img, rain_streaks="medium", wet_roads="medium", light_bloom="medium")

# Compare
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original")
axes[1].imshow(cv2.cvtColor(old_rainy, cv2.COLOR_BGR2RGB))
axes[1].set_title("Old (Artifacts)")
axes[2].imshow(cv2.cvtColor(new_rainy, cv2.COLOR_BGR2RGB))
axes[2].set_title("Fixed (Clean)")
plt.show()
```

## Summary

✅ **Upload** `fixed_rain_generator.py` to Colab
✅ **Test** on one image to verify
✅ **Replace** the old class in Section 5 OR import the fixed version
✅ **Re-run** Section 8 to generate clean dataset

**Expected result:** Photorealistic rain without cyan blobs or color artifacts!
