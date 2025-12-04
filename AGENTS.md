# Agent Guidelines for DriveDeRain Project

## Build/Lint/Test Commands
- **Run script**: `python <script_name>.py [args]` (no formal test suite, but `test_synthetic_rain.py` is runnable)
- **Run single test**: `python test_synthetic_rain.py` or `python demo_synthetic_rain.py` (demo scripts)
- **Lint**: `ruff check .` (config cached in .ruff_cache)
- **Format**: `ruff format .`
- **Dataset prep**: `python prepare.py` (Google Colab script, sections 1-13)
- **Synthetic rain demo**: `python demo_synthetic_rain.py` (creates samples in synthetic_rain_samples/)
- **Notebook**: `DeRain_Diffusion.ipynb` (Colab-compatible, main development environment)

## Code Style Guidelines
- **Imports**: Stdlib first, then third-party (torch, PIL, cv2, numpy, matplotlib, tqdm, diffusers, transformers, sklearn)
- **Formatting**: 4-space indents, ~100 char line length, black/ruff-compatible
- **Types**: Optional - use for clarity (path params, model I/O, image dimensions)
- **Naming**: snake_case for functions/vars, PascalCase for classes, UPPER_CASE for constants
- **Paths**: Always use `pathlib.Path`, check `.exists()` before file I/O
- **CLI**: Use `argparse` with help text and sensible defaults; scripts should be runnable standalone
- **Progress**: Use `tqdm(desc="...", unit="...")` for loops over datasets/images with descriptive labels
- **Random seeds**: Set `random.seed(42)` and `np.random.seed(42)` for reproducibility
- **Error Handling**: Print descriptive errors for missing files, CUDA OOM, invalid paths with ❌ prefix
- **Docstrings**: Simple triple-quoted strings; explain purpose, args (with types), and returns
- **User feedback**: Use emoji prefixes (✓, ❌, 🔍, 📊, etc.) for clear console output in long-running scripts

## Project Context
- Research project for rain removal from driving images using diffusion models (BDD100K 100k dataset)
- Enhanced synthetic rain generation with multi-layer depth rendering, atmospheric scattering, motion blur
- Main scripts: `prepare.py` (dataset prep), `demo_synthetic_rain.py` (rain generation demo)
- Target: realistic rain effects, physics-based rendering, suitable for training derain models
