# Agent Guidelines for BDD100K Derain Project

## Build/Test Commands
- **Run single script**: `python <script_name>.py [args]`
- **Quick test (1 image)**: `python test_quick.py`
- **Prepare dataset**: `python prepare_bdd100k_subsets.py --bdd100k_root /path/to/bdd100k --output_dir ./data`
- **Run inference**: `python run_instructpix2pix.py --input_dir data/test_rainy --output_dir results/deraining --ground_truth_dir data/test_clear --num_images 50`
- **No formal test framework** - scripts output results directly

## Code Style
- **Imports**: Standard library first, then third-party (torch, cv2, PIL, diffusers, sklearn), grouped logically
- **Formatting**: Use ruff (cache exists). Line length ~100 chars. 4-space indents
- **Types**: Not strictly enforced - use where it improves clarity
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- **Functions**: Descriptive names (e.g., `add_realistic_rain`, `derain_image`, `calculate_metrics`)
- **Docstrings**: Simple triple-quoted strings explaining purpose
- **Error handling**: Print messages for missing files/paths, use `exists()` checks before file ops
- **File paths**: Use `pathlib.Path` for path manipulation
- **Progress**: Use `tqdm` for loops processing multiple items
- **CLI args**: Use `argparse` with descriptive help text and sensible defaults

## Project Structure
- Root-level Python scripts for each major operation (prepare, run, test, visualize)
- `data/` contains train/test splits by weather condition
- `results/` for output images and metrics
