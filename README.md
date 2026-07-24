# DTGS

## Installation

### 1. Environment

- Python 3.8+
- CUDA 11.7+ (GPU required)
- PyTorch 1.13.1

### 2. Clone and install dependencies

```bash
git clone <repository-url>
cd DTGS

pip install -r requirements.txt
```

### 3. Install CUDA rasterizer and KNN

DTGS depends on two custom CUDA extensions from the 3DGS ecosystem:

```bash
# Gaussian rasterizer
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization

# Simple KNN
pip install git+https://github.com/graphdeco-inria/simple-knn
```

> **Note:** These submodules require a CUDA toolkit (`nvcc`) installed on your system.

---

## Data Preparation

### Dynamic_LTR Dataset

Download the **Dynamic_LTR** dataset and place it under the `data/` directory.

> **Note:** The Dynamic_LTR dataset is currently being organized and will be released in mid-August.

```
DTGS/
├── data/
│   └── D_thermal/
│       ├── scene1/
│       │   ├── images/          # Thermal infrared frames
│       │   ├── depths/          # Depth maps (optional)
│       │   ├── masks/           # Binary masks (can be generated via make_mask.py)
│       │   ├── sparse/
│       │   │   └── 0/           # COLMAP sparse reconstruction
│       │   └── cameras.npz      # Camera parameters
│       ├── scene2/
│       └── ...
```

### ZJU-MoCap Dataset

To use the ZJU-MoCap dataset, organize the data as:

```
DTGS/
├── data/
│   └── ZJU/
│       ├── scene8/
│       └── ...
```

### Generating Masks

If masks are not provided, use `make_mask.py` to generate all-black single-channel mask images:

```bash
# Edit the input/output paths in make_mask.py first, then run:
python make_mask.py
```

---

## Usage

### 1. Training

```bash
python train_thermal.py
```

By default, this trains on `D_thermal/scene1`. To switch scenes or datasets, edit the following lines in `train_thermal.py`:

```python
exp_name = "D_thermal/scene1"
source_path = "data/D_thermal/scene1"

# For ZJU dataset:
# exp_name = "ZJU/scene8"
# source_path = "data/ZJU/scene8"
```

**Key training parameters** (passed via config or CLI):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--iterations` | 60000 | Total training iterations |
| `--test_iterations` | `[0, 500, 1000, ...]` | Iterations at which to evaluate PSNR |
| `--save_iterations` | `[2000, 2500, 5000, 7000, 10000]` | Iterations at which to save point cloud |
| `--configs` | `arguments/endonerf/default.py` | Hyperparameter config file (mmcv format) |
| `--expname` | `D_thermal/scene1` | Experiment name (output subfolder) |

Trained models and point clouds are saved to:

```
output/<exp_name>/
├── point_cloud/
│   └── iteration_<N>/
│       └── point_cloud.ply
├── input.ply
└── cfg_args
```

### 2. Rendering

Render images, depth maps, videos, and reconstructed point clouds from a trained model:

```bash
python render.py \
    --model_path output/D_thermal/scene1 \
    --skip_train \
    --skip_video \
    --reconstruct_test \
    --configs arguments/endonerf/default.py
```

**Rendering options:**

| Flag | Description |
|------|-------------|
| `--model_path` | Path to the trained model directory |
| `--iteration` | Checkpoint iteration to load (`-1` = latest) |
| `--skip_train` | Skip rendering training views |
| `--skip_test` | Skip rendering test views |
| `--skip_video` | Skip rendering video sequence |
| `--reconstruct_train` | Reconstruct point clouds from training views |
| `--reconstruct_test` | Reconstruct point clouds from test views |
| `--reconstruct_video` | Reconstruct point clouds from video views |
| `--configs` | Hyperparameter config file |

Rendered outputs are saved to:

```
output/<exp_name>/
├── train/
│   └── ours_<iteration>/
│       ├── renders/      # Rendered images
│       ├── gt/           # Ground-truth images
│       ├── depth/        # Rendered depth maps
│       └── ...
├── test/
│   └── ours_<iteration>/
│       └── ...
├── video/
│   └── ours_<iteration>/
│       ├── ours_video.mp4
│       └── gt_video.mp4
└── reconstruct/          # Reconstructed point clouds
```

### 3. Evaluation

Compute PSNR, SSIM, LPIPS, and RMSE metrics on rendered results:

```bash
python metrics.py \
    --model_path output/D_thermal/scene1 \
    -p test
```

| Flag | Description |
|------|-------------|
| `--model_path` / `-m` | Path(s) to the trained model directory |
| `--phase` / `-p` | Evaluation phase: `train`, `test`, or `video` |

Results are saved as JSON files under the model directory.

---

## Project Structure

```
DTGS/
├── train_thermal.py              # Training entry point
├── render.py                     # Rendering & point cloud reconstruction
├── metrics.py                    # Evaluation (PSNR / SSIM / LPIPS / RMSE)
├── make_mask.py                  # Mask generation utility
├── ply_color.py                  # PLY color utility
├── test.py                       # Image dimension testing utility
├── requirements.txt              # Python dependencies
│
├── gaussian_renderer/
│   └── __init__.py               # Rendering pipeline (render_flow)
│
├── scene/
│   ├── __init__.py               # Scene class — camera & point cloud management
│   ├── flexible_deform_model.py  # GaussianModel with deformation model
│   ├── cameras.py                # Camera & MiniCam classes
│   ├── dataset_readers.py        # Dataset loading entry (D_thermal / ZJU)
│   ├── thermal_loader.py         # Dthermal_Dataset — thermal data processing
│   ├── colmap_loader.py          # COLMAP format camera parameter parsing
│   ├── regulation.py             # Regularization losses
│   └── utils.py                  # Camera geometry utilities
│
├── utils/
│   ├── loss_utils.py             # Loss functions (L1, SSIM, etc.)
│   ├── general_utils.py          # LR scheduling, quaternion ops, image utils
│   ├── graphics_utils.py         # Projection matrices, BasicPointCloud
│   ├── image_utils.py            # psnr, rmse, mse metrics
│   ├── camera_utils.py           # Camera loading & JSON serialization
│   ├── sh_utils.py               # Spherical harmonic evaluation
│   ├── scene_utils.py            # Training visualization
│   ├── system_utils.py           # Directory & iteration utilities
│   ├── params_utils.py           # Hyperparameter merging (mmcv config)
│   ├── timer.py                  # Training timer
│   └── stereo_rectify.py         # Stereo rectification utilities
│
├── lpipsPyTorch/                 # LPIPS perceptual loss implementation
│   ├── __init__.py
│   └── modules/
│       ├── lpips.py
│       ├── networks.py
│       └── utils.py
│
└── arguments/                    # (External) mmcv config files
    └── endonerf/
        └── default.py            # Default hyperparameters
```

> **Note:** The `arguments/` directory containing config files (e.g., `arguments/endonerf/default.py`) is not included in this repository by default. Ensure you have the appropriate config file before running training or rendering.
