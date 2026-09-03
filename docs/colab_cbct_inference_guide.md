# 🦷 U-Mamba2 CBCT Inference on Google Colab

## 📌 Overview
This guide walks you through running **U-Mamba2** inference on CBCT (Cone-Beam Computed Tomography) dental scans using **Google Colab**. U-Mamba2 is the first-place solution from KCL TAIR Lab for the ODIN challenges (ToothFairy3 & STSR 2025), combining U-Net with Mamba2 for efficient dental anatomy segmentation.

**What you'll be doing:**
1. Setting up the Colab environment with all dependencies
2. Uploading your CBCT `.nii.gz` scan(s)
3. Loading pretrained U-Mamba2 weights
4. Running inference to produce segmentation masks
5. Downloading / visualising results

---

## 💻 Environment & Workflow Constraints

> [!IMPORTANT]
> - **Execution Environment:** Code runs on **Google Colab** (GPU runtime required — T4 minimum, A100 recommended for large scans).
> - **File Storage:** The repository and pretrained weights are cloned / downloaded into the Colab VM's ephemeral storage. Mount Google Drive if you need persistence.
> - **Modification Rule:** Do NOT destructively modify the upstream repository. All Colab-specific wrappers go in this `docs/` folder or dedicated notebooks.

---

## 🔧 Prerequisites

| Item | Details |
|------|---------|
| **Google Account** | For Colab access |
| **CBCT Scans** | `.nii.gz` format (single-channel, as used in ToothFairy3) |
| **Pretrained Weights** | Download from the [Google Drive folder](https://drive.google.com/drive/folders/1xhUkHCpo_50sNWvGH9CrN8Ws0hSjoa_k?usp=sharing) |
| **GPU Runtime** | Colab → Runtime → Change runtime type → **T4 GPU** (or A100 if available) |

---

## 🚀 Step-by-Step Colab Setup

### Step 0 — Select GPU Runtime
In Colab: **Runtime → Change runtime type → GPU (T4 or A100)**

Verify:
```python
!nvidia-smi
```

### Step 1 — Clone the Repository
```python
!git clone https://github.com/zhiqin1998/U-Mamba2.git
%cd U-Mamba2
```

### Step 2 — Install Dependencies
```python
# Install PyTorch (Colab usually has it, but ensure CUDA version)
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install U-Mamba2 (nnUNet fork) in editable mode
!pip install -e .

# Install Mamba2 dependencies (CRITICAL — requires --no-build-isolation)
!pip install causal-conv1d==1.5.2 mamba-ssm==2.2.5 --no-build-isolation

# Install additional inference dependencies
!pip install cc3d
```

> [!WARNING]
> The `mamba-ssm` build requires a CUDA-capable GPU runtime to be active during installation.
> If you get build errors, ensure you selected a GPU runtime **before** running pip install.

### Step 3 — Set nnUNet Environment Variables
```python
import os

# Create directory structure on Colab VM
os.makedirs('/content/nnUNet_raw', exist_ok=True)
os.makedirs('/content/nnUNet_preprocessed', exist_ok=True)
os.makedirs('/content/nnUNet_results', exist_ok=True)

os.environ['nnUNet_raw'] = '/content/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '/content/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '/content/nnUNet_results'
```

### Step 4 — Download Pretrained Weights

**Option A — From Google Drive (recommended):**
```python
# Mount your Google Drive if weights are stored there
from google.colab import drive
drive.mount('/content/drive')

# Copy weights from your Drive
!cp -r "/content/drive/MyDrive/U-Mamba2_weights/" /content/model_weights/
```

**Option B — Direct download via gdown:**
```python
!pip install gdown

# Download the pretrained weights folder
# Replace <FILE_ID> with the actual Google Drive file ID
!gdown --folder https://drive.google.com/drive/folders/1xhUkHCpo_50sNWvGH9CrN8Ws0hSjoa_k -O /content/model_weights/
```

> [!NOTE]
> The weights folder should contain a `fold_all/` (or `fold_0/`) subdirectory with `checkpoint_best.pth` or `checkpoint_final.pth`.
> Structure expected:
> ```
> /content/model_weights/
> └── fold_all/
>     └── checkpoint_best.pth
> ```

### Step 5 — Prepare Input Data

#### Option A — Generate a Synthetic Phantom (Recommended for First Test)

No real CBCT scan? Create a fake one. The code below builds a 3D NIfTI volume with
geometry that loosely mimics a dental CBCT: a dense jaw bone block, an arch of
sphere-shaped "teeth" embedded in it, a cylindrical "nerve canal", and air/soft-tissue
background — all at HU-like intensity values.

```python
import numpy as np
import nibabel as nib

# ---------- volume parameters ----------
shape = (160, 256, 256)        # D × H × W (fits the default patch size)
spacing = (0.4, 0.4, 0.4)     # mm – typical CBCT voxel size
input_dir = '/content/cbct_input'
os.makedirs(input_dir, exist_ok=True)

vol = np.full(shape, -1000.0, dtype=np.float32)   # air background (≈ HU)

D, H, W = shape
cz, cy, cx = D // 2, H // 2, W // 2   # volume centre

# coordinate grids
zz, yy, xx = np.mgrid[:D, :H, :W]

# --- 1. Jaw bone block (dense rectangle in the lower half) ---
jaw_mask = (
    (zz >= cz - 30) & (zz <= cz + 30) &
    (yy >= cy - 60) & (yy <= cy + 60) &
    (xx >= cx - 90) & (xx <= cx + 90)
)
vol[jaw_mask] = np.random.normal(700, 80, jaw_mask.sum())  # cortical-bone HU

# --- 2. Teeth: 14 spheres in a horseshoe arch ---
n_teeth = 14
tooth_radius = 8   # voxels
arch_radius_x = 70
arch_radius_y = 45
angles = np.linspace(-np.pi * 0.85, np.pi * 0.85, n_teeth)

teeth_centres = []
for angle in angles:
    tx = int(cx + arch_radius_x * np.sin(angle))
    ty = int(cy - arch_radius_y * np.cos(angle))
    tz = cz
    teeth_centres.append((tz, ty, tx))

    dist = np.sqrt((zz - tz)**2 + (yy - ty)**2 + (xx - tx)**2)
    tooth_mask = dist <= tooth_radius
    # enamel shell (outer 3 voxels) vs dentine core
    enamel = tooth_mask & (dist >= tooth_radius - 3)
    dentine = tooth_mask & (dist < tooth_radius - 3)
    vol[enamel] = np.random.normal(2500, 200, enamel.sum())   # enamel HU
    vol[dentine] = np.random.normal(1200, 150, dentine.sum()) # dentine HU

# --- 3. Nerve canal (cylinder running left-right through the jaw) ---
canal_cy, canal_cz = cy + 20, cz + 10
canal_r = 4
canal_mask = ((yy - canal_cy)**2 + (zz - canal_cz)**2) <= canal_r**2
canal_mask &= jaw_mask  # only inside the jaw
vol[canal_mask] = np.random.normal(30, 20, canal_mask.sum())  # soft-tissue HU

# --- 4. Soft tissue fill around the jaw ---
soft_mask = ~jaw_mask & (
    (zz >= cz - 50) & (zz <= cz + 50) &
    (yy >= cy - 80) & (yy <= cy + 80) &
    (xx >= cx - 110) & (xx <= cx + 110)
)
vol[soft_mask] = np.random.normal(40, 30, soft_mask.sum())

# --- 5. Add slight Gaussian noise everywhere ---
vol += np.random.normal(0, 15, shape).astype(np.float32)

# ---------- save as NIfTI ----------
affine = np.diag([*spacing, 1.0])
nii = nib.Nifti1Image(vol, affine)
out_path = os.path.join(input_dir, 'phantom_0000.nii.gz')
nib.save(nii, out_path)
print(f'✅ Saved synthetic phantom → {out_path}')
print(f'   Shape: {vol.shape}, dtype: {vol.dtype}')
print(f'   Intensity range: [{vol.min():.0f}, {vol.max():.0f}] HU')
print(f'   Teeth centres (first 3): {teeth_centres[:3]}')
```

> [!TIP]
> This phantom is **not anatomically accurate** — the model will produce noisy / nonsensical
> segmentation labels on it. That's fine! The purpose is to **smoke-test the full pipeline**
> (imports → model loading → sliding-window inference → post-processing → NIfTI export)
> and verify nothing crashes before you spend time uploading real data.

**Quick sanity-check visualisation of the phantom:**
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].imshow(vol[cz, :, :], cmap='bone', vmin=-1000, vmax=3000)
axes[0].set_title(f'Axial  (z={cz})')
axes[1].imshow(vol[:, cy, :], cmap='bone', vmin=-1000, vmax=3000)
axes[1].set_title(f'Coronal  (y={cy})')
axes[2].imshow(vol[:, :, cx], cmap='bone', vmin=-1000, vmax=3000)
axes[2].set_title(f'Sagittal  (x={cx})')
for ax in axes:
    ax.axis('off')
plt.suptitle('Synthetic CBCT Phantom', fontsize=14)
plt.tight_layout()
plt.show()
```

---

#### Option B — Upload a Real CBCT Scan

```python
from google.colab import files
uploaded = files.upload()  # Select your .nii.gz file(s)

import shutil
input_dir = '/content/cbct_input'
os.makedirs(input_dir, exist_ok=True)
for fname in uploaded.keys():
    shutil.move(fname, os.path.join(input_dir, fname))
```

#### Option C — Copy from Google Drive
```python
input_dir = '/content/cbct_input'
os.makedirs(input_dir, exist_ok=True)

!cp "/content/drive/MyDrive/CBCT_Scans/your_scan_0000.nii.gz" {input_dir}/
```

> [!IMPORTANT]
> **Naming convention:** Input files MUST follow nnUNet format: `{case_id}_0000.nii.gz`
> - The `_0000` suffix indicates the first (and only) imaging modality.
> - Example: `patient001_0000.nii.gz`, `scan_abc_0000.nii.gz`

---

## 🔬 Running Inference

### Option 1 — Smoke Test with SSL Pretrained Weights (Start Here)

If you only have the **SSL pretrained weights** (the `ssl_umamba2_3d_depth7_*.pth` files),
use the smoke-test script. It manually builds the model, loads raw weights, generates a
synthetic phantom, and runs inference — no trained nnUNet checkpoint needed:

```bash
%cd /kaggle/working/U-Mamba2

# Quote the path because the folder name has spaces!
!python docs/smoke_test_inference.py \
    --weights "/kaggle/working/model_weights/UMamba2 pretrained weights/ssl_umamba2_3d_depth7_128x256x256.pth" \
    --patch_size 128 256 256 \
    --num_classes 47 \
    --output_dir /kaggle/working/cbct_output
```

> [!NOTE]
> The segmentation output will be **nonsensical** since these are self-supervised
> pretraining weights, not fine-tuned segmentation weights. The purpose is to verify
> the full pipeline (model build → weight load → sliding-window → NIfTI export) works.

To run on your own `.nii.gz` instead of the phantom, add `--input_nifti`:
```bash
!python docs/smoke_test_inference.py \
    --weights "/kaggle/working/model_weights/UMamba2 pretrained weights/ssl_umamba2_3d_depth7_128x256x256.pth" \
    --input_nifti /kaggle/working/cbct_input/your_scan_0000.nii.gz \
    --output_dir /kaggle/working/cbct_output
```

---

### Option 2 — Using the Task 1 Inference Script (Fully-Trained Model Required)

> [!WARNING]
> This requires a **fully-trained nnUNet checkpoint** (produced by `nnUNetv2_train`),
> NOT the SSL pretrained `.pth` files. The checkpoint must live in a folder structure like:
> ```
> model_weights/
> └── fold_all/
>     └── checkpoint_best.pth   ← contains trainer_name, plans, dataset_json, network_weights
> ```

```python
%cd /kaggle/working/U-Mamba2

!python documentation/competitions/Toothfairy3/task1_inference.py \
    --base_model /kaggle/working/model_weights \
    --folds all \
    -i /kaggle/working/cbct_input \
    -o /kaggle/working/cbct_output
```

### Option 3 — Minimal Python Inference (Fully-Trained Model Required)

```python
import os
os.environ['nnUNet_raw'] = '/content/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '/content/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '/content/nnUNet_results'

import torch
import numpy as np
from pathlib import Path
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

# ---- Import the custom predictor from the competition code ----
import sys
sys.path.insert(0, '/content/U-Mamba2/documentation/competitions/Toothfairy3')
from task1_inference import BasePredictor, predict_semseg

# ---- Define L-R mapping (for TTA mirroring) ----
mapping = [(3, 4), (5, 6), (43, 44)]
for l, r in zip(range(19, 27), range(11, 19)):
    mapping.append((l, r))
for l, r in zip(range(27, 35), range(35, 43)):
    mapping.append((l, r))

# ---- Initialize predictor ----
predictor = BasePredictor(
    tile_step_size=0.95,
    use_mirroring=True,     # Set False for faster (but less accurate) inference
    use_gaussian=True,
    perform_everything_on_device=True,
    allow_tqdm=True,
    tta_batch_size=1,
    lr_mapping=mapping,
    n_class=47,
    verbose=True,
)

predictor.initialize_from_trained_model_folder(
    '/content/model_weights',
    use_folds=['all'],
    checkpoint_name='checkpoint_best.pth',
)

# ---- Run inference ----
rw = SimpleITKIO()
input_path = Path('/content/cbct_input')
output_path = Path('/content/cbct_output/predictions')
output_path.mkdir(parents=True, exist_ok=True)

for nii_file in sorted(input_path.glob('*_0000.nii.gz')):
    print(f'Processing: {nii_file.name}')
    im, prop = rw.read_images([nii_file])
    print(f'  Shape: {im.shape}, Spacing: {prop["spacing"]}')

    seg = predict_semseg(im, prop, predictor)

    out_name = nii_file.name.replace('_0000.nii.gz', '.nii.gz')
    rw.write_seg(seg, output_path / out_name, prop)
    print(f'  Saved → {out_name}')

    del im, prop, seg
    torch.cuda.empty_cache()

print('✅ Inference complete!')
```

---

## 📊 Visualizing Results

```python
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Load the original scan and segmentation
# (replace 'phantom' with your actual case_id if using real data)
scan = nib.load('/content/cbct_input/phantom_0000.nii.gz').get_fdata()
seg  = nib.load('/content/cbct_output/predictions/phantom.nii.gz').get_fdata()

D, H, W = scan.shape
slices = {
    'Axial':    (scan[D//2, :, :],  seg[D//2, :, :]),
    'Coronal':  (scan[:, H//2, :],  seg[:, H//2, :]),
    'Sagittal': (scan[:, :, W//2],  seg[:, :, W//2]),
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for col, (title, (s, g)) in enumerate(slices.items()):
    axes[0, col].imshow(s, cmap='gray')
    axes[0, col].set_title(f'{title} — Input')
    axes[1, col].imshow(s, cmap='gray')
    axes[1, col].imshow(g, cmap='nipy_spectral', alpha=0.45, interpolation='nearest')
    axes[1, col].set_title(f'{title} — Overlay')
for ax in axes.flat:
    ax.axis('off')

unique_labels = np.unique(seg)
plt.suptitle(f'Inference Result  •  {len(unique_labels)} unique labels: {unique_labels[:10]}...', fontsize=13)
plt.tight_layout()
plt.savefig('/content/cbct_output/visualization.png', dpi=150)
plt.show()
print(f'Unique labels in segmentation: {unique_labels}')
```

---

## 📥 Downloading Results

```python
# Download the output segmentation
from google.colab import files

# Single file
files.download('/content/cbct_output/predictions/phantom.nii.gz')

# Or zip everything
!zip -r /content/cbct_output.zip /content/cbct_output/
files.download('/content/cbct_output.zip')
```

---

## ⚡ Performance Tips

| Tip | Effect |
|-----|--------|
| `use_mirroring=False` | ~2x faster inference, slight accuracy drop |
| `explicit_half=True` | Uses FP16 throughout, reduces VRAM ~40% |
| `tile_step_size=0.8` | More overlap between patches, better accuracy at edges |
| Reduce patch size | If OOM: edit plan JSON to use `[128, 256, 256]` instead of `[160, 288, 288]` |
| A100 runtime | Significantly faster than T4 for large 3D volumes |

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `mamba-ssm` build fails | Ensure GPU runtime is active before `pip install` |
| CUDA OOM during inference | Set `use_mirroring=False`, or use smaller `tile_step_size` |
| `ModuleNotFoundError: nnunetv2` | Run `pip install -e .` from the U-Mamba2 directory |
| Wrong output shape / metadata | Ensure input follows `{id}_0000.nii.gz` naming convention |
| `causal-conv1d` version conflict | Pin exactly: `causal-conv1d==1.5.2` |
| NaN in predictions | Uncomment `@autocast('cuda', enabled=False)` in `UMambaBot_3d.py` |

---

## 📂 Output Label Mapping (ToothFairy3 Task 1)

The segmentation output uses the FDI tooth numbering system. Key label groups:

| Label Range | Structure |
|-------------|-----------|
| 11–18 | Upper right teeth |
| 21–28 | Upper left teeth |
| 31–38 | Lower left teeth |
| 41–48 | Lower right teeth |
| 50 | Combined pulp |
| 51 | Inferior alveolar nerve (IAN) |
| 52 | Maxillary sinus |
| 53 | Incisive canal |

---

## 🔗 Related Documentation
- [Main README](../readme.md) — Project overview & full setup
- [ToothFairy3 Competition Details](../documentation/competitions/Toothfairy3/readme.md) — Training & evaluation
- [nnUNet Data Format](../documentation/dataset_format.md) — How to prepare datasets
- [Setting Up Paths](../documentation/setting_up_paths.md) — nnUNet environment variables
