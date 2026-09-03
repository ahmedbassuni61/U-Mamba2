# ✅ U-Mamba2 Colab Inference — Pre-Flight Checklist

Use this checklist every time you start a new Colab session for CBCT inference.

---

## 🔌 Runtime Setup
- [ ] **GPU runtime selected** — Runtime → Change runtime type → GPU (T4/A100)
- [ ] **GPU verified** — Run `!nvidia-smi` and confirm CUDA is available
- [ ] **Sufficient disk space** — Run `!df -h` (need ~10 GB free for repo + weights + data)

## 📦 Installation
- [ ] **Repo cloned** — `!git clone https://github.com/zhiqin1998/U-Mamba2.git`
- [ ] **nnUNet installed** — `pip install -e .` from U-Mamba2 directory
- [ ] **mamba-ssm installed** — `pip install causal-conv1d==1.5.2 mamba-ssm==2.2.5 --no-build-isolation`
- [ ] **cc3d installed** — `pip install cc3d` (for connected component post-processing)
- [ ] **No import errors** — Run `import nnunetv2; from mamba_ssm import Mamba2`

## 🗂 Environment Variables
- [ ] `nnUNet_raw` set → `/content/nnUNet_raw`
- [ ] `nnUNet_preprocessed` set → `/content/nnUNet_preprocessed`
- [ ] `nnUNet_results` set → `/content/nnUNet_results`
- [ ] All three directories created with `os.makedirs(..., exist_ok=True)`

## 🏋️ Model Weights
- [ ] **Weights downloaded** to `/content/model_weights/`
- [ ] **Folder structure verified:**
  ```
  /content/model_weights/
  └── fold_all/          (or fold_0/)
      └── checkpoint_best.pth   (or checkpoint_final.pth)
  ```
- [ ] **File size looks right** — Checkpoint should be several hundred MB to ~1 GB

## 🩻 Input Data
- [ ] **CBCT scan(s) uploaded** to `/content/cbct_input/`
- [ ] **Naming convention** — Files named `{case_id}_0000.nii.gz` (the `_0000` suffix is mandatory)
- [ ] **File format** — `.nii.gz` (NIfTI compressed)
- [ ] **Single channel** — One `_0000` file per case (CBCT is single modality)

## 🧪 Quick Sanity Check
```python
# Run this cell to verify everything is ready
import os, torch
assert torch.cuda.is_available(), "❌ No GPU!"
assert os.path.isdir('/content/U-Mamba2'), "❌ Repo not cloned!"
assert os.path.exists(os.environ.get('nnUNet_raw', '')), "❌ nnUNet_raw not set!"

import sys
sys.path.insert(0, '/content/U-Mamba2')
import nnunetv2
from mamba_ssm import Mamba2
print("✅ All imports OK")

# Check weights
import glob
ckpts = glob.glob('/content/model_weights/**/checkpoint_*.pth', recursive=True)
assert len(ckpts) > 0, "❌ No checkpoints found!"
print(f"✅ Found {len(ckpts)} checkpoint(s): {ckpts}")

# Check input
inputs = glob.glob('/content/cbct_input/*_0000.nii.gz')
assert len(inputs) > 0, "❌ No input files found!"
print(f"✅ Found {len(inputs)} input scan(s): {[os.path.basename(f) for f in inputs]}")

print("\n🚀 Ready to run inference!")
```

## 🏃 Run Inference
- [ ] Predictor initialized without errors
- [ ] First scan processed successfully
- [ ] Output saved to `/content/cbct_output/predictions/`
- [ ] Visualization generated

## 📤 Export Results
- [ ] Results downloaded or saved to Google Drive
- [ ] GPU stats logged (time per case, memory usage)

---

## ⚠️ Common Gotchas

1. **Colab session timeout** — Long inference on many scans may exceed Colab's idle timeout. Keep the tab active or use Colab Pro.
2. **Disk space** — Large CBCT volumes + model weights can fill up the ~80 GB Colab disk. Delete intermediary files as needed.
3. **Re-installation on reconnect** — If your Colab runtime disconnects, you need to re-run ALL installation cells (the VM is wiped).
4. **mamba-ssm version** — Must match the CUDA version of the runtime. If builds fail, try `pip install mamba-ssm --no-build-isolation` without pinning.
