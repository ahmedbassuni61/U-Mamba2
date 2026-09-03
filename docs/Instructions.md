# 🦷 U-Mamba2: Project & Context Documentation

## 📌 Overview
This repository contains the local clone of **U-Mamba2**, a state-space model (SSM) architecture for 3D dental anatomy segmentation in CBCT volumes. It is built on top of nnUNet v2.6.2 and integrates Mamba2 blocks for efficient global long-range dependency modelling.

**Key capabilities:**
- **Task 1 (ToothFairy3):** Full semantic segmentation of teeth, nerves, sinuses, canals, and pulp
- **Task 2 (ToothFairy3):** Interactive segmentation with user-provided clicks (cross-attention + SAM-style point encoder)
- **STSR 2025:** Semi-supervised learning with consistency regularization and pseudo-labeling

## 💻 Environment & Workflow Constraints
**CRITICAL AGENT INSTRUCTION:**
* **Documentation Rule:** This `Instructions.md` document is the source of truth and must be updated regularly to reflect the current project status and completed tasks.
* **Execution Environment:** Code is primarily being executed on **Google Colab** for inference.
* **File Storage:** The repository resides **locally** on the user's PC. Model weights and CBCT data may be on Google Drive.
* **Modification Rule:** DO NOT destructively modify the local repository files. Colab-specific scripts and wrappers go in the `docs/` folder.

## 🧬 Current Objective: CBCT Inference on Colab
The goal is to run U-Mamba2 inference on dental CBCT scans using Google Colab's GPU runtime, leveraging pretrained weights from the ToothFairy3 challenge.

**Phase 1: Environment Setup (Colab)**
1. Clone the U-Mamba2 repository in Colab
2. Install all dependencies (PyTorch, nnUNet, mamba-ssm, causal-conv1d)
3. Set nnUNet environment variables
4. Download pretrained model weights

**Phase 2: Data Preparation**
1. Upload CBCT scans in `.nii.gz` format
2. Ensure naming follows nnUNet convention: `{case_id}_0000.nii.gz`

**Phase 3: Inference**
1. Load the pretrained U-Mamba2 model
2. Run sliding-window inference with optional TTA (test-time augmentation via mirroring)
3. Apply post-processing (volume thresholds, connected component analysis)
4. Save and visualise segmentation results

## 🎯 Downstream Context
The segmentation outputs from U-Mamba2 serve as inputs for:
- **VR Haptic Dental Simulation** — Segmented structures (teeth, nerves, pulp) drive haptic force-feedback in a dental training simulator
- **Treatment Planning** — Accurate anatomy segmentation supports implant planning and orthodontic analysis
- **Integration with 3DTeethSAM** — Complementary pipeline for surface mesh segmentation (see sister repo)

## 📂 Repository Structure
```
U-Mamba2/
├── docs/                          # ← YOUR DOCS (Colab guides, this file)
│   ├── Instructions.md            # This file — project context & status
│   ├── colab_cbct_inference_guide.md  # Step-by-step Colab guide
│   └── checklist.md               # Pre-flight checklist for Colab runs
├── documentation/                 # Original nnUNet + U-Mamba2 docs
│   ├── competitions/
│   │   ├── Toothfairy3/           # Challenge code & inference scripts
│   │   ├── STSR25/
│   │   └── Pretrain_DAE/
│   ├── dataset_format.md
│   ├── how_to_use_nnunet.md
│   └── ...
├── nnunetv2/                      # Core nnUNet v2 + U-Mamba2 code
│   ├── nets/                      # U-Mamba2 network architectures
│   │   ├── UMambaBot_3d.py        # Main U-Mamba2 model
│   │   ├── attention.py           # Cross-attention blocks
│   │   └── point_encoder.py       # SAM-style point encoder
│   ├── inference/                 # nnUNet inference pipeline
│   ├── training/                  # nnUNet trainers
│   └── ...
├── pyproject.toml                 # Dependencies & entry points
├── readme.md                      # Main project README
└── setup.py
```

## 🔗 Key Links
| Resource | URL |
|----------|-----|
| Pretrained Weights | [Google Drive](https://drive.google.com/drive/folders/1xhUkHCpo_50sNWvGH9CrN8Ws0hSjoa_k?usp=sharing) |
| U-Mamba2 Paper | [arXiv:2509.12069](https://arxiv.org/abs/2509.12069) |
| U-Mamba2-SSL Paper | [arXiv:2509.20154](https://arxiv.org/abs/2509.20154) |
| ToothFairy3 Challenge | [grand-challenge.org](https://toothfairy3.grand-challenge.org/) |
| ToothFairy3 Dataset | [ditto.ing.unimore.it](https://ditto.ing.unimore.it/toothfairy3/) |

## 🚀 Current Status & Agent Tasks
*(Update this section before prompting the agent with specific tasks)*

* [x] Repository cloned locally.
* [x] `docs/` folder created with inference guide.
* [x] Colab CBCT inference guide written (`docs/colab_cbct_inference_guide.md`).
* [x] Pre-flight checklist created (`docs/checklist.md`).
* [ ] Create Colab notebook (`.ipynb`) for end-to-end inference.
* [ ] Test inference on a sample CBCT scan.
* [ ] Add visualization utilities (3D Slicer export, multi-planar views).
* [ ] Document Task 2 (interactive click-based) inference for Colab.
