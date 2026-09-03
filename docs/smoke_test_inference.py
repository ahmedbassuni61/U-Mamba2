"""
U-Mamba2 Smoke-Test Inference Script
====================================
Runs inference on a synthetic phantom using SSL pretrained weights directly.

Unlike task1_inference.py (which needs a fully-trained nnUNet checkpoint with
plans/dataset_json/trainer metadata), this script manually constructs the
UMamba2 architecture, loads the raw SSL pretrained weights, and runs a forward
pass — just to verify the pipeline doesn't crash.

The segmentation output will be NONSENSICAL because these weights were trained
with self-supervised learning (DAE), not supervised segmentation. That's fine —
this is a smoke test.

Usage (Kaggle / Colab):
    python docs/smoke_test_inference.py \
        --weights "/kaggle/working/model_weights/UMamba2 pretrained weights/ssl_umamba2_3d_depth7_128x256x256.pth" \
        --patch_size 128 256 256 \
        --output_dir /kaggle/working/cbct_output
"""

import os
import sys
import argparse
import gc
import time
import numpy as np
import torch

# ─── Synthetic phantom generator ──────────────────────────────────────────────
def create_synthetic_phantom(shape=(160, 256, 256), spacing=(0.4, 0.4, 0.4)):
    """Create a fake CBCT volume with jaw, teeth, nerve canal, and soft tissue."""
    import nibabel as nib

    vol = np.full(shape, -1000.0, dtype=np.float32)
    D, H, W = shape
    cz, cy, cx = D // 2, H // 2, W // 2
    zz, yy, xx = np.mgrid[:D, :H, :W]

    # Jaw bone block
    jaw_mask = (
        (zz >= cz - 30) & (zz <= cz + 30) &
        (yy >= cy - 60) & (yy <= cy + 60) &
        (xx >= cx - 90) & (xx <= cx + 90)
    )
    vol[jaw_mask] = np.random.normal(700, 80, jaw_mask.sum())

    # Teeth: 14 spheres in a horseshoe arch
    tooth_radius = 8
    arch_rx, arch_ry = 70, 45
    for angle in np.linspace(-np.pi * 0.85, np.pi * 0.85, 14):
        tx = int(cx + arch_rx * np.sin(angle))
        ty = int(cy - arch_ry * np.cos(angle))
        dist = np.sqrt((zz - cz)**2 + (yy - ty)**2 + (xx - tx)**2)
        enamel = (dist <= tooth_radius) & (dist >= tooth_radius - 3)
        dentine = (dist <= tooth_radius) & (dist < tooth_radius - 3)
        vol[enamel] = np.random.normal(2500, 200, enamel.sum())
        vol[dentine] = np.random.normal(1200, 150, dentine.sum())

    # Nerve canal (cylinder)
    canal_cy, canal_cz = cy + 20, cz + 10
    canal_mask = ((yy - canal_cy)**2 + (zz - canal_cz)**2) <= 16
    canal_mask &= jaw_mask
    vol[canal_mask] = np.random.normal(30, 20, canal_mask.sum())

    # Soft tissue
    soft_mask = ~jaw_mask & (
        (zz >= cz - 50) & (zz <= cz + 50) &
        (yy >= cy - 80) & (yy <= cy + 80) &
        (xx >= cx - 110) & (xx <= cx + 110)
    )
    vol[soft_mask] = np.random.normal(40, 30, soft_mask.sum())

    # Gaussian noise
    vol += np.random.normal(0, 15, shape).astype(np.float32)

    return vol, spacing


# ─── Model builder ────────────────────────────────────────────────────────────
def build_umamba2(patch_size, num_input_channels=1, num_classes=47, deep_supervision=False):
    """Build a UMamba2 network with the ToothFairy3 Task-1 architecture."""
    from nnunetv2.nets.UMambaBot_3d import UMambaBot2

    # Decide strides based on patch size (last stage stride differs)
    if patch_size == [160, 288, 288]:
        last_stride = [1, 1, 1]
    else:  # 128x256x256 or similar
        last_stride = [1, 2, 2]

    arch_kwargs = {
        "n_stages": 7,
        "features_per_stage": [32, 64, 128, 256, 320, 320, 320],
        "conv_op": torch.nn.modules.conv.Conv3d,
        "kernel_sizes": [[3,3,3]] * 7,
        "strides": [[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2], [2,2,2], last_stride],
        "n_blocks_per_stage": [1, 3, 4, 6, 6, 6, 6],
        "n_conv_per_stage_decoder": [1, 1, 1, 1, 1, 1],
        "conv_bias": True,
        "norm_op": torch.nn.modules.instancenorm.InstanceNorm3d,
        "norm_op_kwargs": {"eps": 1e-05, "affine": True},
        "dropout_op": None,
        "dropout_op_kwargs": None,
        "nonlin": torch.nn.LeakyReLU,
        "nonlin_kwargs": {"inplace": True},
        "input_channels": num_input_channels,
        "num_classes": num_classes,
        "deep_supervision": deep_supervision,
    }

    model = UMambaBot2(**arch_kwargs)
    return model


# ─── Weight loader ────────────────────────────────────────────────────────────
def load_ssl_weights(model, weights_path, device='cuda'):
    """
    Load SSL pretrained weights into the model.

    Handles both checkpoint formats:
      - DAE pretrain format: checkpoint['model']
      - nnUNet format:       checkpoint['network_weights']
    """
    print(f"Loading weights from: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    # Figure out which key holds the state dict
    if 'network_weights' in checkpoint:
        state_dict = checkpoint['network_weights']
        fmt = 'nnUNet'
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        fmt = 'DAE/SSL'
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        fmt = 'generic'
    else:
        # Assume the checkpoint IS the state dict
        state_dict = checkpoint
        fmt = 'raw state_dict'

    print(f"  Checkpoint format: {fmt}")
    print(f"  Keys in checkpoint: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'N/A'}")
    print(f"  Number of weight tensors: {len(state_dict)}")

    # Strip common prefixes from DDP / torch.compile
    cleaned = {}
    for k, v in state_dict.items():
        clean_key = k.replace('module.', '').replace('_orig_mod.', '')
        cleaned[clean_key] = v

    # Load with strict=False so mismatched seg_layers / missing keys don't crash
    msg = model.load_state_dict(cleaned, strict=False)
    print(f"  Missing keys:    {len(msg.missing_keys)}")
    print(f"  Unexpected keys: {len(msg.unexpected_keys)}")
    if msg.missing_keys:
        print(f"  (first few missing): {msg.missing_keys[:5]}")
    if msg.unexpected_keys:
        print(f"  (first few unexpected): {msg.unexpected_keys[:5]}")

    return model


# ─── Simple sliding-window inference ─────────────────────────────────────────
@torch.inference_mode()
def sliding_window_inference(model, volume, patch_size, step_size=0.5, device='cuda'):
    """
    Run sliding-window inference on a 3D volume.

    Args:
        model: the segmentation network
        volume: numpy array of shape (D, H, W)
        patch_size: list of [d, h, w]
        step_size: overlap fraction (0.5 = 50% overlap)
    """
    from scipy.ndimage import gaussian_filter

    D, H, W = volume.shape
    pd, ph, pw = patch_size
    sd = max(1, int(pd * step_size))
    sh = max(1, int(ph * step_size))
    sw = max(1, int(pw * step_size))

    # Pad volume if needed
    pad_d = max(0, pd - D)
    pad_h = max(0, ph - H)
    pad_w = max(0, pw - W)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        volume = np.pad(volume, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant', constant_values=-1000)

    Dp, Hp, Wp = volume.shape

    # Get number of output classes from model
    num_classes = model.decoder.seg_layers[-1].out_channels if hasattr(model, 'decoder') else 47

    # Accumulators
    output_sum = np.zeros((num_classes, Dp, Hp, Wp), dtype=np.float32)
    count = np.zeros((Dp, Hp, Wp), dtype=np.float32)

    # Gaussian weighting for patch
    gaussian_weight = np.ones(patch_size, dtype=np.float32)
    gaussian_weight = gaussian_filter(gaussian_weight, sigma=[s/8 for s in patch_size])
    gaussian_weight = gaussian_weight / gaussian_weight.max()

    # Generate slice positions
    d_starts = list(range(0, Dp - pd + 1, sd))
    h_starts = list(range(0, Hp - ph + 1, sh))
    w_starts = list(range(0, Wp - pw + 1, sw))

    if not d_starts or d_starts[-1] + pd < Dp:
        d_starts.append(max(0, Dp - pd))
    if not h_starts or h_starts[-1] + ph < Hp:
        h_starts.append(max(0, Hp - ph))
    if not w_starts or w_starts[-1] + pw < Wp:
        w_starts.append(max(0, Wp - pw))

    total_patches = len(d_starts) * len(h_starts) * len(w_starts)
    print(f"  Running {total_patches} patches ({len(d_starts)}×{len(h_starts)}×{len(w_starts)})")

    patch_idx = 0
    for di in d_starts:
        for hi in h_starts:
            for wi in w_starts:
                patch_idx += 1
                patch = volume[di:di+pd, hi:hi+ph, wi:wi+pw]
                # Add batch + channel dims: (1, 1, D, H, W)
                x = torch.from_numpy(patch[None, None]).to(device=device, dtype=torch.float32)

                with torch.autocast(device, enabled=True):
                    pred = model(x)  # (1, C, D, H, W)

                pred = pred[0].float().cpu().numpy()  # (C, D, H, W)
                output_sum[:, di:di+pd, hi:hi+ph, wi:wi+pw] += pred * gaussian_weight[None]
                count[di:di+pd, hi:hi+ph, wi:wi+pw] += gaussian_weight

                if patch_idx % 5 == 0 or patch_idx == total_patches:
                    print(f"    Patch {patch_idx}/{total_patches}", end='\r')

    print()

    # Average
    count = np.maximum(count, 1e-8)
    output_sum /= count[None]

    # Argmax → segmentation
    seg = output_sum.argmax(axis=0).astype(np.uint8)

    # Crop back to original size
    seg = seg[:D, :H, :W]

    return seg


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='U-Mamba2 Smoke-Test Inference')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to SSL pretrained .pth weights file')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[128, 256, 256],
                        help='Patch size [D H W] (default: 128 256 256)')
    parser.add_argument('--num_classes', type=int, default=47,
                        help='Number of output classes (default: 47 for ToothFairy3 Task1)')
    parser.add_argument('--input_nifti', type=str, default=None,
                        help='Path to input .nii.gz (if omitted, generates synthetic phantom)')
    parser.add_argument('--output_dir', type=str, default='./smoke_test_output',
                        help='Output directory')
    parser.add_argument('--step_size', type=float, default=0.75,
                        help='Sliding window step size as fraction of patch (default: 0.75)')
    parser.add_argument('--no_viz', action='store_true',
                        help='Skip visualization')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # ── Step 1: Prepare input volume ──
    if args.input_nifti and os.path.exists(args.input_nifti):
        import nibabel as nib
        print(f"\n📂 Loading input: {args.input_nifti}")
        nii = nib.load(args.input_nifti)
        volume = nii.get_fdata().astype(np.float32)
        spacing = nii.header.get_zooms()[:3]
    else:
        print("\n🧪 Generating synthetic phantom...")
        volume, spacing = create_synthetic_phantom()

    print(f"  Shape: {volume.shape}")
    print(f"  Spacing: {spacing}")
    print(f"  Intensity range: [{volume.min():.0f}, {volume.max():.0f}]")

    # ── Step 2: Build model ──
    print(f"\n🏗️  Building UMamba2 (patch={args.patch_size}, classes={args.num_classes})...")
    model = build_umamba2(
        patch_size=args.patch_size,
        num_input_channels=1,
        num_classes=args.num_classes,
        deep_supervision=False,
    )
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {total_params:.1f}M")

    # ── Step 3: Load weights ──
    print(f"\n📦 Loading pretrained weights...")
    model = load_ssl_weights(model, args.weights, device=device)
    model = model.to(device)
    model.eval()
    print("  ✅ Model ready")

    # ── Step 4: Run inference ──
    print(f"\n🔬 Running sliding-window inference (step_size={args.step_size})...")
    t0 = time.time()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    seg = sliding_window_inference(
        model, volume, args.patch_size,
        step_size=args.step_size, device=device,
    )

    elapsed = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3 if device == 'cuda' else 0
    print(f"  ⏱️  Time: {elapsed:.1f}s")
    print(f"  💾 Peak GPU memory: {peak_mem:.2f} GB")

    # ── Step 5: Save output ──
    import nibabel as nib
    affine = np.diag([*spacing, 1.0])
    out_nii = nib.Nifti1Image(seg, affine)
    out_path = os.path.join(args.output_dir, 'phantom_seg.nii.gz')
    nib.save(out_nii, out_path)
    print(f"\n💾 Saved segmentation → {out_path}")

    unique_labels = np.unique(seg)
    print(f"  Unique labels ({len(unique_labels)}): {unique_labels}")
    for lbl in unique_labels[:10]:
        count = (seg == lbl).sum()
        print(f"    Label {lbl:3d}: {count:>10,} voxels ({100*count/seg.size:.2f}%)")

    # ── Step 6: Visualize ──
    if not args.no_viz:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            D, H, W = volume.shape
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            slices = {
                'Axial':    (volume[D//2], seg[D//2]),
                'Coronal':  (volume[:, H//2], seg[:, H//2]),
                'Sagittal': (volume[:, :, W//2], seg[:, :, W//2]),
            }
            for col, (title, (s, g)) in enumerate(slices.items()):
                axes[0, col].imshow(s, cmap='bone', vmin=-1000, vmax=3000)
                axes[0, col].set_title(f'{title} — Input')
                axes[1, col].imshow(s, cmap='bone', vmin=-1000, vmax=3000)
                axes[1, col].imshow(g, cmap='nipy_spectral', alpha=0.45, interpolation='nearest')
                axes[1, col].set_title(f'{title} — Segmentation')
            for ax in axes.flat:
                ax.axis('off')
            plt.suptitle(
                f'Smoke Test  |  {elapsed:.1f}s  |  {peak_mem:.1f} GB  |  {len(unique_labels)} labels',
                fontsize=14
            )
            plt.tight_layout()
            viz_path = os.path.join(args.output_dir, 'smoke_test_viz.png')
            plt.savefig(viz_path, dpi=150)
            print(f"  📊 Visualization → {viz_path}")
            plt.close()
        except ImportError:
            print("  (matplotlib not available, skipping viz)")

    print("\n✅ Smoke test complete!")


if __name__ == '__main__':
    main()
