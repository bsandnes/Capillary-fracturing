from pathlib import Path

import cv2
import numpy as np
import os

# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(
    r"D:\Work\Exp-Data\Capillary-fracturing\data\all-data\controlled-flow-rate-0_01ml_min\phi_58\all_images"
)

# Reference frames (no fractures) to average
REF_NAMES = ["DSC_2584.JPG", "DSC_2585.JPG", "DSC_2586.JPG"]

# Frames you want to process
TARGET_NAMES = ["DSC_2587.JPG", "DSC_2627.JPG", "DSC_2717.JPG", "DSC_2867.JPG"]

# Circle parameters (same as before)
cx = 1715
cy = 1288
radius_px = 1200

# Output directory
OUT_DIR = BASE_DIR.parent / "bgsub_cropped_masked"
OUT_DIR.mkdir(exist_ok=True)


# =========================================================
# HELPERS
# =========================================================

def load_and_crop_gray(fname: str) -> np.ndarray:
    """
    Load image as grayscale and crop a square centred on (cx, cy)
    with side length ~ 2*radius_px.
    No masking here; just cropping.
    """
    path = BASE_DIR / fname
    img_gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise RuntimeError(f"Could not load image: {path}")

    h, w = img_gray.shape

    half_side = int(radius_px)
    x0 = max(int(cx - half_side), 0)
    x1 = min(int(cx + half_side), w)
    y0 = max(int(cy - half_side), 0)
    y1 = min(int(cy + half_side), h)

    img_cropped = img_gray[y0:y1, x0:x1]
    return img_cropped


# =========================================================
# 1. BUILD AVERAGED BACKGROUND (IN CROPPED SPACE)
# =========================================================

ref_stack = []
for name in REF_NAMES:
    img_c = load_and_crop_gray(name).astype(np.float32)
    ref_stack.append(img_c)

ref_avg = np.mean(ref_stack, axis=0).astype(np.float32)  # float32 average
h_c, w_c = ref_avg.shape
print("Reference (avg) shape:", ref_avg.shape)

# Build circular mask in cropped coordinates
Yc, Xc = np.ogrid[:h_c, :w_c]
dist2_c = (Xc - w_c / 2.0) ** 2 + (Yc - h_c / 2.0) ** 2
mask_bool = dist2_c <= radius_px**2     # True inside the disk
mask_alpha = (mask_bool.astype(np.uint8) * 255)  # 255 inside, 0 outside


# =========================================================
# 2. PROCESS EACH TARGET IMAGE
#    Background subtraction + circular mask + PNG with transparency
# =========================================================

for name in TARGET_NAMES:
    img_c = load_and_crop_gray(name).astype(np.float32)

    if img_c.shape != ref_avg.shape:
        raise ValueError(f"Size mismatch between {name} and background average.")

    # Simple absolute difference background subtraction
    diff = np.abs(ref_avg - img_c)

    # Optionally rescale to 0–255 for 8-bit PNG
    # Here we use min/max over the diff for this image
    d_min, d_max = float(diff.min()), float(diff.max())
    if d_max > d_min:
        diff_norm = (diff - d_min) / (d_max - d_min)
    else:
        diff_norm = np.zeros_like(diff, dtype=np.float32)

    diff_u8 = (255 * np.clip(diff_norm, 0.0, 1.0)).astype(np.uint8)

    # Apply circular mask to intensity (set outside circle to 0)
    diff_u8_masked = diff_u8.copy()
    diff_u8_masked[~mask_bool] = 0

    # Build RGBA image for PNG with transparency:
    # RGB from diff_u8_masked, alpha from mask_alpha
    rgb = cv2.cvtColor(diff_u8_masked, cv2.COLOR_GRAY2BGR)
    rgba = np.dstack([rgb, mask_alpha])

    out_name = Path(name).stem + "_bgsub_masked.png"
    out_path = OUT_DIR / out_name

    cv2.imwrite(str(out_path), rgba)
    print(f"Saved: {out_path}")

print("Done.")
