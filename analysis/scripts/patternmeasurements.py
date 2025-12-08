from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# If you don't have scikit-image yet:  pip install scikit-image
from skimage import morphology
from skimage.morphology import (
    disk,
    binary_closing,
    binary_opening,
    remove_small_objects,
    remove_small_holes,
    skeletonize,
)
from scipy.ndimage import binary_fill_holes
from skimage.measure import label

from scipy.ndimage import binary_fill_holes

from skimage.measure import label  
import scipy.ndimage as ndi  # you may already have this imported


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(
    r"D:\Work\Exp-Data\Capillary-fracturing\data\all-data\controlled-flow-rate-0_01ml_min\phi_58\all_images"
)

# Fracture-only frame to analyse
FRACTURE_NAME = "DSC_2627.JPG"

# Reference frames (no fractures)
REF_NAMES = ["DSC_2584.JPG", "DSC_2585.JPG", "DSC_2586.JPG"]

# Circle (same as in preprocessing)
cx = 1715
cy = 1288
radius_px = 1200

# ---- Hard-coded analysis parameters (adjust here) ----
THRESH_FRAC = 3.9        # absolute-difference threshold (in raw diff units)
MIN_CLUSTER_AREA = 220   # px, min area for components to keep (first pass)


# =========================================================
# 1. HELPER: LOAD, MASK, CROP
# =========================================================

def load_mask_crop(fname: str) -> np.ndarray:
    """
    Load image as grayscale, apply the circular mask (white outside),
    and crop to a square of side ~ 2*radius centred on (cx, cy).
    """
    path = BASE_DIR / fname
    img_gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise RuntimeError(f"Could not load image: {path}")

    h, w = img_gray.shape

    # circular mask on full image
    Y_full, X_full = np.ogrid[:h, :w]
    dist2 = (X_full - cx) ** 2 + (Y_full - cy) ** 2
    mask_full = dist2 <= radius_px**2

    # mask outside circle to white (255)
    img_masked = np.full_like(img_gray, 255)
    img_masked[mask_full] = img_gray[mask_full]

    # crop to square around centre
    half_side = int(radius_px)
    x0 = max(int(cx - half_side), 0)
    x1 = min(int(cx + half_side), w)
    y0 = max(int(cy - half_side), 0)
    y1 = min(int(cy + half_side), h)

    img_cropped = img_masked[y0:y1, x0:x1]

    return img_cropped


def stretch_to_uint8(img: np.ndarray,
                     p_lo: float = 2.0,
                     p_hi: float = 98.0) -> np.ndarray:
    """
    Stretch intensities to 0–255 for visualisation only.
    Uses percentile clipping [p_lo, p_hi] to ignore outliers.
    """
    img_f = img.astype(np.float32)
    lo = np.percentile(img_f.ravel(), p_lo)
    hi = np.percentile(img_f.ravel(), p_hi)
    out = (img_f - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    return (255 * out).astype(np.uint8)


# =========================================================
# 2. BUILD AVERAGED REFERENCE (CROPPED SPACE)
# =========================================================

ref_stack = []
for name in REF_NAMES:
    img_c = load_mask_crop(name).astype(np.float32)
    ref_stack.append(img_c)

ref_avg = np.mean(ref_stack, axis=0)  # float32, masked + cropped
h_c, w_c = ref_avg.shape
print("Reference (avg) shape:", ref_avg.shape)

# circular mask in cropped coordinates
Yc, Xc = np.ogrid[:h_c, :w_c]
dist2_c = (Xc - w_c / 2.0) ** 2 + (Yc - h_c / 2.0) ** 2
mask_crop = dist2_c <= radius_px**2      # True inside the disk


# =========================================================
# 3. LOAD FRACTURE IMAGE AND COMPUTE ABSOLUTE DIFFERENCE
# =========================================================

img_frac = load_mask_crop(FRACTURE_NAME).astype(np.float32)

# Absolute difference: detects both bright and dark fractures
diff_frac_f = np.abs(ref_avg - img_frac)
diff_frac_f[~mask_crop] = 0.0  # outside circle -> 0

print("FRAC diff (float): min", float(diff_frac_f.min()),
      "max", float(diff_frac_f.max()),
      "mean", float(diff_frac_f.mean()))


# =========================================================
# 4. THRESHOLD + MIN-CLUSTER FILTER
# =========================================================

def filter_clusters_by_size(bin_mask: np.ndarray,
                            min_area: int) -> np.ndarray:
    """
    Keep only connected components with area >= min_area.
    bin_mask: uint8 0/255
    Returns uint8 0/255 mask.
    """
    m = (bin_mask > 0).astype(np.uint8)  # 0/1
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        m, connectivity=8
    )

    if n_labels <= 1:
        return (m * 255).astype(np.uint8)  # nothing but background

    out = np.zeros_like(m, dtype=np.uint8)
    for i in range(1, n_labels):  # skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 1

    return (out * 255).astype(np.uint8)


# Threshold (raw diff units)
binary_thresh = (diff_frac_f > THRESH_FRAC).astype(np.uint8) * 255
binary_thresh[~mask_crop] = 0  # enforce circular region

# Filter by cluster size
filtered_mask = filter_clusters_by_size(binary_thresh, MIN_CLUSTER_AREA)




# =========================================================
# 6. PLOTS FOR QUICK INSPECTION
# =========================================================

diff_disp = stretch_to_uint8(diff_frac_f)

fig, axes = plt.subplots(1, 2, figsize=(10, 10))





axes[0].imshow(filtered_mask, cmap="gray", vmin=0, vmax=255)
axes[0].set_title(f"Filtered clusters (min area = {MIN_CLUSTER_AREA})")
axes[0].axis("off")
axes[0].set_aspect("equal")

axes[1].imshow(open, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Final skeleton")
axes[1].axis("off")
axes[1].set_aspect("equal")

plt.tight_layout()
plt.show()

# === VISUALISATION: thresholded vs filtered clusters (side-by-side) ===

# ensure we have uint8 images for display
bin_u8 = (binary_thresh.astype(np.uint8) if binary_thresh.dtype != np.uint8
          else binary_thresh)
filtered_u8 = (filtered_mask.astype(np.uint8) if filtered_mask.dtype != np.uint8
               else filtered_mask)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
ax0, ax1 = axes  # two side-by-side axes

ax0.imshow(bin_u8, cmap="gray", vmin=0, vmax=255)
ax0.set_title(f"Thresholded (T = {THRESH_FRAC})")
ax0.axis("off")
ax0.set_aspect("equal")

ax1.imshow(filtered_u8, cmap="gray", vmin=0, vmax=255)
ax1.set_title(f"Filtered clusters (min area = {MIN_CLUSTER_AREA})")
ax1.axis("off")
ax1.set_aspect("equal")

plt.tight_layout()
plt.show()

# =========================================================
# 7. OPTIONAL: SAVE RESULTS
# =========================================================
# out_dir = BASE_DIR / "analysis_outputs"
# out_dir.mkdir(exist_ok=True)
# cv2.imwrite(str(out_dir / "fracture_diff_stretched.png"), diff_disp)
# cv2.imwrite(str(out_dir / "fracture_binary_thresh.png"), binary_thresh)
# cv2.imwrite(str(out_dir / "fracture_filtered_mask.png"), filtered_mask)
# cv2.imwrite(str(out_dir / "fracture_main_cluster.png"), main_cluster)
# cv2.imwrite(str(out_dir / "fracture_skeleton.png"), skel)
