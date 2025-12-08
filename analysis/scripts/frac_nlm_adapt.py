from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.morphology import (
    disk,
    remove_small_objects,
    binary_closing,
    binary_opening,
    skeletonize,
)
from scipy.ndimage import binary_fill_holes


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(
    r"D:\Work\Exp-Data\Capillary-fracturing\data\all-data"
      r"\controlled-flow-rate-0_01ml_min\phi_58\all_images"
)

# Frames
FRACTURE_NAME = "DSC_2627.JPG"             # fracture frame to analyse
REF_NAMES     = ["DSC_2584.JPG",
                 "DSC_2585.JPG",
                 "DSC_2586.JPG"]          # reference frames (no fractures)

# Circle (same as before)
cx = 1715
cy = 1288
radius_px = 1200

# ---- Tunable parameters (start with these) ----
# Percentile scaling for diff image -> 0–255
P_LO, P_HI = 2.0, 98.0

# Non-local means denoising
NLM_H = 10                 # filter strength (8–12 reasonable)
NLM_TEMPLATE = 7
NLM_SEARCH   = 21

# Adaptive threshold
ADAPT_BLOCK = 51           # neighbourhood size (must be odd)
ADAPT_C     = -5           # subtract from local mean (more negative => more pixels white)

# Post-threshold cleaning
MIN_CLUSTER_AREA = 200     # px, remove tiny specks
CLOSE_RADIUS      = 2      # px, for closing (bridging tiny gaps)
OPEN_RADIUS       = 1      # px, for final de-hairing
MIN_SKELETON_SPUR = 10     # px, remove tiny skeleton fragments


# =========================================================
# 1. HELPERS
# =========================================================

def load_mask_crop(fname: str) -> np.ndarray:
    """
    Load image as grayscale, apply circular mask (white outside),
    crop to square of side ~ 2*radius centred on (cx, cy).
    """
    path = BASE_DIR / fname
    img_gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise RuntimeError(f"Could not load image: {path}")

    h, w = img_gray.shape

    # full-image circular mask
    Y_full, X_full = np.ogrid[:h, :w]
    dist2 = (X_full - cx) ** 2 + (Y_full - cy) ** 2
    mask_full = dist2 <= radius_px**2

    # outside circle -> white
    img_masked = np.full_like(img_gray, 255)
    img_masked[mask_full] = img_gray[mask_full]

    # crop around centre
    half_side = int(radius_px)
    x0 = max(int(cx - half_side), 0)
    x1 = min(int(cx + half_side), w)
    y0 = max(int(cy - half_side), 0)
    y1 = min(int(cy + half_side), h)

    img_cropped = img_masked[y0:y1, x0:x1]

    return img_cropped


def stretch_to_uint8(img: np.ndarray,
                     p_lo: float = P_LO,
                     p_hi: float = P_HI) -> np.ndarray:
    """
    Linear stretch to 0–255 for visualisation/processing.
    Uses percentile clipping [p_lo, p_hi] to ignore outliers.
    """
    img_f = img.astype(np.float32)
    lo = np.percentile(img_f.ravel(), p_lo)
    hi = np.percentile(img_f.ravel(), p_hi)
    out = (img_f - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    return (255 * out).astype(np.uint8)


def filter_clusters_by_size(bin_mask: np.ndarray,
                            min_area: int) -> np.ndarray:
    """
    Keep only connected components with area >= min_area.
    bin_mask: uint8 0/255
    Returns uint8 0/255.
    """
    m = (bin_mask > 0).astype(np.uint8)  # 0/1

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        m, connectivity=8
    )
    if n_labels <= 1:
        return (m * 255).astype(np.uint8)

    out = np.zeros_like(m, dtype=np.uint8)
    for i in range(1, n_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 1

    return (out * 255).astype(np.uint8)


# =========================================================
# 2. BUILD AVERAGED REFERENCE (CROPPED)
# =========================================================

ref_stack = []
for name in REF_NAMES:
    img_c = load_mask_crop(name).astype(np.float32)
    ref_stack.append(img_c)

ref_avg = np.mean(ref_stack, axis=0)
h_c, w_c = ref_avg.shape
print("Reference (avg) shape:", ref_avg.shape)

# circular mask in cropped coordinates
Yc, Xc = np.ogrid[:h_c, :w_c]
dist2_c = (Xc - w_c / 2.0) ** 2 + (Yc - h_c / 2.0) ** 2
mask_crop = dist2_c <= radius_px**2   # True inside disk


# =========================================================
# 3. LOAD FRACTURE, ABSOLUTE DIFFERENCE, SCALE
# =========================================================

img_frac = load_mask_crop(FRACTURE_NAME).astype(np.float32)

# abs(subtraction) to kill uneven illumination
diff_frac_f = np.abs(ref_avg - img_frac)
diff_frac_f[~mask_crop] = 0.0

print("FRAC diff: min", float(diff_frac_f.min()),
      "max", float(diff_frac_f.max()),
      "mean", float(diff_frac_f.mean()))

# scale to 0–255 for denoising / threshold
diff_scaled = stretch_to_uint8(diff_frac_f)


# =========================================================
# 4. NON-LOCAL MEANS DENOISING
# =========================================================

den = cv2.fastNlMeansDenoising(
    diff_scaled,
    h=NLM_H,
    templateWindowSize=NLM_TEMPLATE,
    searchWindowSize=NLM_SEARCH,
)

# outside disk -> 0
den[~mask_crop] = 0


# =========================================================
# 5. ADAPTIVE THRESHOLD + CLEANING
# =========================================================

# Adaptive threshold (fractures -> white)
bin_adapt = cv2.adaptiveThreshold(
    den,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    blockSize=ADAPT_BLOCK,
    C=ADAPT_C,
)
bin_adapt[~mask_crop] = 0

# remove tiny specks
bin_clean = filter_clusters_by_size(bin_adapt, MIN_CLUSTER_AREA)


# =========================================================
# 6. GAP BRIDGING, HOLE FILL, SKELETON
# =========================================================

# closing to bridge small gaps
se_close = disk(CLOSE_RADIUS)
bridged = binary_closing(bin_clean > 0, se_close)

# fill holes within branches
filled = binary_fill_holes(bridged)

# light opening to shave off side hairs
se_open = disk(OPEN_RADIUS)
smoothed = binary_opening(filled, se_open)

# skeleton
skel = skeletonize(smoothed)

# prune tiny skeleton fragments
skel = remove_small_objects(skel, min_size=MIN_SKELETON_SPUR)

# convert to uint8 for plotting / saving
bridged_u8  = (bridged.astype(np.uint8) * 255)
filled_u8   = (filled.astype(np.uint8) * 255)
smoothed_u8 = (smoothed.astype(np.uint8) * 255)
skel_u8     = (skel.astype(np.uint8) * 255)


# =========================================================
# 7. PLOTS – STAGE-BY-STAGE
# =========================================================

fig, axes = plt.subplots(2, 3, figsize=(14, 9))

axes[0, 0].imshow(diff_scaled, cmap="gray")
axes[0, 0].set_title("Abs diff (scaled)")
axes[0, 0].axis("off")
axes[0, 0].set_aspect("equal")

axes[0, 1].imshow(den, cmap="gray")
axes[0, 1].set_title("Non-local means denoised")
axes[0, 1].axis("off")
axes[0, 1].set_aspect("equal")

axes[0, 2].imshow(bin_adapt, cmap="gray", vmin=0, vmax=255)
axes[0, 2].set_title("Adaptive threshold")
axes[0, 2].axis("off")
axes[0, 2].set_aspect("equal")

axes[1, 0].imshow(bin_clean, cmap="gray", vmin=0, vmax=255)
axes[1, 0].set_title(f"Cleaned (min area = {MIN_CLUSTER_AREA})")
axes[1, 0].axis("off")
axes[1, 0].set_aspect("equal")

axes[1, 1].imshow(smoothed_u8, cmap="gray", vmin=0, vmax=255)
axes[1, 1].set_title("Bridged + filled + smoothed")
axes[1, 1].axis("off")
axes[1, 1].set_aspect("equal")

axes[1, 2].imshow(skel_u8, cmap="gray", vmin=0, vmax=255)
axes[1, 2].set_title("Final skeleton")
axes[1, 2].axis("off")
axes[1, 2].set_aspect("equal")

plt.tight_layout()
plt.show()

# =========================================================
# 8. OPTIONAL: SAVE RESULTS
# =========================================================
# out_dir = BASE_DIR / "analysis_outputs_nlm_adapt"
# out_dir.mkdir(exist_ok=True)
# cv2.imwrite(str(out_dir / "fracture_diff_scaled.png"), diff_scaled)
# cv2.imwrite(str(out_dir / "fracture_den.png"), den)
# cv2.imwrite(str(out_dir / "fracture_bin_adapt.png"), bin_adapt)
# cv2.imwrite(str(out_dir / "fracture_bin_clean.png"), bin_clean)
# cv2.imwrite(str(out_dir / "fracture_smoothed.png"), smoothed_u8)
# cv2.imwrite(str(out_dir / "fracture_skeleton.png"), skel_u8)
