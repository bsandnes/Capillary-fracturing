from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(
    r"D:\Work\Exp-Data\Capillary-fracturing\data\all-data\controlled-flow-rate-0_01ml_min\phi_58\all_images"
)

# Reference frames (no fractures)
REF_NAMES = ["DSC_2584.JPG", "DSC_2585.JPG", "DSC_2586.JPG"]

# Example frames
FRACTURE_NAME = "DSC_2627.JPG"   # fracture only
IP_NAME       = "DSC_2861.JPG"   # fracture + pore invasion

USE_IP_IMAGE = True  # set False if you only care about fracture frame

# Circle geometry in full-resolution image
cx = 1715
cy = 1288
radius_px = 1200


# =========================================================
# 1. BUILD CROP INDICES AND MASK (ONCE)
# =========================================================

# Read one ref to get image size
tmp_ref = cv2.imread(str(BASE_DIR / REF_NAMES[0]), cv2.IMREAD_GRAYSCALE)
if tmp_ref is None:
    raise RuntimeError("Could not load reference image to initialise mask")

h_full, w_full = tmp_ref.shape

# Full image circular mask
Y_full, X_full = np.ogrid[:h_full, :w_full]
dist2_full = (X_full - cx) ** 2 + (Y_full - cy) ** 2
mask_full = dist2_full <= radius_px**2  # bool

# Crop indices for a square around the centre
half_side = int(radius_px)
x0 = max(int(cx - half_side), 0)
x1 = min(int(cx + half_side), w_full)
y0 = max(int(cy - half_side), 0)
y1 = min(int(cy + half_side), h_full)

# Cropped circular mask (bool)
mask_crop = mask_full[y0:y1, x0:x1]
print("Cropped mask shape:", mask_crop.shape)


# =========================================================
# 2. HELPER: LOAD, MASK, CROP
# =========================================================

def load_mask_crop(fname: str) -> np.ndarray:
    """
    Load image as grayscale, apply circular mask in full space,
    then crop to the same square region.
    """
    path = BASE_DIR / fname
    img_gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise RuntimeError(f"Could not load image: {path}")

    # apply circular mask: outside circle -> white
    img_masked = np.full_like(img_gray, 255)
    img_masked[mask_full] = img_gray[mask_full]

    # crop
    img_cropped = img_masked[y0:y1, x0:x1]
    return img_cropped


# =========================================================
# 3. BUILD AVERAGED REFERENCE (CROPPED)
# =========================================================

ref_stack = []
for name in REF_NAMES:
    img_c = load_mask_crop(name).astype(np.float32)
    ref_stack.append(img_c)

ref_avg = np.mean(ref_stack, axis=0)  # float32
print("Reference (avg, cropped) shape:", ref_avg.shape)


# =========================================================
# 4. LOAD EXAMPLE FRAMES AND COMPUTE ABS DIFF
# =========================================================

img_frac = load_mask_crop(FRACTURE_NAME).astype(np.float32)

if USE_IP_IMAGE:
    img_frac_ip = load_mask_crop(IP_NAME).astype(np.float32)
else:
    img_frac_ip = None

# Absolute difference so both brighter and darker fractures show up
diff_frac = np.abs(ref_avg - img_frac)

if USE_IP_IMAGE:
    diff_frac_ip = np.abs(ref_avg - img_frac_ip)
else:
    diff_frac_ip = None

# Outside the circle: set to zero so it never thresholds as signal
diff_frac[~mask_crop] = 0
if USE_IP_IMAGE:
    diff_frac_ip[~mask_crop] = 0




# =========================================================
# 6. INTERACTIVE THRESHOLD + HISTOGRAM
# =========================================================

def interactive_threshold(diff_img: np.ndarray,
                          mask: np.ndarray,
                          title_prefix: str = "Interactive threshold"):
    """
    Interactive thresholding tool operating on raw diff image values (no stretching).
      - Left panel: binary image (inside circle)
      - Right panel: histogram of raw diff_img (inside circle) with
        a vertical line at the current raw threshold.
    """
    img_f = diff_img.astype(np.float32)

    # pixel values inside mask (raw units)
    vals_raw = img_f[mask]
    if vals_raw.size == 0:
        print("Mask contains no pixels — nothing to threshold.")
        return

    lo = float(vals_raw.min())
    hi = float(vals_raw.max())
    if hi <= lo:
        hi = lo + 1.0

    # sensible display vmax for visualising the diff (99th percentile)
    vmax = float(np.percentile(vals_raw, 99))
    vmax = max(vmax, 1e-6)

    # initial threshold (raw units)
    T0_raw = float(np.median(vals_raw))

    # initial binary image (raw threshold)
    bin0 = (img_f > T0_raw).astype(np.uint8) * 255
    bin0[~mask] = 0

    # figure layout: binary image + histogram
    fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(bottom=0.22)  # space for slider

    im = ax_img.imshow(bin0, cmap="gray", vmin=0, vmax=255)
    ax_img.set_title(f"{title_prefix}\nRaw T = {T0_raw:.2f}")
    ax_img.axis("off")
    ax_img.set_aspect("equal")

    # histogram of raw values
    ax_hist.hist(vals_raw.ravel(), bins=256, range=(lo, hi), color="0.7")
    ax_hist.set_xlabel("Raw diff value")
    ax_hist.set_ylabel("Count")

    # vertical line showing current raw threshold
    thresh_line = ax_hist.axvline(T0_raw, color="r")

    # slider: operate directly in raw units (float)
    ax_slider = plt.axes([0.20, 0.06, 0.60, 0.03])
    slider = Slider(ax_slider, "Threshold (raw)", lo, hi, valinit=T0_raw)

    # update callback: threshold raw image and update binary display + histogram line
    def update(val):
        rawT = float(slider.val)
        bin_img = (img_f > rawT).astype(np.uint8) * 255
        bin_img[~mask] = 0
        im.set_data(bin_img)
        ax_img.set_title(f"{title_prefix}\nRaw T = {rawT:.2f}")
        thresh_line.set_xdata([rawT, rawT])
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Also show a small inset of the diff image (for context) using vmax
    ax_inset = fig.add_axes([0.02, 0.70, 0.18, 0.25])
    ax_inset.imshow(img_f, cmap="gray", vmin=0.0, vmax=vmax)
    ax_inset.set_title("Diff (99th pct)")
    ax_inset.axis("off")

    plt.show()


# =========================================================
# 7. RUN TOOLS
# =========================================================

if __name__ == "__main__":
    # Quick look at greyscale differences (raw) using masked vmax for visibility
    if USE_IP_IMAGE:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes_list = axes
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        axes_list = [ax]

    # compute vmax from masked values so small diffs are visible
    vals0 = diff_frac[mask_crop]
    vmax0 = float(np.percentile(vals0, 99)) if vals0.size else float(diff_frac.max())
    vmax0 = max(vmax0, 1e-6)

    axes_list[0].imshow(diff_frac, cmap="gray", vmin=0.0, vmax=vmax0)
    axes_list[0].set_title("Fracture abs diff (raw)")
    axes_list[0].axis("off")
    axes_list[0].set_aspect("equal")

    if USE_IP_IMAGE and diff_frac_ip is not None:
        vals1 = diff_frac_ip[mask_crop]
        vmax1 = float(np.percentile(vals1, 99)) if vals1.size else float(diff_frac_ip.max())
        vmax1 = max(vmax1, 1e-6)

        axes_list[1].imshow(diff_frac_ip, cmap="gray", vmin=0.0, vmax=vmax1)
        axes_list[1].set_title("Fracture + IP abs diff (raw)")
        axes_list[1].axis("off")
        axes_list[1].set_aspect("equal")

    plt.tight_layout()
    plt.show()

    # Interactive threshold for fracture pattern (raw units)
    interactive_threshold(diff_frac, mask_crop, title_prefix="Fracture frame")

    # And optionally for fracture + IP
    if USE_IP_IMAGE and diff_frac_ip is not None:
        interactive_threshold(diff_frac_ip, mask_crop, title_prefix="Fracture + IP frame")
