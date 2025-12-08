from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURATION
# =========================================================

# Reference image (no fractures)
REF_PATH = Path(
    r"D:\Work\Exp-Data\Capillary-fracturing\data\all-data\controlled-flow-rate-0_01ml_min\phi_58\all_images\DSC_2567.JPG"
)

# Base directory of the image series (same folder)
BASE_DIR = REF_PATH.parent


# Output directory for batch-processed images
OUTPUT_DIR = Path(
    r"D:\Work\Exp-Data\Capillary-fracturing\data\all-data\controlled-flow-rate-0_01ml_min\phi_58\all_images_batchtreated"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)





# ---------------------------------------------------------
# Background removal method selection
# ---------------------------------------------------------
# "poly"         -> elliptical polynomial only
# "morph"        -> morphological opening only
# "poly_morph"   -> elliptical polynomial + residual morph opening  
BG_MODE = "poly_morph"   # choose one

# ---------------------------------------------------------
# ENHANCEMENT SELECTION (after background correction)
# ---------------------------------------------------------
# "none"           -> no enhancement
# "clahe"          -> CLAHE only
# "unsharp"        -> gentle unsharp only
# "clahe_unsharp"  -> CLAHE first, then gentle unsharp
ENHANCE_MODE = "clahe_unsharp"




# A test frame with fractures / pore invasion
TEST_IMG_NAME = "DSC_2860.JPG"   # change if you like


# =========================================================
# 1. LOAD REFERENCE, BUILD MASK + CROP
# =========================================================

print("Using reference image:", REF_PATH)

img_gray = cv2.imread(str(REF_PATH), cv2.IMREAD_GRAYSCALE)
if img_gray is None:
    raise SystemExit("ERROR: could not load reference image")

h, w = img_gray.shape
print(f"Reference image size: width = {w} px, height = {h} px")

# --- circle centre and radius (pixels) ---
cx = 1715
cy = 1288
radius_px = 1200

print(f"Circle centre: ({cx:.1f}, {cy:.1f}) px")
print(f"Circle radius: {radius_px:.1f} px")

# --- full-image circular mask ---
Y_full, X_full = np.ogrid[:h, :w]
dist2 = (X_full - cx) ** 2 + (Y_full - cy) ** 2
mask_full = dist2 <= radius_px**2  # True inside circle

# --- masked reference (white outside circle) ---
img_masked_ref_full = np.full_like(img_gray, 255)
img_masked_ref_full[mask_full] = img_gray[mask_full]

# --- square crop centred on (cx, cy) with side ~ 2*radius ---
half_side = int(radius_px)
x0 = max(int(cx - half_side), 0)
x1 = min(int(cx + half_side), w)
y0 = max(int(cy - half_side), 0)
y1 = min(int(cy + half_side), h)

img_cropped_ref = img_masked_ref_full[y0:y1, x0:x1]
hc, wc = img_cropped_ref.shape

# cropped circle mask
mask_crop = mask_full[y0:y1, x0:x1]

print(f"Cropped size: width = {wc} px, height = {hc} px")


# =========================================================
# 2. BACKGROUND MODELS & Enhancement FUNCTIONs
# =========================================================

def compute_background_poly(img_c_ref, mask_crop, radius_px):
    """
    Very smooth elliptical polynomial background, fitted on the
    masked, cropped reference.
    Currently: quadratic with x, y, x^2, y^2 terms.
    """
    img_c_ref = img_c_ref.astype(np.float32)
    hc, wc = img_c_ref.shape

    # valid pixels: inside circle, exclude top band (LED glare)
    valid = mask_crop.astype(bool)
    exclude_rows = int(0.05 * hc)  # top 5 %
    valid[:exclude_rows, :] = False

    ys, xs = np.nonzero(valid)

    # normalised coords
    xn = (xs - wc / 2.0) / wc
    yn = (ys - hc / 2.0) / hc

    # elliptical quadratic:
    # f(x,y) = a0 + a1 x + a2 y + a3 x^2 + a4 y^2
    A = np.column_stack([
        np.ones_like(xn),
        xn,
        yn,
        xn**2,
        yn**2,
    ])
    b = img_c_ref[valid].ravel()

    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)

    # evaluate on full cropped grid
    Yc, Xc = np.mgrid[0:hc, 0:wc]
    Xn = (Xc - wc / 2.0) / wc
    Yn = (Yc - hc / 2.0) / hc

    bg_poly = (
        coeffs[0]
        + coeffs[1] * Xn
        + coeffs[2] * Yn
        + coeffs[3] * Xn**2
        + coeffs[4] * Yn**2
    )

    # light smoothing to avoid tiny ripples
    bg_smooth = cv2.GaussianBlur(
        bg_poly.astype(np.float32),
        (0, 0),
        sigmaX=radius_px / 4.0,
    )

    bg_med = float(np.median(bg_smooth[mask_crop == 1]))
    return bg_smooth, bg_med


def compute_background_morph_open(img_c_ref, mask_crop, radius_px):
    """
    Background via large-scale morphological opening on the
    masked, cropped reference. This version:
      - neutralises pixels outside the circle,
      - slightly blurs before opening,
      - uses a larger kernel so illumination is really smooth.
    """
    img_c_ref = img_c_ref.astype(np.float32)
    hc, wc = img_c_ref.shape

    # 1) Fill outside circle with interior median
    interior_med = float(np.median(img_c_ref[mask_crop == 1]))
    img_for_bg = img_c_ref.copy()
    img_for_bg[mask_crop == 0] = interior_med

    # 2) Light pre-blur to kill tiny speckle before the big opening
    pre_blur_sigma = radius_px / 10.0  # quite mild
    img_for_bg = cv2.GaussianBlur(
        img_for_bg, (0, 0), sigmaX=pre_blur_sigma
    )

    # 3) Large elliptical structuring element
    #    (bigger than before so it sees only very broad trends)
    max_dim = min(hc, wc)
    kernel_size = int(radius_px * 0.3)  # 
    kernel_size = min(kernel_size, max_dim - 1)
    if kernel_size < 3:
        kernel_size = 3
    if kernel_size % 2 == 0:
        kernel_size += 1  # must be odd

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    background_raw = cv2.morphologyEx(
        img_for_bg, cv2.MORPH_OPEN, kernel
    )

    # 4) Very light post-blur just to remove any “steppy” artefacts
    post_blur_sigma = radius_px / 6.0
    bg_smooth = cv2.GaussianBlur(
        background_raw.astype(np.float32),
        (0, 0),
        sigmaX=post_blur_sigma,
    )

    # 5) Median background level (inside circle) for brightness offset
    bg_med = float(np.median(bg_smooth[mask_crop == 1]))
    return bg_smooth, bg_med


def compute_background_poly_morph(img_c_ref, mask_crop, radius_px):
    """
    Combined background model: polynomial FIRST, then a small-scale
    morphological opening on the residual.

    Steps:
      1. Fit a smooth elliptical polynomial background (large-scale gradient).
      2. Correct the reference with that polynomial background to get a residual.
      3. On the residual, use a *small* morph opening to capture mid-scale,
         non-symmetric illumination (LED hotspots, shadows).
      4. Final background = poly_background + morph_residual_background.

    Returns:
        background_combined (float32 2D array)
        bg_med_combined     (float): median inside the circle
    """
    img_c_ref = img_c_ref.astype(np.float32)
    hc, wc = img_c_ref.shape

    # ---------- Step 1: large-scale elliptical polynomial background ----------
    bg_poly, bg_med_poly = compute_background_poly(
        img_c_ref, mask_crop, radius_px
    )

    # ---------- Step 2: residual after polynomial correction ----------
    # Same correction formula as we use later:
    # residual = image - (bg_poly - bg_med_poly)
    residual = img_c_ref.copy()
    residual[mask_crop == 1] = img_c_ref[mask_crop == 1] - (
        bg_poly[mask_crop == 1] - bg_med_poly
    )

    # ---------- Step 3: small-scale morph-opening on the residual ----------
    # Fill outside the circle with the interior median of the residual
    interior_med = float(np.median(residual[mask_crop == 1]))
    res_for_bg = residual.copy()
    res_for_bg[mask_crop == 0] = interior_med

    # Light pre-blur: we only want to see mid-scale bumps, not pixel noise
    pre_blur_sigma = radius_px / 20.0  # smaller than before
    res_for_bg = cv2.GaussianBlur(
        res_for_bg, (0, 0), sigmaX=pre_blur_sigma
    )

    # Small elliptical structuring element (tens of pixels, not hundreds)
    max_dim = min(hc, wc)
    kernel_size = int(radius_px * 0.05)  # ~60 px if radius ~1200
    kernel_size = min(kernel_size, max_dim - 1)
    if kernel_size < 3:
        kernel_size = 3
    if kernel_size % 2 == 0:
        kernel_size += 1  # must be odd

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    bg_residual_raw = cv2.morphologyEx(
        res_for_bg, cv2.MORPH_OPEN, kernel
    )

    # Mild post-blur to ensure smoothness
    post_blur_sigma = radius_px / 12.0
    bg_residual_smooth = cv2.GaussianBlur(
        bg_residual_raw.astype(np.float32),
        (0, 0),
        sigmaX=post_blur_sigma,
    )

    # Keep only the *variation* of this residual background (remove constant offset)
    res_med = float(np.median(bg_residual_smooth[mask_crop == 1]))
    bg_residual_zero = bg_residual_smooth - res_med

    # ---------- Step 4: combine polynomial background + residual morph background ----------
    background_combined = bg_poly + bg_residual_zero

    bg_med_combined = float(
        np.median(background_combined[mask_crop == 1])
    )

    return background_combined.astype(np.float32), bg_med_combined


def enhance_fractures_clahe(img_in: np.ndarray, mask_in: np.ndarray) -> np.ndarray:
    """
    Enhance the fracture pattern using CLAHE on the
    background-corrected, masked+cropped image.

    img_in : uint8, background-corrected image
    mask_in: 0/1 or bool mask for the circular region

    Returns:
        enhanced (uint8) image with fractures more visible.
    """
    # work on a copy
    img = img_in.copy().astype(np.uint8)

    # optional: very light blur to suppress tiny speckle before CLAHE
    # (sigma ~1 keeps real structures, kills single-pixel noise)
    img_smooth = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)

    # CLAHE parameters:
    # - clipLimit: how strong local contrast can get
    # - tileGridSize: size of "local neighbourhood" for histogram equalisation
    clahe = cv2.createCLAHE(
        clipLimit=2.0,      # try 1.5–3.0; higher = stronger local contrast
        tileGridSize=(16, 16)  # larger tiles = smoother; smaller = more local, more noise
    )

    clahe_full = clahe.apply(img_smooth)

    # keep outside-of-circle region unchanged (white ring etc.)
    enhanced = img.copy()
    mask_bool = (mask_in == 1)
    enhanced[mask_bool] = clahe_full[mask_bool]

    return enhanced.astype(np.uint8)




# =========================================================
# 3. COMPUTE BACKGROUND ON REFERENCE
# =========================================================


if BG_MODE == "poly":
    print("Using elliptical polynomial background.")
    background, bg_med = compute_background_poly(
        img_cropped_ref, mask_crop, radius_px
    )
elif BG_MODE == "morph":
    print("Using morphological opening background.")
    background, bg_med = compute_background_morph_open(
        img_cropped_ref, mask_crop, radius_px
    )
elif BG_MODE == "poly_morph":
    print("Using polynomial background + residual morph opening")
    background, bg_med = compute_background_poly_morph(
        img_cropped_ref, mask_crop, radius_px
    )
else:
    raise ValueError(f"Unknown BG_MODE: {BG_MODE}")


# preview: corrected reference (for sanity check)
img_c_ref = img_cropped_ref.astype(np.float32)
corr_ref = img_c_ref.copy()
corr_ref[mask_crop == 1] = img_c_ref[mask_crop == 1] - (
    background[mask_crop == 1] - bg_med
)
corr_ref = np.clip(corr_ref, 0, 255).astype(np.uint8)




# =========================================================
# 4. ENHANCEMENT METHODS (AFTER BACKGROUND CORRECTION)
# =========================================================

def enhance_clahe(img_in: np.ndarray, mask_in: np.ndarray) -> np.ndarray:
    """
    CLAHE applied mainly inside the circular region.
    We temporarily neutralise outside-of-circle so CLAHE
    does not get distracted by the LEDs / borders.
    """
    img = img_in.astype(np.uint8)
    mask_bool = (mask_in == 1)

    # Fill outside circle with interior median for the CLAHE step
    interior_med = int(np.median(img[mask_bool]))
    img_for = img.copy()
    img_for[~mask_bool] = interior_med

    # CLAHE parameters: mild but noticeable
    clahe = cv2.createCLAHE(
        clipLimit=2.0,          # increase for stronger contrast, decrease for gentler
        tileGridSize=(24, 24),  # larger tiles => smoother / less noisy
    )
    clahe_full = clahe.apply(img_for)

    # Put original outside-of-circle back
    out = img.copy()
    out[mask_bool] = clahe_full[mask_bool]

    return out.astype(np.uint8)


def enhance_unsharp(img_in: np.ndarray, mask_in: np.ndarray) -> np.ndarray:
    """
    Gentle unsharp masking, restricted to inside the circle.
    Designed to be mild, to avoid too much noise amplification.
    """
    img = img_in.astype(np.float32)
    mask_bool = (mask_in == 1)

    # Parameters: tune these if needed
    sigma = 3.0        # blur scale; larger => only broad variations removed
    amount = 0.3       # 0.2–0.4 is usually good here

    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    sharp = img + amount * (img - blur)

    out = img.copy()
    out[mask_bool] = sharp[mask_bool]

    return np.clip(out, 0, 255).astype(np.uint8)


def apply_enhancement(img_corrected: np.ndarray,
                      mask_in: np.ndarray) -> np.ndarray:
    """
    Wrapper that applies the chosen enhancement method to a
    background-corrected, masked+cropped frame.
    """
    if ENHANCE_MODE == "none":
        return img_corrected.copy()

    elif ENHANCE_MODE == "clahe":
        return enhance_clahe(img_corrected, mask_in)

    elif ENHANCE_MODE == "unsharp":
        return enhance_unsharp(img_corrected, mask_in)

    elif ENHANCE_MODE == "clahe_unsharp":
        # 1) CLAHE to bring out fractures
        tmp = enhance_clahe(img_corrected, mask_in)
        # 2) gentle unsharp on top
        return enhance_unsharp(tmp, mask_in)

    else:
        raise ValueError(f"Unknown ENHANCE_MODE: {ENHANCE_MODE}")



def apply_levels(img: np.ndarray,
                 black: float,
                 white: float,
                 gamma: float = 1.0) -> np.ndarray:
    """
    Simple Levels-style mapping:
      - values <= black  -> 0
      - values >= white  -> 255
      - in between linearly stretched, with optional gamma.
    Intended for *display/export only*, not analysis.
    """
    img_f = img.astype(np.float32)
    img_f = (img_f - black) / (white - black)
    img_f = np.clip(img_f, 0.0, 1.0)

    if gamma != 1.0:
        img_f = img_f**(1.0 / gamma)

    return np.round(255.0 * img_f).astype(np.uint8)





# =========================================================
# 5. APPLY BACKGROUND CORRECTION TO ANY FRAME
# =========================================================

def apply_background_correction(img_path: Path):
    """
    Load a frame, crop + mask it in the same way as the reference,
    and apply the chosen background model.

    Returns:
        orig_masked  (uint8): masked + cropped frame
        corrected    (uint8): background-corrected frame
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not load image: {img_path}")

    # crop
    img_crop = img[y0:y1, x0:x1].astype(np.float32)

    # mask: white outside circle
    mask_local = mask_crop
    orig_masked = np.full_like(img_crop, 255.0)
    orig_masked[mask_local == 1] = img_crop[mask_local == 1]

    # background correction inside circle
    corrected = orig_masked.copy()
    corrected[mask_local == 1] = orig_masked[mask_local == 1] - (
        background[mask_local == 1] - bg_med
    )

    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    return orig_masked.astype(np.uint8), corrected



# =========================================================
# 6. TEST AND VISUALISE ENHANCEMENT ON ONE FRACTURE FRAME
# =========================================================

test_img_path = BASE_DIR / TEST_IMG_NAME
print("Testing enhancement on:", test_img_path)

orig_masked_test, corrected_test = apply_background_correction(test_img_path)

# Apply enhancement
enhanced_test = apply_enhancement(corrected_test, mask_crop)

# Optional: apply Levels for paper-ready figure
# Choose black/white from percentiles inside the circle
mask_bool = (mask_crop == 1)
inside_vals = enhanced_test[mask_bool]

p10, p99 = np.percentile(inside_vals, [10, 99])  # tweak 1/99 to taste
black = p10
white = p99
gamma = 1.0   # >1 brightens midtones, <1 darkens midtones

fig_img = apply_levels(enhanced_test, black=black, white=white, gamma=gamma)




fig, axes = plt.subplots(1, 3, figsize=(21, 7))

axes[0].imshow(corrected_test, cmap="gray")
axes[0].set_title(
    f"Fracture frame\nbackground-corrected\n(BG_MODE = {BG_MODE})"
)
axes[0].axis("off")
axes[0].set_aspect("equal")

axes[1].imshow(enhanced_test, cmap="gray")

axes[1].set_title(
    "Fracture frame\nbackground-corrected + enhanced\n"
    f"(ENHANCE_MODE = {ENHANCE_MODE})"
)
axes[1].axis("off")
axes[1].set_aspect("equal")


axes[2].imshow(fig_img, cmap="gray")
axes[2].set_title(
    "Fracture frame\nbackground-corrected + enhanced\n"
    f"(ENHANCE_MODE = {ENHANCE_MODE}, Levels)"
)



plt.tight_layout()
plt.show()




# =========================================================
# 7. BATCH PROCESSING: APPLY TO ALL FRAMES AND SAVE
# =========================================================

print("\nStarting batch processing...")

# Range of frame numbers to process
start_idx = 2567
end_idx   = 2867  # inclusive

for idx in range(start_idx, end_idx + 1):
    fname = f"DSC_{idx:04d}.JPG"
    in_path = BASE_DIR / fname

    if not in_path.exists():
        print(f"  [SKIP] {fname} not found")
        continue

    try:
        # 1) Background correction (crop + mask done inside)
        orig_masked, corrected = apply_background_correction(in_path)

        # 2) Enhancement (CLAHE + unsharp etc., according to ENHANCE_MODE)
        enhanced = apply_enhancement(corrected, mask_crop)

        # --- optional denoising step ---
        denoised = cv2.fastNlMeansDenoising(
        enhanced,
        h=3,              # filter strength (2–6 is good)
        templateWindowSize=7,
        searchWindowSize=21
    )


        # 3) Levels for display/export
        #    Compute black/white from intensities *inside* the circle
        inside_vals = denoised[mask_crop == 1]
        black = float(np.percentile(inside_vals, 10))   # p10
        white = float(np.percentile(inside_vals, 99))   # p99
        gamma = 1.0

        fig_img = apply_levels(denoised, black=black, white=white, gamma=gamma)

        # 4) Resize to 800 x 800 for timelapse / figures
        fig_resized = cv2.resize(
            fig_img,
            (800, 800),
            interpolation=cv2.INTER_AREA
        )

        # 5) Save as JPEG in output directory (reduced file size)
        out_path = OUTPUT_DIR / fname  # same basename
        cv2.imwrite(str(out_path), fig_resized, [cv2.IMWRITE_JPEG_QUALITY, 90])

        print(f"  [OK]  {fname} -> {out_path.name}")

    except Exception as e:
        print(f"  [ERR] {fname}: {e}")

print("Batch processing complete.")
