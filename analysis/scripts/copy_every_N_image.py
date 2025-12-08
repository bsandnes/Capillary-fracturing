import os
import shutil

# --- Settings ---
source_dir = r"D:\Work\Data copies\DawangsPC\PHD\experiment\2020-02-11 and later constant P_gasDriven\0.6mm gap\0.07_p"
dest_dir = r"D:\Work\Exp-Data\Capillary-fracturing\data\all-data\controlled-pressure\gap06mm\p007\every20image"
copy_every = 20   # copy every 100th image

# Create destination folder if missing
os.makedirs(dest_dir, exist_ok=True)

# Get sorted list of image files (only JPGs)
files = sorted([f for f in os.listdir(source_dir) if f.lower().endswith(".jpg")])

print(f"Found {len(files)} JPG files.")

saved = 0
for idx, filename in enumerate(files):
    if idx % copy_every == 0:   # select every Nth file
        src = os.path.join(source_dir, filename)
        dst = os.path.join(dest_dir, filename)
        shutil.copy2(src, dst)
        saved += 1

print(f"Copied {saved} files into '{dest_dir}'")
