import cv2
import os

# --- Settings ---
video_path = r"D:\Work\Exp-Data\Capillary-fracturing\data\all-data\controlled-pressure\gap06mm\p3\p3.avi"
output_dir = r"D:\Work\Exp-Data\Capillary-fracturing\data\all-data\controlled-pressure\gap06mm\p3\extracted_frames"
prefix = "frame_"
save_every = 1   # save every N frame

# Create output folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_idx = 0      # counts all frames
saved_idx = 0      # counts saved frames

success, frame = cap.read()

while success:
    if frame_idx % save_every == 0:
        filename = os.path.join(output_dir, f"{prefix}{saved_idx:05d}.png")
        cv2.imwrite(filename, frame)
        saved_idx += 1

    frame_idx += 1
    success, frame = cap.read()

cap.release()

print(f"Done. Processed {frame_idx} frames, saved {saved_idx} frames into '{output_dir}'")
