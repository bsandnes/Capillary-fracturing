import cv2
import numpy as np
from pathlib import Path
import csv
from datetime import datetime

BASE_DIR = Path(
    r"D:\Work\Exp-Data\Capillary-fracturing\data\all-data\controlled-flow-rate-0_01ml_min\phi_58\all_images"
)

img_path = BASE_DIR / "DSC_2567.JPG"
print("Loading:", img_path)

img = cv2.imread(str(img_path))
if img is None:
    print("Could not load image:", img_path)
    raise SystemExit

# grayscale -> back to BGR just so we can draw coloured overlays later if we want
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
display = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

h, w = display.shape[:2]
print(f"Image size: {w} x {h} px")

# --- scale to fit screen but KEEP aspect ratio -------------------------
screen_w = 1600
screen_h = 900
scale = min(screen_w / w, screen_h / h)
disp_resized = cv2.resize(display, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
# compute inverse scale from actual resized width to avoid rounding mismatch
inv_scale = w / disp_resized.shape[1]

print("Left-click: print TRUE pixel coords + BGR + gray.")
print("Right-click: distance from last left-click (in pixels).")
print("Keys: 'q' quit, 'r' reset last point, 'c' clear all points, 's' save points to CSV.")

last_point = None
points = []  # saved left-click points (true coords)

WINDOW = "image"

def mouse_callback(event, x, y, flags, param):
    global last_point, points

    # convert from display coords -> original image coords
    true_x = int(round(x * inv_scale))
    true_y = int(round(y * inv_scale))

    # clamp to image bounds
    true_x = max(0, min(true_x, w - 1))
    true_y = max(0, min(true_y, h - 1))

    if event == cv2.EVENT_LBUTTONDOWN:
        last_point = (true_x, true_y)
        points.append(last_point)
        b, g, r = map(int, img[true_y, true_x])  # note: img[y,x] indexing
        gray = int(img_gray[true_y, true_x])
        print(f"Left click at (x={true_x}, y={true_y})  BGR=({b},{g},{r})  gray={gray}")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if last_point is None:
            print("Right-click: no starting point yet.")
            return
        dx = true_x - last_point[0]
        dy = true_y - last_point[1]
        dist = (dx*dx + dy*dy)**0.5
        print(
            f"Right click at (x={true_x}, y={true_y}) → distance from last left-click = {dist:.2f} px"
        )

cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW, mouse_callback)

while True:
    overlay = disp_resized.copy()  # draw markers on a fresh copy each frame

    # draw all saved points (scaled to display)
    for p in points:
        px = int(round(p[0] / inv_scale))
        py = int(round(p[1] / inv_scale))
        cv2.circle(overlay, (px, py), 6, (0, 200, 0), -1)  # filled green dot

    # draw last_point highlight (if exists)
    if last_point is not None:
        lp_x = int(round(last_point[0] / inv_scale))
        lp_y = int(round(last_point[1] / inv_scale))
        cv2.circle(overlay, (lp_x, lp_y), 10, (0, 0, 255), 2)  # red ring

    # if we have at least two points, draw a line from last_point to the last left-clicked point
    if last_point is not None and len(points) >= 1:
        # draw line between last_point and current mouse position only when right-click occurs,
        # but here we draw lines between consecutive saved points for quick visual feedback
        for i in range(1, len(points)):
            p0 = (int(round(points[i-1][0] / inv_scale)), int(round(points[i-1][1] / inv_scale)))
            p1 = (int(round(points[i][0] / inv_scale)), int(round(points[i][1] / inv_scale)))
            cv2.line(overlay, p0, p1, (255, 0, 0), 1)

    cv2.imshow(WINDOW, overlay)
    key = cv2.waitKey(20) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        last_point = None
        print("Last point reset.")
    elif key == ord("c"):
        last_point = None
        points = []
        print("Cleared all saved points.")
    elif key == ord("s"):
        if not points:
            print("No points to save.")
        else:
            fn = f"clicked_points_{datetime.now():%Y%m%d_%H%M%S}.csv"
            with open(fn, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y"])
                writer.writerows(points)
            print(f"Saved {len(points)} points to {fn}")

cv2.destroyAllWindows()
