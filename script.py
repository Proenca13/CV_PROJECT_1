import os
import json

import cv2
import numpy as np


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────

TOP_VIEW_TOL = np.array([15,  80,  80])   # tighter — used for perspective warp
BALL_TOL     = np.array([15, 120, 120])   # looser  — used for ball detection


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def preprocess_lighting(hsv_image):
    h, s, v = cv2.split(hsv_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_eq = clahe.apply(v)
    return cv2.merge((h, s, v_eq))


def get_table_mask(image, tol):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = preprocess_lighting(hsv)
    h, w = image.shape[:2]
    points = [
        (h // 2,w // 2),
        (int(h * 0.4), int(w * 0.4)),
        (int(h * 0.4), int(w * 0.6)),
        (int(h * 0.6), int(w * 0.4)),
        (int(h * 0.6), int(w * 0.6)),
    ]
    samples = [hsv[py - 20:py + 20, px - 20:px + 20] for py, px in points]
    median_hsv = np.median(np.vstack(samples), axis=(0, 1))
    tol = np.array(tol)
    lower = np.clip(median_hsv - tol, 0, 255).astype(np.uint8)
    upper = np.clip(median_hsv + tol, 0, 255).astype(np.uint8)
    return cv2.inRange(hsv, lower, upper)


def isolate_largest_blob(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)
    largest = max(contours, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)
    return clean


def order_points(pts):
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)].astype("float32")


def get_table_corners(clean_mask):
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)
    epsilon = 0.02 * cv2.arcLength(hull, True)
    corners = cv2.approxPolyDP(hull, epsilon, True)
    if len(corners) == 4:
        return corners
    print(f"  Warning: expected 4 corners, got {len(corners)}.")
    return None


def warp_to_top_view(original_image, corners, pad_x=70, pad_y=50, tilt_correction=-2.5):
    if corners is None or len(corners) != 4:
        return None
    pts  = corners.reshape(4, 2)
    rect = order_points(pts)
    tl, tr, br, bl = rect

    inner_width  = 1000
    inner_height = 500

    width_top    = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    convergence_ratio = width_bottom / width_top if width_top != 0 else 1.0

    if convergence_ratio > 1.2:
        mid_bottom_x = (bl[0] + br[0]) / 2.0
        mid_top_x    = (tl[0] + tr[0]) / 2.0
        skew         = abs(mid_bottom_x - mid_top_x)
        is_angled    = skew > (width_bottom * 0.10)

        extra_right_pad = 40 if is_angled else 0
        extra_top_pad   = 40 if is_angled else 0

        total_width  = inner_width  + (pad_x * 2) + extra_right_pad
        total_height = inner_height + (pad_y * 2) + extra_top_pad

        dst = np.array([
            [pad_x + inner_width - 1, pad_y + inner_height - 1 + extra_top_pad],
            [pad_x + inner_width - 1, pad_y + extra_top_pad],
            [pad_x,                   pad_y + extra_top_pad],
            [pad_x,                   pad_y + inner_height - 1 + extra_top_pad],
        ], dtype="float32")
    else:
        total_width  = inner_width  + (pad_x * 2)
        total_height = inner_height + (pad_y * 2)

        dst = np.array([
            [pad_x,                   pad_y],
            [pad_x + inner_width - 1, pad_y],
            [pad_x + inner_width - 1, pad_y + inner_height - 1],
            [pad_x,                   pad_y + inner_height - 1],
        ], dtype="float32")

        if tilt_correction != 0:
            matrix       = cv2.getPerspectiveTransform(rect, dst)
            warped_image = cv2.warpPerspective(original_image, matrix, (total_width, total_height))
            center       = (total_width // 2, total_height // 2)
            rot_matrix   = cv2.getRotationMatrix2D(center, tilt_correction, 1.0)
            return cv2.warpAffine(warped_image, rot_matrix, (total_width, total_height))

    matrix       = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(original_image, matrix, (total_width, total_height))


def get_top_view(image):
    raw_mask   = get_table_mask(image, TOP_VIEW_TOL)
    clean_mask = isolate_largest_blob(raw_mask)
    corners    = get_table_corners(clean_mask)
    if corners is None:
        return None
    return warp_to_top_view(image, corners)


# ─────────────────────────────────────────────
#  BALL DETECTION
# ─────────────────────────────────────────────

def _sample_table_hsv(hsv, img_shape):
    """Sample the median HSV of the table felt from 5 inner patches."""
    h_img, w_img = img_shape[:2]
    pts = [
        (h_img // 2,             w_img // 2),
        (int(h_img * 0.4), int(w_img * 0.4)),
        (int(h_img * 0.4), int(w_img * 0.6)),
        (int(h_img * 0.6), int(w_img * 0.4)),
        (int(h_img * 0.6), int(w_img * 0.6)),
    ]
    patches = [hsv[py - 20:py + 20, px - 20:px + 20] for py, px in pts]
    return np.median(np.vstack(patches), axis=(0, 1))


def detect_missed_balls(img, playing_area_mask, existing_circles=None):
    """
    Fallback Hough-Circles detector.
    Finds any ball-shaped region that detect_balls missed,
    regardless of colour — rejects only plain table felt.
    """
    if existing_circles is None:
        existing_circles = []

    hsv       = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv       = preprocess_lighting(hsv)
    table_hsv = _sample_table_hsv(hsv, img.shape)
    table_h, table_s, table_v = table_hsv

    median_r = int(np.median([r for _, _, r in existing_circles])) if existing_circles else 18
    r_min    = max(8,  int(median_r * 0.70))
    r_max    = min(45, int(median_r * 1.35))

    play_img  = cv2.bitwise_and(img, img, mask=playing_area_mask)
    bilateral = cv2.bilateralFilter(play_img, d=9, sigmaColor=75, sigmaSpace=75)
    gray      = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)

    raw = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=int(median_r * 1.5),
        param1=50,
        param2=18,
        minRadius=r_min,
        maxRadius=r_max,
    )
    if raw is None:
        return [], []

    new_circles, new_bboxes = [], []

    for (cx, cy, r) in np.round(raw[0]).astype(int):
        if playing_area_mask[cy, cx] == 0:
            continue

        if any(np.hypot(cx - ex, cy - ey) < (r + er) * 0.6
               for (ex, ey, er) in existing_circles):
            continue

        inner_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(inner_mask, (cx, cy), max(int(r * 0.75), 3), 255, -1)
        ball_pixels = hsv[inner_mask == 255]
        if len(ball_pixels) < 10:
            continue

        h, s, v = ball_pixels[:, 0], ball_pixels[:, 1], ball_pixels[:, 2]
        is_table = (
            (np.abs(h.astype(int) - int(table_h)) < 10) &
            (np.abs(s.astype(int) - int(table_s)) < 40) &
            (np.abs(v.astype(int) - int(table_v)) < 40)
        )
        if np.sum(is_table) / len(ball_pixels) > 0.60:
            continue

        final_r = int(r * 1.1)
        new_circles.append((cx, cy, final_r))
        new_bboxes.append((cx - final_r, cy - final_r, 2 * final_r, 2 * final_r))

    return new_circles, new_bboxes


def detect_balls(img):
    """
    Main ball detector. Uses BALL_TOL (looser tolerance) so the table
    mask correctly exposes the balls without the tighter top-view tolerance.
    Returns (circles, bboxes).
    """
    table_mask        = get_table_mask(img, BALL_TOL)          # ← BALL_TOL, not TOP_VIEW_TOL
    playing_area_mask = isolate_largest_blob(table_mask)

    not_table_mask = cv2.bitwise_not(table_mask)
    balls_mask     = cv2.bitwise_and(not_table_mask, playing_area_mask)

    heal_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    balls_mask  = cv2.morphologyEx(balls_mask, cv2.MORPH_CLOSE, heal_kernel, iterations=1)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    balls_mask   = cv2.morphologyEx(balls_mask, cv2.MORPH_OPEN, morph_kernel, iterations=1)

    dist_transform = cv2.distanceTransform(balls_mask, cv2.DIST_L2, 5)
    if dist_transform.max() == 0:
        return [], []

    blobs, _ = cv2.findContours(balls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circles, bboxes = [], []

    for blob in blobs:
        blob_mask = np.zeros_like(balls_mask)
        cv2.drawContours(blob_mask, [blob], -1, 255, thickness=cv2.FILLED)
        blob_mask = cv2.morphologyEx(blob_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        blob_dist = np.zeros_like(dist_transform)
        blob_dist[blob_mask == 255] = dist_transform[blob_mask == 255]

        max_val = blob_dist.max()
        if max_val < 5:
            continue

        sharpened_dist = cv2.pow(blob_dist / max_val, 2) * max_val

        area       = cv2.contourArea(blob)
        perimeter  = cv2.arcLength(blob, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        thresh_ratio = 0.7 if circularity > 0.8 else 0.4
        _, peaks = cv2.threshold(sharpened_dist, thresh_ratio * max_val, 255, 0)
        peaks    = np.uint8(peaks)

        peak_contours, _ = cv2.findContours(peaks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in peak_contours:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx     = int(M["m10"] / M["m00"])
                cy     = int(M["m01"] / M["m00"])
                radius = int(dist_transform[cy, cx])
                if 8 < radius < 45:
                    final_r = int(radius * 1.1)
                    circles.append((cx, cy, final_r))
                    bboxes.append((cx - final_r, cy - final_r, 2 * final_r, 2 * final_r))

    # Fallback: rescue any missed balls regardless of colour
    missed_circles, missed_bboxes = detect_missed_balls(
        img, playing_area_mask, existing_circles=circles
    )
    circles.extend(missed_circles)
    bboxes.extend(missed_bboxes)

    # Dedup / overlap cleanup
    final_circles, final_bboxes = [], []
    ABSOLUTE_MIN_RADIUS = 8
    combined = sorted(zip(circles, bboxes), key=lambda item: item[0][2], reverse=True)

    for (cx, cy, r), bbox in combined:
        if r < ABSOLUTE_MIN_RADIUS:
            continue
        if not any(np.hypot(cx - fx, cy - fy) < (r + fr) * 0.7
                   for (fx, fy, fr) in final_circles):
            final_circles.append((cx, cy, r))
            final_bboxes.append(bbox)

    return final_circles, final_bboxes


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────

def process_images(input_json: str, output_dir: str):
    with open(input_json, "r") as f:
        data = json.load(f)
    image_paths = data["image_path"] if isinstance(data, dict) else data

    print(f"Loaded {len(image_paths)} image(s) from '{input_json}'")
    os.makedirs(output_dir, exist_ok=True)

    results = []

    for path in image_paths:
        filename = os.path.basename(path)
        print(f"Processing {filename}", end=" ")

        image = cv2.imread(path)
        if image is None:
            print("FAILED (could not read image)")
            results.append({"image": filename, "error": "could not read image"})
            continue

        # 1. Top-view warp
        top_view = get_top_view(image)
        if top_view is None:
            print("FAILED (top view could not be produced)")
            results.append({"image": filename, "error": "top view could not be produced"})
            continue

        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, top_view)

        # 2. Ball count on the ORIGINAL image
        circles, bboxes = detect_balls(image)
        num_balls  = len(circles)

        print(f"→  {out_path}  |  balls: {num_balls}")

        results.append({
            "image":     path,
            "top_view":  out_path,
            "num_balls": num_balls,
        })

    results_path = "output.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Top-view images saved to '{output_dir}/'")
    print(f"Results summary saved to '{results_path}'")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    process_images("input.json", "output")