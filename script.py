import os
import json

import cv2
import numpy as np


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def preprocess_lighting(hsv_image):
    """Apply CLAHE only to the V channel to even out lighting."""
    h, s, v = cv2.split(hsv_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_eq = clahe.apply(v)
    return cv2.merge((h, s, v_eq))


def get_table_mask(image,tol):
    """Return a binary mask of the table felt by sampling the dominant colour."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = preprocess_lighting(hsv)
    h, w = image.shape[:2]

    points = [
        (h // 2, w // 2),  # Center
        (int(h * 0.4), int(w * 0.4)),  # Top-Left inner
        (int(h * 0.4), int(w * 0.6)),  # Top-Right inner
        (int(h * 0.6), int(w * 0.4)),  # Bottom-Left inner
        (int(h * 0.6), int(w * 0.6))  # Bottom-Right inner
    ]
    samples = []
    for py, px in points:
        patch = hsv[py - 20:py + 20, px - 20:px + 20]
        samples.append(patch)

    all_samples = np.vstack(samples)
    median_hsv = np.median(all_samples, axis=(0, 1))

    tol = np.array(tol)
    lower = np.clip(median_hsv - tol, 0, 255).astype(np.uint8)
    upper = np.clip(median_hsv + tol, 0, 255).astype(np.uint8)
    return cv2.inRange(hsv, lower, upper)


def isolate_largest_blob(mask):
    """Keep only the largest connected region in a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)
    largest = max(contours, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)
    return clean


def order_points(pts):
    """Order 4 points clockwise starting from the one with the smallest angle."""
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)].astype("float32")


def get_table_corners(clean_mask):
    """Find and return the 4 corners of the table from its clean mask."""
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
    """Perform a perspective warp to produce a flat top-down view of the table."""
    if corners is None or len(corners) != 4:
        return None

    pts = corners.reshape(4, 2)
    rect = order_points(pts)
    tl, tr, br, bl = rect

    inner_width  = 1000
    inner_height = 500

    width_top    = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    convergence_ratio = width_bottom / width_top if width_top != 0 else 1.0

    if convergence_ratio > 1.2:
        # Broadcast / angled view
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

        matrix       = cv2.getPerspectiveTransform(rect, dst)
        warped_image = cv2.warpPerspective(original_image, matrix, (total_width, total_height))

    else:
        # True top-down view
        total_width  = inner_width  + (pad_x * 2)
        total_height = inner_height + (pad_y * 2)

        dst = np.array([
            [pad_x,                   pad_y],
            [pad_x + inner_width - 1, pad_y],
            [pad_x + inner_width - 1, pad_y + inner_height - 1],
            [pad_x,                   pad_y + inner_height - 1],
        ], dtype="float32")

        matrix       = cv2.getPerspectiveTransform(rect, dst)
        warped_image = cv2.warpPerspective(original_image, matrix, (total_width, total_height))

        # Small rotation correction for slightly tilted top-down shots
        if tilt_correction != 0:
            center     = (total_width // 2, total_height // 2)
            rot_matrix = cv2.getRotationMatrix2D(center, tilt_correction, 1.0)
            warped_image = cv2.warpAffine(warped_image, rot_matrix, (total_width, total_height))

    return warped_image


def get_top_view(image):
    """
    Given a BGR image, return a top-down perspective-corrected view of the
    pool table, or None if any step fails.
    """
    top_view_tol = np.array([15, 80, 80])
    raw_mask   = get_table_mask(image,top_view_tol)
    clean_mask = isolate_largest_blob(raw_mask)

    corners = get_table_corners(clean_mask)
    if corners is None:
        return None

    return warp_to_top_view(image, corners)

# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────

def process_images(input_json: str, output_dir: str):
    """
    Read a JSON file listing image paths, produce a top-view image for each,
    and save them to output_dir/<original_filename>.jpg.
    """
    # --- Load input list ---
    with open(input_json, "r") as f:
        data = json.load(f)
    image_paths = data["image_path"] if isinstance(data, dict) else data

    print(f"Loaded {len(image_paths)} image(s) from '{input_json}'")

    # --- Prepare output directory ---
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

        top_view = get_top_view(image)
        if top_view is None:
            print("FAILED (top view could not be produced)")
            results.append({"image": filename, "error": "top view could not be produced"})
            continue

        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, top_view)
        print(f"→  {out_path}")

        results.append({"image": filename, "top_view": out_path})

    # --- Save results JSON ---
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