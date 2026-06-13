import os
import json
import shutil
import pandas as pd

# --- CONFIGURATION ---
CSV_PATH = "partition.csv"
IMAGES_DIR = "./images"
COCO_JSON_PATH = "images/_annotations.coco.json"
OUTPUT_DIR = "./yolo_dataset"

# 1. Create target directories
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

# 2. Load Partitions (Map 'valid' to 'val' to match YOLO standard)
df = pd.read_csv(CSV_PATH)
df['partition'] = df['partition'].replace('valid', 'val')
partition_dict = dict(zip(df['image_name'], df['partition']))

# 3. Load COCO JSON
with open(COCO_JSON_PATH, 'r') as f:
    coco_data = json.load(f)

# Map category IDs to YOLO class indexes (0, 1, 2...)
categories = coco_data['categories']
cat_id_to_yolo_idx = {cat['id']: idx for idx, cat in enumerate(categories)}
class_names = [cat['name'] for cat in categories]

# Map image IDs to image filename and dimensions
image_map = {img['id']: img for img in coco_data['images']}

# Initialize a dictionary to gather labels for each image
image_labels = {img['file_name']: [] for img in coco_data['images']}

# 4. Process Annotations to YOLO Format (Normalized cx, cy, w, h)
for ann in coco_data['annotations']:
    img_id = ann['image_id']
    img_info = image_map[img_id]
    file_name = img_info['file_name']
    
    img_w = img_info['width']
    img_h = img_info['height']
    
    # COCO bbox format: [top_left_x, top_left_y, width, height]
    bbox = ann['bbox']
    x_tl, y_tl, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    
    # Convert to YOLO format: [center_x, center_y, width, height] normalized
    cx = (x_tl + w / 2) / img_w
    cy = (y_tl + h / 2) / img_h
    normalized_w = w / img_w
    normalized_h = h / img_h
    
    yolo_class = cat_id_to_yolo_idx[ann['category_id']]
    
    image_labels[file_name].append(f"{yolo_class} {cx:.6f} {cy:.6f} {normalized_w:.6f} {normalized_h:.6f}")

# 5. Move files and write labels based on CSV partitions
for file_name, labels in image_labels.items():
    # Find out if it belongs to train, val, or test
    split = partition_dict.get(file_name)
    if not split:
        print(f"Warning: {file_name} not found in partition.csv. Skipping.")
        continue
        
    src_img_path = os.path.join(IMAGES_DIR, file_name)
    if not os.path.exists(src_img_path):
        continue

    # Paths for copy targets
    dst_img_path = os.path.join(OUTPUT_DIR, split, "images", file_name)
    base_name, _ = os.path.splitext(file_name)
    dst_lbl_path = os.path.join(OUTPUT_DIR, split, "labels", f"{base_name}.txt")
    
    # Copy Image
    shutil.copy(src_img_path, dst_img_path)
    
    # Write YOLO label file
    with open(dst_lbl_path, 'w') as lf:
        lf.write("\n".join(labels))

# 6. Create the data.yaml file required by YOLO
yaml_content = f"""
path: {os.path.abspath(OUTPUT_DIR)} # dataset root dir
train: train/images
val: val/images
test: test/images

names:
"""
for idx, name in enumerate(class_names):
    yaml_content += f"  {idx}: {name}\n"

with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w") as f:
    f.write(yaml_content.strip())

print(f"Success! YOLO dataset generated at: {OUTPUT_DIR}")
print(f"Detected classes: {class_names}")