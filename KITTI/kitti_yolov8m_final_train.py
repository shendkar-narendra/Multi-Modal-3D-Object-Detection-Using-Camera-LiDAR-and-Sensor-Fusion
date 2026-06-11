# =========================================================
# KITTI + YOLOv8m FULL RESEARCH PIPELINE
# =========================================================
#
# PURPOSE:
# - Train YOLOv8m on KITTI
# - Validate on KITTI validation split
# - Compute:
#       Precision
#       Recall
#       mAP50
#       mAP50-95
# - Scenario analysis:
#       single car
#       two cars
#       multiple cars
#       near / medium / far
#       dark / medium / bright lighting
# - Save visual detection results
# - Save CSV analysis
#
# =========================================================
# IMPORTANT
# =========================================================
#
# Optimized for:
#
# GPU:
#   GTX 1050 Ti 4GB
#
# Uses:
#   YOLOv8m
#   imgsz=960
#   batch=2
#
# =========================================================

import os

# Helps reduce CUDA fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import shutil
import cv2
import yaml
import numpy as np
import pandas as pd

from PIL import Image
from ultralytics import YOLO

# =========================================================
# PATHS
# =========================================================

BASE_PATH = r"E:\kitti"

TRAIN_IMAGE_DIR = os.path.join(
    BASE_PATH,
    "Training",
    "image_2"
)

TRAIN_LABEL_DIR = os.path.join(
    BASE_PATH,
    "Training",
    "label_2"
)

IMAGESETS_DIR = os.path.join(
    BASE_PATH,
    "ImageSets"
)

TRAIN_SPLIT = os.path.join(
    IMAGESETS_DIR,
    "train.txt"
)

VAL_SPLIT = os.path.join(
    IMAGESETS_DIR,
    "val.txt"
)

# =========================================================
# OUTPUTS
# =========================================================

YOLO_DATASET = os.path.join(
    BASE_PATH,
    "yolo_dataset"
)

RESULTS_DIR = os.path.join(
    BASE_PATH,
    "analysis_results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================================================
# CLASSES
# =========================================================

CLASSES = {
    "Car": 0,
    "Pedestrian": 1,
    "Cyclist": 2
}

# =========================================================
# CREATE YOLO DATASET STRUCTURE
# =========================================================

folders = [
    "images/train",
    "images/val",

    "labels/train",
    "labels/val"
]

for folder in folders:

    os.makedirs(
        os.path.join(YOLO_DATASET, folder),
        exist_ok=True
    )

# =========================================================
# READ SPLITS
# =========================================================

def read_split(txt_path):

    with open(txt_path, "r") as f:

        ids = [
            line.strip()
            for line in f.readlines()
        ]

    return ids

train_ids = read_split(TRAIN_SPLIT)
val_ids = read_split(VAL_SPLIT)

print(f"Train samples      : {len(train_ids)}")
print(f"Validation samples : {len(val_ids)}")

# =========================================================
# CONVERT KITTI -> YOLO FORMAT
# =========================================================

def convert_sample(sample_id, split):

    image_name = sample_id + ".png"
    label_name = sample_id + ".txt"

    image_path = os.path.join(
        TRAIN_IMAGE_DIR,
        image_name
    )

    label_path = os.path.join(
        TRAIN_LABEL_DIR,
        label_name
    )

    # =====================================================
    # COPY IMAGE
    # =====================================================

    output_image = os.path.join(
        YOLO_DATASET,
        f"images/{split}",
        image_name
    )

    shutil.copy(image_path, output_image)

    # =====================================================
    # IMAGE SIZE
    # =====================================================

    image = Image.open(image_path)

    w, h = image.size

    yolo_lines = []

    # =====================================================
    # READ KITTI LABELS
    # =====================================================

    with open(label_path, "r") as f:

        lines = f.readlines()

    for line in lines:

        data = line.strip().split()

        class_name = data[0]

        # Ignore unwanted classes
        if class_name not in CLASSES:
            continue

        class_id = CLASSES[class_name]

        # KITTI bbox
        x1 = float(data[4])
        y1 = float(data[5])
        x2 = float(data[6])
        y2 = float(data[7])

        # =================================================
        # CONVERT TO YOLO FORMAT
        # =================================================

        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h

        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        yolo_lines.append(
            f"{class_id} "
            f"{x_center} "
            f"{y_center} "
            f"{bw} "
            f"{bh}"
        )

    # =====================================================
    # SAVE YOLO LABEL
    # =====================================================

    output_label = os.path.join(
        YOLO_DATASET,
        f"labels/{split}",
        label_name
    )

    with open(output_label, "w") as f:

        f.write("\n".join(yolo_lines))

# =========================================================
# PREPARE TRAIN DATA
# =========================================================

print("\nPreparing training set...")

for sid in train_ids:

    convert_sample(sid, "train")

print("Training set completed.")

# =========================================================
# PREPARE VALIDATION DATA
# =========================================================

print("\nPreparing validation set...")

for sid in val_ids:

    convert_sample(sid, "val")

print("Validation set completed.")

# =========================================================
# CREATE YAML
# =========================================================

yaml_data = {

    "path": YOLO_DATASET,

    "train": "images/train",

    "val": "images/val",

    "names": {

        0: "Car",
        1: "Pedestrian",
        2: "Cyclist"
    }
}

yaml_path = os.path.join(
    YOLO_DATASET,
    "kitti.yaml"
)

with open(yaml_path, "w") as f:

    yaml.dump(yaml_data, f)

print("\nYAML file created.")

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    # =====================================================
    # LOAD MODEL
    # =====================================================

    model = YOLO("yolov8m.pt")

    # =====================================================
    # TRAINING
    # =====================================================

    print("\nStarting training...")

    model.train(

        data=yaml_path,

        epochs=100,

        imgsz=960,

        batch=2,

        device=0,

        workers=0,

        cache=False,

        optimizer="AdamW",

        lr0=0.001,

        cos_lr=True,

        patience=30,

        amp=True,

        augment=True,

        mixup=0.1,

        copy_paste=0.1,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=5,

        translate=0.1,

        scale=0.5,

        fliplr=0.5,

        mosaic=1.0,

        close_mosaic=10,

        project=RESULTS_DIR,

        name="yolov8m_kitti"
    )

    # =====================================================
    # VALIDATION METRICS
    # =====================================================

    print("\nRunning validation...")

    metrics = model.val()

    print("\n================ METRICS ================")

    print(f"mAP50      : {metrics.box.map50:.4f}")
    print(f"mAP50-95   : {metrics.box.map:.4f}")
    print(f"Precision  : {metrics.box.mp:.4f}")
    print(f"Recall     : {metrics.box.mr:.4f}")

    # =====================================================
    # LOAD BEST MODEL
    # =====================================================

    best_model_path = os.path.join(
        RESULTS_DIR,
        "yolov8m_kitti",
        "weights",
        "best.pt"
    )

    best_model = YOLO(best_model_path)

    # =====================================================
    # SCENARIO ANALYSIS
    # =====================================================

    print("\nRunning scenario analysis...")

    analysis_data = []

    for sid in val_ids:

        image_name = sid + ".png"

        image_path = os.path.join(
            TRAIN_IMAGE_DIR,
            image_name
        )

        image = cv2.imread(image_path)

        h, w, _ = image.shape

        # =================================================
        # LIGHTING ESTIMATION
        # =================================================

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        brightness = np.mean(gray)

        if brightness < 70:

            lighting = "dark"

        elif brightness < 140:

            lighting = "medium"

        else:

            lighting = "bright"

        # =================================================
        # PREDICTION
        # =================================================

        results = best_model.predict(

            source=image_path,

            conf=0.25,

            verbose=False,

            save=True,

            project=RESULTS_DIR,

            name="scenario_visuals",

            exist_ok=True
        )

        result = results[0]

        boxes = result.boxes

        num_cars = 0

        max_box_area = 0

        if boxes is not None:

            for box in boxes:

                cls_id = int(box.cls[0])

                # Car only
                if cls_id != 0:
                    continue

                num_cars += 1

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                area = (x2 - x1) * (y2 - y1)

                max_box_area = max(
                    max_box_area,
                    area
                )

        # =================================================
        # DISTANCE ESTIMATION
        # =================================================

        image_area = w * h

        area_ratio = max_box_area / image_area

        if area_ratio > 0.15:

            distance = "near"

        elif area_ratio > 0.04:

            distance = "medium"

        else:

            distance = "far"

        # =================================================
        # SCENARIO TYPE
        # =================================================

        if num_cars == 1:

            scenario = "single_car"

        elif num_cars == 2:

            scenario = "two_cars"

        elif num_cars >= 3:

            scenario = "multiple_cars"

        else:

            scenario = "no_car"

        # =================================================
        # SAVE ANALYSIS
        # =================================================

        analysis_data.append({

            "image_id": sid,

            "cars_detected": num_cars,

            "scenario": scenario,

            "distance": distance,

            "lighting": lighting
        })

    # =====================================================
    # SAVE CSV
    # =====================================================

    df = pd.DataFrame(analysis_data)

    csv_path = os.path.join(
        RESULTS_DIR,
        "scenario_analysis.csv"
    )

    df.to_csv(csv_path, index=False)

    print("\n================================================")
    print("TRAINING + ANALYSIS COMPLETED")
    print("================================================")

    print("\nResults folder:")
    print(RESULTS_DIR)

    print("\nCSV saved:")
    print(csv_path)