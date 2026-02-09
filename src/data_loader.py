import os
import cv2
import numpy as np

LABEL_MAP = {"I": 0, "M": 1, "S": 2}

def load_rock_images(data_path, target_size=(64, 64)):
    """
    Loads images and labels inferred from filename prefix:
    I*, M*, S*  -> 3 classes
    Returns:
      X: (N, H, W) float32 in [0,1]
      y: (N,) int labels_toggle
      filenames: list[str]
    """
    images, labels, filenames = [], [], []

    for file_name in sorted(os.listdir(data_path)):
        if not file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            continue

        prefix = file_name[0].upper()
        if prefix not in LABEL_MAP:
            continue

        img_path = os.path.join(data_path, file_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        resized = cv2.resize(img, target_size)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) / 255.0

        images.append(gray.astype(np.float32))
        labels.append(LABEL_MAP[prefix])
        filenames.append(file_name)

    return np.array(images), np.array(labels), filenames

def flatten_images(X):
    """(N,H,W) -> (N, H*W)"""
    return X.reshape(X.shape[0], -1)
