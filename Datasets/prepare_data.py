import os
import yaml
from collections import Counter
from tqdm import tqdm

def check_dataset_structure(base_path):
    with open(os.path.join(base_path, "data.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    sets = ["train", "valid", "test"]
    for s in sets:
        img_dir = os.path.join(base_path, s, "images")
        lbl_dir = os.path.join(base_path, s, "labels")
        imgs = os.listdir(img_dir)
        lbls = os.listdir(lbl_dir)
        print(f"{s.upper()} -> {len(imgs)} images, {len(lbls)} labels")
        missing = set([i.replace(".jpg",".txt") for i in imgs]) - set(lbls)
        if missing:
            print(f"Missing labels in {s}: {len(missing)} files")
        else:
            print("All labels matched.")
