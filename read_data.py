import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import shutil
import yaml

def xyxy2xywh(x_min, y_min, x_max, y_max, width, height):
    w = float((x_max - x_min) / width)
    h = float((y_max - y_min) / height)
    x = float(x_min / width) + w/2
    y = float(y_min / height) + h/2
    return x, y, w, h

def create_yaml(train_classes, data_root):
    data = dict(
        names = train_classes,
        nc = len(train_classes),
        train = f"{data_root}/train.txt",
        val = f"{data_root}/val.txt",
    )

    with open(f"{data_root}/data.yaml", "w") as file:
        yaml.dump(data, file, default_flow_style=False)

if __name__ == "__main__":
    df = pd.read_csv("./dataset/stratified5folds.csv")

    fold = 0
    data_root = f"./fold{fold}"
    classes = df["class_name"].unique()
    train_classes = np.delete(classes, np.where(classes=="No finding")).tolist()
    class2id = {k:v for k, v in zip(train_classes, range(13))}

    os.makedirs(f"{data_root}/images/train", exist_ok=True)
    os.makedirs(f"{data_root}/images/val", exist_ok=True)
    os.makedirs(f"{data_root}/labels/train", exist_ok=True)
    os.makedirs(f"{data_root}/labels/val", exist_ok=True)

    train_files = set()
    val_files = set()
    for i, row in tqdm(df.iterrows()):
        if row.class_name != "No finding":
            name = row.image_id.split(".")[0]
            x, y, w, h = xyxy2xywh(row.x_min, row.y_min, row.x_max, row.y_max, row.width, row.height)
            if row.fold != 0:
                f = open(f"{data_root}/labels/train/{name}.txt", "a")
                f.write(f"{class2id[row.class_name]} {x} {y} {w} {h}\n")
                f.close()
                train_files.add(f"./images/train/{row.image_id}")
                shutil.copy(f"./dataset/train/{row.image_id}", f"{data_root}/images/train")
            else:
                f = open(f"{data_root}/labels/val/{name}.txt", "a")
                f.write(f"{class2id[row.class_name]} {x} {y} {w} {h}\n")
                f.close()
                val_files.add(f"./images/val/{row.image_id}")
                shutil.copy(f"./dataset/train/{row.image_id}", f"{data_root}/images/val")
        
    with open(f"{data_root}/train.txt", "w") as f:
        for p in list(train_files):
            f.write(p + "\n")
    with open(f"{data_root}/val.txt", "w") as f:
        for p in list(val_files):
            f.write(p + "\n")

    create_yaml(train_classes, data_root)    
