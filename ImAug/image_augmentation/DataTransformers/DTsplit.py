import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np


def split_data(
        base_dir: Path,
        train_dir: Path,
        validate_dir: Path,
        test_dir: Path,
        validate_size: float = 0.25,
        test_size: float = 0,
        random_state: int = 7):

    images = sorted(os.listdir(base_dir / "images"))
    labels = sorted(os.listdir(base_dir / "labels"))
    data = list(zip(images, labels))

    os.makedirs(f"{train_dir}/images", exist_ok=True)
    os.makedirs(f"{train_dir}/labels", exist_ok=True)

    test_data = None
    train_data = None
    validate_data = None

    if (test_dir == None) or (test_size == 0):
        train_data, validate_data = train_test_split(
            data, 
            test_size = validate_size,
            random_state = random_state
        )
        os.makedirs(f"{validate_dir}/images", exist_ok=True)
        os.makedirs(f"{validate_dir}/labels", exist_ok=True)

    else:
        middle_data, test_data = train_test_split(
            data, 
            test_size = test_size,
            random_state = random_state
        )
        tuned_testsize = validate_size / (1 - test_size)
        if tuned_testsize != 0:
            train_data, validate_data = train_test_split(
                middle_data,
                test_size = int(tuned_testsize) if tuned_testsize == int(tuned_testsize) else tuned_testsize,
                random_state = random_state
            )
        else:
            train_data = middle_data
            
        os.makedirs(f"{test_dir}/images", exist_ok=True)
        os.makedirs(f"{test_dir}/labels", exist_ok=True)
        os.makedirs(f"{validate_dir}/images", exist_ok=True)
        os.makedirs(f"{validate_dir}/labels", exist_ok=True)

    if test_data:
        for image, label in test_data:
            shutil.copy(os.path.join(base_dir, "images", image), f"{test_dir}/images")
            shutil.copy(os.path.join(base_dir, "labels", label), f"{test_dir}/labels")

    if train_data:
        for image, label in train_data:
            shutil.copy(os.path.join(base_dir, "images", image), f"{train_dir}/images")
            shutil.copy(os.path.join(base_dir, "labels", label), f"{train_dir}/labels")

    if validate_data:
        for image, label in validate_data:
            shutil.copy(os.path.join(base_dir, "images", image), f"{validate_dir}/images")
            shutil.copy(os.path.join(base_dir, "labels", label), f"{validate_dir}/labels")
