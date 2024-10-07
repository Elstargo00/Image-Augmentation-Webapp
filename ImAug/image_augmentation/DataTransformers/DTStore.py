from torch.utils.data import Dataset
from .utils import text_to_hex, extract_transforming_name
from pathlib import Path
import os
from PIL import Image
import numpy as np
import albumentations as A

class TransPack(Dataset):

    def __init__(self, dataset_dir="./datasets/", transform=None, with_label_augmented=False):
        self.data = []
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        self.images_dir = self.dataset_dir / "images"
        self.labels_dir = self.dataset_dir / "labels"

        images = os.listdir(self.images_dir)
        labels = os.listdir(self.labels_dir)

        images.sort()
        labels.sort()
        
        if (not with_label_augmented) and labels is None or len(images) != len(labels):
            labels = labels + [None] * np.abs(len(images) - len(labels))

        self.data += list(zip(images, labels))
        
        print("image and label pairs")
        for img, label in self.data:
            print(img, label)


    
    def validate_and_correct_bbox(self, bbox):
        x_c, y_c, width, height, label = map(float, bbox)

        epsilon = 1e-6  # Small positive value to avoid exactly zero

        # Calculate half width and height for further adjustments
        w_half = width / 2
        h_half = height / 2

        # Calculate x_min and x_max, y_min and y_max
        x_min = x_c - w_half
        y_min = y_c - h_half
        x_max = x_c + w_half
        y_max = y_c + h_half

        # Clip the coordinates so they are within bounds, instead of shrinking
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(1, x_max)
        y_max = min(1, y_max)

        # Calculate the new width and height based on the clipped coordinates
        new_width = x_max - x_min
        new_height = y_max - y_min

        # Ensure that the width and height are above a small threshold (epsilon)
        new_width = max(epsilon, new_width)
        new_height = max(epsilon, new_height)

        # Recalculate the new center
        new_x_c = x_min + new_width / 2
        new_y_c = y_min + new_height / 2

        # Return the corrected values
        return [new_x_c, new_y_c, new_width, new_height, int(label)]




    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        image_filename, label_filename = self.data[index]
        print("image and label filename:", image_filename, label_filename)
        img = Image.open(self.images_dir / image_filename)
        img = np.array(img)
        
        bboxes = []
        class_labels = []

        if label_filename:
            print(label_filename)
            label_path = self.labels_dir / label_filename

            with open(label_path) as file:
                labels = file.readlines()
                labels = [label.strip().split(" ") for label in labels]

                for label in labels:
                    # changable format
                    bbox = [float(param) for param in label[1:]] + [str(label[0])]
                    validated_bbox = self.validate_and_correct_bbox(bbox)
                    bboxes.append(validated_bbox) 
                    class_labels.append(str(label[0]))

        if self.transform is not None:
            # Log bboxes to confirm their values are within the expected range
            augmentations = self.transform(image=img, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        return image_filename, image, label_filename, bboxes
    



class TransFormat:

    def __init__(self, dirname="augmented"):
        outs = Path(dirname)
        image_outs = outs / "images"
        label_outs = outs / "labels"
        image_outs.mkdir(parents=True, exist_ok=True)
        label_outs.mkdir(parents=True, exist_ok=True)        
        self.transforming_format = []


    def append_format(self, info):
        """
            {
                format_type: 
                    - RandomCrop
                    - HorizontalFlip
                    - VerticalFlip
                
                params:
                * MUST corresponding with format_type
                eg. 
                - RandomCrop -> width=100, height=100,
                - HorizontalFlip -> p=0.8
                - VerticalFlip -> p=0.8
            }
        """
        trans_type = info["format_type"]
        trans_format = trans_type(**info["params"])
        self.transforming_format.append(trans_format)
        

    def compose(self, format="yolo", min_vis=0.7):
        composed = A.Compose(
            self.transforming_format,
            bbox_params = A.BboxParams(
                format = format,
                min_visibility = min_vis
            )
        )
        return composed
    
    
    


# _____________________________________________________________________________
# Build a function
def apply_transform(dataset_dir, transforming_option, transforming_list, output_dir):

    with_label_augmented, augmented_scheme = transforming_option
    if augmented_scheme == "oneTrans":
        for each in transforming_list:
            transformat = TransFormat(output_dir)
            filename_extension = text_to_hex(
                extract_transforming_name(each["format_type"])
            )
            transformat.append_format(each)
            transform = transformat.compose(format="yolo", min_vis=0.7)

            dataset = TransPack(
                dataset_dir = dataset_dir,
                transform = transform,
                with_label_augmented= with_label_augmented,
            )

            tag_ver = output_dir.parts[-1].split('_')[-1]

            for image_filename, image, label_filename, bboxes in dataset:

                if label_filename:
                    label_filename = f"{label_filename[: -4]}_{tag_ver}_{filename_extension}.txt"
                    saved_label = output_dir / "labels" / label_filename

                    with open(saved_label, 'a') as label_file:
                        for bbox in bboxes:
                            label_file.write(f"{str(bbox[-1])} { round(float(bbox[0]), 6) } { round(float(bbox[1]), 6) } { round(float(bbox[2]), 6) } { round(float(bbox[3]), 6) }\n")

                image_filename = f"{image_filename[: -4]}_{tag_ver}_{filename_extension}.png"
                saved_image = output_dir / "images" / image_filename
                image = Image.fromarray(image)
                image.save(saved_image)


    elif augmented_scheme == "allTrans":
        transformat = TransFormat(output_dir)
        filename_extension = []
        for each in transforming_list:
            filename_extension.append(extract_transforming_name(each["format_type"]))
            transformat.append_format(each)
        
        filename_extension = text_to_hex(", ".join(filename_extension))
        transform = transformat.compose(format="yolo", min_vis=0.7)
        
        dataset = TransPack(
            dataset_dir = dataset_dir,
            transform = transform,
            with_label_augmented = with_label_augmented
        )

        tag_ver = output_dir.parts[-1].split('_')[-1]

        for image_filename, image, label_filename, bboxes in dataset:
            if label_filename:
                label_filename = f"{label_filename[: -4]}_{tag_ver}_{filename_extension}.txt"
                saved_label = output_dir / "labels" / label_filename


                with open(saved_label, 'a') as label_file:
                    for bbox in bboxes:
                        label_file.write(f"{str(bbox[-1])} { round(float(bbox[0]), 6) } { round(float(bbox[1]), 6) } { round(float(bbox[2]), 6) } { round(float(bbox[3]), 6) }\n")

            image_filename = f"{image_filename[: -4]}_{tag_ver}_{filename_extension}.png"
            saved_image = output_dir / "images" / image_filename
            image = Image.fromarray(image)
            image.save(saved_image)