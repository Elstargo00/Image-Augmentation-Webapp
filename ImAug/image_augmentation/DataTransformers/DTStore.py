from torch.utils.data import Dataset
from .utils import text_to_hex, extract_transforming_name
from pathlib import Path
import os
from PIL import Image
import numpy as np

import albumentations as A

class TransPack(Dataset):

    def __init__(self, dataset_dir="./datasets/", transform=None):
        self.data = []
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        self.images_dir = self.dataset_dir / "images"
        self.labels_dir = self.dataset_dir / "labels"

        images = os.listdir(self.images_dir)
        labels = os.listdir(self.labels_dir)
        images.sort()
        labels.sort()
        self.data += list(zip(images, labels))


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        image_filename, label_filename = self.data[index]
        img = Image.open(self.images_dir / image_filename)
        img = np.array(img)

        with open(self.labels_dir / label_filename) as file:
            labels = file.readlines()
            labels = [label.strip().split(" ") for label in labels]
            bboxes = []
            class_labels = []

            print(label_filename)

            for label in labels:
                # changable format
                bboxes.append([float(param) for param in label[1:]] + [str(label[0])]) 
                class_labels.append(str(label[0]))

            if self.transform is not None:
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
        

    def compose(self, format="yolo", min_vis=0.8):
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

    if transforming_option == "oneTrans":
        for each in transforming_list:
            transformat = TransFormat(output_dir)
            filename_extension = text_to_hex(
                extract_transforming_name(each["format_type"])
            )
            transformat.append_format(each)
            transform = transformat.compose(format="yolo", min_vis=0.7)

            dataset = TransPack(
                dataset_dir = dataset_dir,
                transform = transform
            )

            tag_ver = output_dir.parts[-1].split('_')[-1]

            for image_filename, image, label_filename, bboxes in dataset:
                label_filename = f"{label_filename[: -4]}_{tag_ver}_{filename_extension}.txt"
                saved_label = output_dir / "labels" / label_filename

                with open(saved_label, 'a') as label_file:
                    for bbox in bboxes:
                        label_file.write(f"{str(bbox[-1])} {str(bbox[0])} {str(bbox[1])} {str(bbox[2])} {str(bbox[3])}\n")

                image_filename = f"{image_filename[: -4]}_{tag_ver}_{filename_extension}.png"
                saved_image = output_dir / "images" / image_filename
                image = Image.fromarray(image)
                image.save(saved_image)


    elif transforming_option == "allTrans":
        transformat = TransFormat(output_dir)
        filename_extension = []
        for each in transforming_list:
            filename_extension.append(extract_transforming_name(each["format_type"]))
            transformat.append_format(each)
        
        filename_extension = text_to_hex(", ".join(filename_extension))
        transform = transformat.compose(format="yolo", min_vis=0.7)
        
        dataset = TransPack(
            dataset_dir = dataset_dir,
            transform = transform
        )

        tag_ver = output_dir.parts[-1].split('_')[-1]

        for image_filename, image, label_filename, bboxes in dataset:
            label_filename = f"{label_filename[: -4]}_{tag_ver}_{filename_extension}.txt"
            saved_label = output_dir / "labels" / label_filename

            with open(saved_label, 'a') as label_file:
                for bbox in bboxes:
                    label_file.write(f"{str(bbox[-1])} {str(bbox[0])} {str(bbox[1])} {str(bbox[2])} {str(bbox[3])}\n")

            image_filename = f"{image_filename[: -4]}_{tag_ver}_{filename_extension}.png"
            saved_image = output_dir / "images" / image_filename
            image = Image.fromarray(image)
            image.save(saved_image)