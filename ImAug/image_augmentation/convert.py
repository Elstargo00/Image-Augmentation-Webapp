import json
import cv2
import os
import shutil

class_info = {"dotted": 0, "body": 1, "marked": 2}    


def convert_to_yolo_format(datapath, json_file, output_dir):


    with open(json_file, 'r') as f:
        data = json.load(f)
    
    for row in data:
        file_name = row["External ID"]
        src = os.path.join(DATAPATH, file_name)
        img = cv2.imread(src)
        shutil.copyfile(src, os.path.join(output_dir, file_name))
        img_height, img_width, _ = img.shape
        objects = row["Label"]["objects"]
        file_name = file_name.replace("bmp", "txt")
                
        output_file = open(f"{output_dir}/{file_name}", 'w')

        for obj in objects:
            class_name = obj["value"]
            class_id = class_info[class_name]
            bbox = obj["bbox"]
            top, left, height, width = bbox["top"], bbox["left"], bbox["height"], bbox["width"]

            # Convert to YOLO format
            x_center = (left + width / 2) / img_width
            y_center = (top + height / 2) / img_height
            width = width / img_width
            height = height / img_height

            output_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        output_file.close()


if __name__ == "__main__":

    DATAPATH = "/home/elstargo00/Projects/donaldson/actuator/datasets_v2"

    convert_to_yolo_format(DATAPATH, "labelbox_export.json", "./datasets_converted")
