from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.conf import settings

import albumentations as A
from .DataTransformers.DTStore import TransPack, TransFormat
from pathlib import Path
from PIL import Image


def index(request):
    return render(request, "image_augmentation/index.html")


def process_augmentation(request):
    if request.method == "POST":

        transforming_list = []

        
        if bool(request.POST.get("affine")):
            translate_percent = float(request.POST.get("affine_translate_percent")) # field check
            p = request.POST.get("affine_p")
            if not p:
                p = 0.5
            p = float(p)
            rotate = float(request.POST.get("affine_rotate"))
            transforming_list.append({
                "format_type": A.Affine,
                "params": {"translate_percent": translate_percent, "p": p, "rotate": rotate}
            })


        if bool(request.POST.get("random_crop")):
            width = int(request.POST.get("random_crop_width")) # field check
            height = int(request.POST.get("random_crop_height")) # field check
            p = request.POST.get("random_crop_p")
            if not p:
                p = 0.5
            p = float(p) 
            transforming_list.append({
                "format_type": A.RandomCrop,
                "params": {"width": width, "height": height, "p": p}
            })


        if bool(request.POST.get("center_crop")):
            width = int(request.POST.get("center_crop_width")) # field check
            height = int(request.POST.get("center_crop_height")) # field check
            p = request.POST.get("center_crop_p")
            if not p:
                p = 0.5
            p = float(p) 
            transforming_list.append({
                "format_type": A.CenterCrop,
                "params": {"width": width, "height": height, "p": p}
            })


        if bool(request.POST.get("horizontal_flip")):
            p = request.POST.get("horizontal_flip_p")
            if not p:
                p = 0.5
            p = float(p)
            transforming_list.append({
                "format_type": A.HorizontalFlip,
                "params": {"p": p}
            })


        if bool(request.POST.get("vertical_flip")):
            p = float(request.POST.get("vertical_flip_p"))
            if not p:
                p = 0.5
            p = float(p)
            transforming_list.append({
                "format_type": A.VerticalFlip,
                "params": {"p": p}
            })


        print(transforming_list)


        # _____ _____ setup augmented outputs _____ _____
        fn_suffix = "v2"
        outs = Path(f"{settings.MEDIA_ROOT}/augmented_{fn_suffix}")
        image_outs = outs / "images"
        label_outs = outs / "labels"
        image_outs.mkdir(parents=True, exist_ok=True)
        label_outs.mkdir(parents=True, exist_ok=True)
        # ____ ____ ____ ____ ____ ____ ____ ____ ____

        print(request.POST.get("TransformOption"))

        # There are two options for applying augmentation

        transform_option = request.POST.get("TransformOption")

        # 1st: 1 augment / 1 sample
        if transform_option == "oneTrans":
            for each in transforming_list:
                # resetting factor
                transformat = TransFormat(outs)
                filename_extension = text_to_hex(
                    extract_transforming_name(each["format_type"])
                )
                #
                transformat.append_format(each)
                transform = transformat.compose(format="yolo", min_vis=0.7)

                dataset = TransPack(
                    dataset_dir = f"{settings.MEDIA_ROOT}/datasets",
                    transform = transform
                )
                
                for image_filename, image, label_filename, bboxes in dataset:

                    label_filename = f"{label_filename[:-4]}_{fn_suffix}_{filename_extension}.txt"
                    saved_label = label_outs / label_filename

                    with open(saved_label, 'a') as label_file:
                        for bbox in bboxes:
                            label_file.write(f"{str(bbox[-1])} {str(bbox[0])} {str(bbox[1])} {str(bbox[2])} {str(bbox[3])}\n")

                    image_filename = f"{image_filename[:-4]}_{fn_suffix}_{filename_extension}.png"
                    saved_image = image_outs / image_filename
                    image = Image.fromarray(image)
                    image.save(saved_image)


        # 2nd: all augments / 1 sample
        if transform_option == "AllTrans":
            transformat = TransFormat(outs)
            filename_extension = []
            for each in transforming_list:
                filename_extension.append(extract_transforming_name(each["format_type"]))
                transformat.append_format(each)

            filename_extension = text_to_hex(", ".join(filename_extension))

            print(filename_extension)

            transform = transformat.compose(format="yolo", min_vis=0.7)

            dataset = TransPack(
                dataset_dir = f"{settings.MEDIA_ROOT}/datasets",
                transform = transform
            )

            for image_filename, image, label_filename, bboxes in dataset:
                label_filename = f"{label_filename[:-4]}_{fn_suffix}_{filename_extension}.txt"
                saved_label = label_outs / label_filename

                with open(saved_label, 'a') as label_file:
                    for bbox in bboxes:
                        label_file.write(f"{str(bbox[-1])} {str(bbox[0])} {str(bbox[1])} {str(bbox[2])} {str(bbox[3])}\n")

                image_filename = f"{image_filename[:-4]}_{fn_suffix}_{filename_extension}.png"
                saved_image = image_outs / image_filename
                image = Image.fromarray(image)
                image.save(saved_image)


        return HttpResponse("Finish Augmentation!")
    else:
        return HttpResponseRedirect('index')
    



# helper function
def text_to_hex(text):
    # encode the text to bytes, then convert to hexadecimal
    hex_output = text.encode("utf-8").hex()
    return hex_output


def extract_transforming_name(obj):
    # Get the full class name
    full_name = str(obj)
    # Extract the class name
    class_name = full_name.split('.')[-1].replace("'>", '')
    return class_name