from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.conf import settings

import albumentations as A
from .DataTransformers.DTStore import TransPack, TransFormat, apply_transform
from .DataTransformers.DTsplit import split_data
from pathlib import Path
from PIL import Image
import ast


def index(request):
    return render(request, "image_augmentation/index.html")


def process_augmentation(request):
    if request.method == "POST":

        transforming_list = []
        
        if bool(request.POST.get("affine")):
            translate_percent = request.POST.get("affine_translate_percent")
            translate_percent = float(translate_percent) if translate_percent else 0 # field check
            p = request.POST.get("affine_p")
            p = float(p) if p else 0.5
            rotate = float(request.POST.get("affine_rotate"))
            shear = ast.literal_eval(request.POST.get("affine_shear"))
            if not p:
                p = 0.5
            transforming_list.append({
                "format_type": A.Affine,
                "params": {
                    "translate_percent": translate_percent,
                    "p": p,
                    "rotate": rotate,
                    "shear": shear
                }
            })


        if bool(request.POST.get("random_crop")):
            width = int(float((request.POST.get("random_crop_width")))) # field check
            height = int(float(request.POST.get("random_crop_height"))) # field check
            p = float(request.POST.get("random_crop_p"))
            if not p:
                p = 0.5
            transforming_list.append({
                "format_type": A.RandomCrop,
                "params": {"width": width, "height": height, "p": p}
            })


        if bool(request.POST.get("center_crop")):
            width = int(float(request.POST.get("center_crop_width"))) # field check
            height = int(float(request.POST.get("center_crop_height"))) # field check
            p = float(request.POST.get("center_crop_p"))
            if not p:
                p = 0.5
            transforming_list.append({
                "format_type": A.CenterCrop,
                "params": {"width": width, "height": height, "p": p}
            })


        if bool(request.POST.get("horizontal_flip")):
            p = float(request.POST.get("horizontal_flip_p"))
            if not p:
                p = 0.5
            transforming_list.append({
                "format_type": A.HorizontalFlip,
                "params": {"p": p}
            })


        if bool(request.POST.get("vertical_flip")):
            p = float(request.POST.get("vertical_flip_p"))
            if not p:
                p = 0.5
            transforming_list.append({
                "format_type": A.VerticalFlip,
                "params": {'p': p}
            })

        if bool(request.POST.get("togray")):
            p = float(request.POST.get("togray_p"))
            if not p:
                p = 0.5
            transforming_list.append({
                "format_type": A.ToGray,
                "params": {'p': p}
            })

        if bool(request.POST.get("gauss_noise")):
            p = float(request.POST.get("gauss_noise_p"))
            if not p:
                p = 0.5
            mean = float(request.POST.get("gauss_noise_mean"))
            if not mean:
                mean = 0
            var_limit = float(request.POST.get("gauss_noise_var"))
            transforming_list.append({
                "format_type": A.GaussNoise,
                "params": {'p': p, "mean": mean, "var_limit": var_limit}
            })

        if request.POST.get("split_data"):
            train_validate_testsize = request.POST.get("train_validate_testsize")

            if not train_validate_testsize:
                train_validate_testsize = 0
            else:
                train_validate_testsize = float(train_validate_testsize)

            test_testsize = request.POST.get("test_testsize")

            if not test_testsize:
                test_testsize = 0
            else:
                test_testsize = float(test_testsize)
        else:
            train_validate_testsize = 0
            test_testsize = 0

        if request.POST.get("images_w_label_augmented"):
            images_w_label_augmented = request.POST.get


        # _____ _____ setup augmented outputs _____ _____
        output_folder = request.POST.get("augmented_name")
        tag_ver = request.POST.get("tag_ver")
        if not output_folder:
            output_folder = "augmented"
        if not tag_ver:
            tag_ver = "V1"
        outs = Path(f"{settings.MEDIA_ROOT}/{output_folder}_{tag_ver}")

        dataset_dir = Path(f"{settings.MEDIA_ROOT}/datasets")

        transforming_option = request.POST.get("TransformOption")

        # Generate directory according to the split params
        # 1. no split
        if train_validate_testsize == 0 and test_testsize == 0:
            outs.mkdir(parents=True, exist_ok=True)

            apply_transform(dataset_dir, transforming_option, transforming_list, outs)


        # 2. split into train (train_validate) & test
        elif train_validate_testsize == 0 and test_testsize != 0:
            train_validate_dir = outs / "train"
            test_dir = outs / "test"

            split_data(
                base_dir = dataset_dir,
                train_dir = train_validate_dir,
                validate_dir = None,
                test_dir = test_dir,
                validate_size = train_validate_testsize,
                test_size = test_testsize
            )

            apply_transform(dataset_dir, transforming_option, transforming_list, train_validate_dir)

        # 3. split into train & validate (Ultimate Training)
        elif train_validate_testsize != 0 and test_testsize == 0:
            train_dir = outs / "train"
            validate_dir = outs / "validate"

            split_data(
                base_dir = dataset_dir,
                train_dir = train_dir,
                validate_dir = validate_dir,
                test_dir = None,
                validate_size = train_validate_testsize,
                test_size = test_testsize, 
            )

            apply_transform(dataset_dir, transforming_option, transforming_list, train_dir)
            apply_transform(dataset_dir, transforming_option, transforming_list, validate_dir)

        # 4. split into train & validate & test
        elif train_validate_testsize != 0 and test_testsize != 0:

            train_dir = outs / "train"
            validate_dir = outs / "validate"
            test_dir = outs / "test"

            split_data(
                base_dir = dataset_dir,
                train_dir = train_dir,
                validate_dir = validate_dir,
                test_dir = test_dir, 
                validate_size = train_validate_testsize,
                test_size = test_testsize,
            )

            apply_transform(dataset_dir, transforming_option, transforming_list, train_dir)


        return HttpResponse("Finish Augmentation!")
    
    else:
        return HttpResponseRedirect('index')
    







        # There are two options for applying augmentation


        # 1st: 1 augment / 1 sample
        # if transform_option == "oneTrans":
        #     for each in transforming_list:
        #         # resetting factor
        #         transformat = TransFormat(outs)
        #         filename_extension = text_to_hex(
        #             extract_transforming_name(each["format_type"])
        #         )
        #         #
        #         transformat.append_format(each)
        #         transform = transformat.compose(format="yolo", min_vis=0.7)

        #         dataset = TransPack(
        #             dataset_dir = f"{settings.MEDIA_ROOT}/datasets",
        #             transform = transform
        #         )
                
        #         for image_filename, image, label_filename, bboxes in dataset:

        #             label_filename = f"{label_filename[:-4]}_{tag_ver}_{filename_extension}.txt"
        #             saved_label = label_outs / label_filename

        #             with open(saved_label, 'a') as label_file:
        #                 for bbox in bboxes:
        #                     label_file.write(f"{str(bbox[-1])} {str(bbox[0])} {str(bbox[1])} {str(bbox[2])} {str(bbox[3])}\n")

        #             image_filename = f"{image_filename[:-4]}_{tag_ver}_{filename_extension}.png"
        #             saved_image = image_outs / image_filename
        #             image = Image.fromarray(image)
        #             image.save(saved_image)


        # # 2nd: all augments / 1 sample
        # if transform_option == "AllTrans":
        #     transformat = TransFormat(outs)
        #     filename_extension = []
        #     for each in transforming_list:
        #         filename_extension.append(extract_transforming_name(each["format_type"]))
        #         transformat.append_format(each)

        #     filename_extension = text_to_hex(", ".join(filename_extension))

        #     transform = transformat.compose(format="yolo", min_vis=0.7)

        #     dataset = TransPack(
        #         dataset_dir = f"{settings.MEDIA_ROOT}/datasets",
        #         transform = transform
        #     )

        #     for image_filename, image, label_filename, bboxes in dataset:
        #         label_filename = f"{label_filename[:-4]}_{tag_ver}_{filename_extension}.txt"
        #         saved_label = label_outs / label_filename

        #         with open(saved_label, 'a') as label_file:
        #             for bbox in bboxes:
        #                 label_file.write(f"{str(bbox[-1])} {str(bbox[0])} {str(bbox[1])} {str(bbox[2])} {str(bbox[3])}\n")

        #         image_filename = f"{image_filename[:-4]}_{tag_ver}_{filename_extension}.png"
        #         saved_image = image_outs / image_filename
        #         image = Image.fromarray(image)
        #         image.save(saved_image)


