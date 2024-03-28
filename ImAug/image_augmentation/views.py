from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.conf import settings

import albumentations as A
from .DataTransformers.DTStore import TransPack, TransFormat, apply_transform
from .DataTransformers.DTsplit import split_data
from pathlib import Path
from PIL import Image
import ast

from .AugParams.parameters import (get_affine_params, get_random_crop_params,
                                   get_center_crop_params, get_horizontal_flip_params,
                                   get_vertical_flip_params, get_togray_params,
                                   get_gauss_noise_params)


def index(request):
    return render(request, "image_augmentation/index.html")


def process_augmentation(request):
    if request.method == "POST":

        transforming_list = []

        affine_params = get_affine_params(request)
        if affine_params:
            translate_percent, p, rotate, shear = affine_params
            transforming_list.append({
                "format_type": A.Affine,
                "params": {
                    "translate_percent": translate_percent,
                    "p": p,
                    "rotate": rotate,
                    "shear": shear
                }
            })

        random_crop_params = get_random_crop_params(request)
        if random_crop_params:
            width, height, p = random_crop_params
            transforming_list.append({
                "format_type": A.RandomCrop,
                "params": {"width": width, "height": height, "p": p}
            })

        center_crop_params = get_center_crop_params(request)
        if center_crop_params:
            width, height, p = center_crop_params
            transforming_list.append({
                "format_type": A.CenterCrop,
                "params": {"width": width, "height": height, "p": p}
            })

        horizontal_flip_params = get_horizontal_flip_params(request)
        if horizontal_flip_params:
            p = horizontal_flip_params
            transforming_list.append({
                "format_type": A.HorizontalFlip,
                "params": {"p": p}
            })

        vertical_flip_params = get_vertical_flip_params(request)
        if vertical_flip_params:
            p = vertical_flip_params
            transforming_list.append({
                "format_type": A.VerticalFlip,
                "params": {'p': p}
            })


        togray_params = get_togray_params(request)
        if togray_params:
            p = togray_params
            transforming_list.append({
                "format_type": A.ToGray,
                "params": {'p': p}
            })


        gauss_noise_params = get_gauss_noise_params(request)
        if gauss_noise_params:
            p, mean, var_limit = gauss_noise_params
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

        if request.POST.get("with_label_augmented"):
            with_label_augmented = request.POST.get("with_label_augmented")
            with_label_augmented = bool(with_label_augmented)
        else:
            with_label_augmented = False

        # _____ _____ setup augmented outputs _____ _____
        output_folder = request.POST.get("augmented_name")
        tag_ver = request.POST.get("tag_ver")
        if not output_folder:
            output_folder = "augmented"
        if not tag_ver:
            tag_ver = "V1"
        outs = Path(f"{settings.MEDIA_ROOT}/{output_folder}_{tag_ver}")

        dataset_dir = Path(f"{settings.MEDIA_ROOT}/datasets")

        augmented_scheme = request.POST.get("AugmentedScheme")
        print(augmented_scheme)
        transforming_option = [with_label_augmented, augmented_scheme]
        print("transforming_option", transforming_option)

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
    


