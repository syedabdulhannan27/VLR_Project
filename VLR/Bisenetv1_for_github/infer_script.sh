#!/bin/bash

# Define the paths and dictionaries
# img_path="/home/ishan/Bisenetv1_for_github/test-img/validation"
img_path="/home/ishan/Bisenetv1/test-img/val_100"

dspth_dict=(
    "/home/ishan/Bisenetv1/test-img"
    "/home/ishan/Bisenetv1_for_github/test-img/validation"
    "/home/ishan/Bisenetv1/test-img/val_100"
    "/home/ishan/Bisenetv1_for_github/test-img/experimental_images"
)

respth_dict=(
    "/home/ishan/Bisenetv1/res/test_res/latest_val"
    "/home/ishan/Bisenetv1/res/test_res/val_100"
    "/home/ishan/Bisenetv1/res/test_res/unet_run"
    "/home/ishan/Bisenetv1/res/test_res/val_100_no_init"
    "/home/ishan/Bisenetv1_for_github/res/test_res/experimental_output"
)

cp_dict=(
    "/home/ishan/Bisenetv1/res/model_best_v2.pth"
    "/home/ishan/Bisenetv1/res/terminal_test/P3M_test/50.pth"
    "/home/ishan/Bisenetv1_for_github/res/terminal_test/P3M_test_1_1.pth"
    "/home/ishan/Bisenetv1_for_github/res/terminal_test/Unet_2.pth"
    "/home/ishan/Bisenetv1_for_github/res/terminal_test/bisenet_no_init.pth"
)
dspth_index=3
respth_index=4
cp_index=0

# Get the paths based on the index
dspth="${dspth_dict[$dspth_index]}"
respth="${respth_dict[$respth_index]}"
cp_path="${cp_dict[$cp_index]}"

infer=/home/ishan/Bisenetv1_for_github/inference.py
# infer=/home/ishan/Bisenetv1_for_github/inference_P3M.py
# infer=/home/ishan/Bisenetv1_for_github/unet_inference_P3M.py

# Inference command
python "$infer" --dspth "$dspth" --respth "$respth" --cp_path "$cp_path"