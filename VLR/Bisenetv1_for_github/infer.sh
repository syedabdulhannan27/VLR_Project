#!/bin/bash

# Define the paths and dictionaries
# img_path="/home/ishan/Bisenetv1_for_github/test-img/validation"
img_path="/home/ishan_mamadapur/VLR_Project/bisenetv1/Bisenetv1_for_github/test-img"

dspth_dict=(
    "/home/syeda/VLR_Project/VLR/testVLR/images"
    "/home/syeda/VLR_Project/VLR/P3M-10k/validation/P3M-500-NP/original_image"
)

respth_dict=(
    "/home/syeda/VLR_Project/VLR/testVLR/cropped_segment_outputs"
    "/home/syeda/VLR_Project/VLR/P3M-10k/segmented_outputs"
)

cp_dict=(
    "/home/syeda/VLR_Project/VLR/Bisenetv1_for_github/res/model_best_v2.pth"
)
dspth_index=0
respth_index=0
cp_index=0

# Get the paths based on the index
dspth="${dspth_dict[$dspth_index]}"
respth="${respth_dict[$respth_index]}"
cp_path="${cp_dict[$cp_index]}"

infer=/home/syeda/VLR_Project/VLR/Bisenetv1_for_github/inference.py
# infer=/home/ishan/Bisenetv1_for_github/inference_P3M.py
# infer=/home/ishan/Bisenetv1_for_github/unet_inference_P3M.py

# Inference command
python "$infer" --dspth "$dspth" --respth "$respth" --cp_path "$cp_path"