import os
import argparse
from detection import infer
from inpaint import remove_mask,utils

# Create a parser
parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", type=str, default="./test_img", help="Input your image folder.")
parser.add_argument("--output_folder", type=str, default='./out', help="Path to save image folder.")
args = parser.parse_args()

# Check that the input folder exists
if not os.path.exists(args.input_folder):
    print(f"Enter the folder:{args.input_folder} does not exist.")
    exit(1)

# Checks if the output folder exists, and creates if it does not
if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

# Gets all the image paths
img_path_list = utils.get_img(args.input_folder)

# Process each image
for img_path in img_path_list:
    # task1：Use target detection to draw a box
    boxes = infer.yolo_detection(img_path)
    # task2：Restore the picture using inpaint
    img_inpainted = remove_mask.remove(img_path, boxes)
    # task3：Save the picture
    utils.save_array_to_img(img_inpainted, os.path.join(args.output_folder, os.path.basename(img_path)))
    print(f'{os.path.basename(img_path)} has removed Watermark')
