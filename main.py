import os
from tqdm import tqdm
import argparse
from modules.models.inpainting.utils.tools import get_img, save_array_to_img
from modules.models.inpainting.inpainting import Inpanting
from modules.models.detect.detect import YOLOv8
from omegaconf import OmegaConf


# Create a parser
parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", type=str, default="./test_img", help="Input your image folder.")
parser.add_argument("--output_folder", type=str, default='./out', help="Path to save image folder.")
parser.add_argument("--config", type=str,default="./configs/default.yaml",help="The path to the config file of lama model. ""Default: the config of big-lama")
args = parser.parse_args()


config = OmegaConf.load(args.config)
config.input_folder = args.input_folder
config.output_folder = args.output_folder

detect = YOLOv8(config)
inpaint = Inpanting(config)

img_path_list = get_img(args.input_folder)
# Process each image
for img_path in tqdm(img_path_list):
    # task1：Use target detection to draw a box
    detect_result = detect.infer(img_path)
    boxes = detect_result['box']
    # task2：Restore the picture using inpaint
    img_inpainted = inpaint.infer(img_path, boxes)
    
