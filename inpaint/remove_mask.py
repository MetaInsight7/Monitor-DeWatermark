import torch
import sys
import argparse
import cv2
import numpy as np
from PIL import Image

from .lama_inpaint import inpaint_img_with_lama
from .utils import load_img_to_array, dilate_mask

def mask2(image, boxes):
    image = cv2.imread(image)
    height, width = image.shape[:2]

    mask = np.zeros((height, width), dtype=np.uint8)
    for box in boxes:
        top_left = (box[0], box[1]) 
        bottom_right = (box[0]+box[2],box[1]+box[3])  
        cv2.rectangle(mask, top_left, bottom_right, 255, -1)

    return mask

def setup_args(parser):

    parser.add_argument(
    "--point_labels", type=int, nargs='+', default=1,
        help="The labels of the point prompt, 1 or 0.",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=25,
        help="Dilate kernel size. Default: None",
    )
    parser.add_argument(
        "--output_dir", type=str, default='results',
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--lama_config", type=str,
        default="inpaint/lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str, default='inpaint/big-lama/',
        help="The path to the lama checkpoint.",
    )
    parser.add_argument(
        "--input_folder", type=str, default="./test_img", 
        help="Input your image folder.")
    
    parser.add_argument(
        "--output_folder", type=str, default='./out', 
                        help="Path to save image folder.")
import os
import yaml
from omegaconf import OmegaConf
from saicinpainting.training.trainers import load_checkpoint

def load_model(config_p,ckpt_p, device):
    predict_config = OmegaConf.load(config_p)
    predict_config.model.path = ckpt_p
    device = torch.device(device)

    train_config_path = os.path.join(
        predict_config.model.path, 'config.yaml')

    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
  
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(
        predict_config.model.path, 'models',
        predict_config.model.checkpoint
    )
    model = load_checkpoint(
        train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    if not predict_config.get('refine', False):
        model.to(device)    

    return model

def remove(img, boxes):
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_1 = load_img_to_array(img)    
    mask = mask2(img, boxes)

    if args.dilate_kernel_size is not None:
        mask = dilate_mask(mask, args.dilate_kernel_size) 
    model = load_model(args.lama_config, args.lama_ckpt, device=device)

    img_inpainted = inpaint_img_with_lama(
        img_1, mask,args.lama_config,  model, device=device)
    
    return img_inpainted
