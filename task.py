import os
import argparse
from detection import infer
import inpaint.remove_mask


parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", type=str, default="./test_img", 
                        help="Input your ONNX model.")
parser.add_argument("--output_folder", type=str, default='./out', help="Path to save image folder.")

#task1：目标检测部分（使用yolov8m识别水印）
#task2：将水印还原(使用inpainting将水印还原为背景)
#inpaint部分模型参数已经写入配置文件中，如果需要修改模型，请修改inpaint/config.py文件


def get_img(self):
    if os.path.isfile(self.input_folder):
        img_path_list = [self.input_folder]
    else:
        img_name_list = os.listdir(self.input_folder)
        img_path_list = [os.path.join(self.input_folder, img_name) for img_name in img_name_list if img_name.endswith('.jpg') or img_name.endswith('.png')]
    return img_path_list

if __name__ == '__main__':
    args = parser.parse_args()
    img_path_list = get_img(args)
    for img_path in img_path_list:
        boxes = infer.yolo_detection(img_path)
        img_inpainted = inpaint.remove_mask.remove(img_path,boxes)
        inpaint.remove_mask.save_array_to_img(img_inpainted,os.path.join(args.output_folder,os.path.basename(img_path)))
        print(f'{img_path}已完成')
