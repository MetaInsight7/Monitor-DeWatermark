import argparse

from modules.models.detect.detect import YOLOv8


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="./modules/weights/detect/detect.onnx", help="Input your ONNX model.")
parser.add_argument("--input_folder", type=str, default='./img', help="Path to input image folder.")
parser.add_argument("--output_folder", type=str, default='./out', help="Path to save image folder.")
parser.add_argument("--conf_thres", type=float, default=0.4, help="Confidence threshold")
parser.add_argument("--iou_thres", type=float, default=0.7, help="NMS IoU threshold")
parser.add_argument("--save_img", action='store_true', help=" save or not.")
args = parser.parse_args()

# Create an instance of the YOLOv8 class with the specified arguments
detection = YOLOv8(args.model, args.input_folder, args.output_folder, args.conf_thres, args.iou_thres, args.save_img)

# Perform object detection and obtain the output image
detection.main()