import os
import cv2
import numpy as np
from collections import defaultdict

class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose."""

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
        """Initialize LetterBox object with specific parameters."""
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left

    def __call__(self, labels=None, image=None):
        """Return updated labels and image with added border."""
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels."""
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels

def get_img(input_folder):
    if os.path.isfile(input_folder):
        img_path_list = [input_folder]
    else:
        img_name_list = os.listdir(input_folder)
        img_path_list = [os.path.join(input_folder, img_name) for img_name in img_name_list
                          if img_name.endswith('.jpg') or img_name.endswith('.png')]
    return img_path_list


def save_labels(result, save_dir):
    """
    保存目标检测的标签信息,按照class_id顺序写入
    
    Args:
        results (dict): 目标检测的推理结果,格式同您提供的
        save_dir (str): 保存标签文件的目录
    """
    # 获取图像名称
    img_name = result['img_name']
    boxes = result['box']
    class_ids = result['class_id']
    scores = result['score']

    img = cv2.imread(img_name)
    img_h= img.shape[0]
    img_w = img.shape[1]
    
    # 使用defaultdict分类存储标签信息
    label_dict = defaultdict(list)
    for box, class_id, score in zip(boxes, class_ids, scores):
        x, y, w, h = [int(v) for v in box]
        label_str = f"{class_id} {(x + w /2) / img_w} {(y + h /2) / img_h} {w / img_w} {h /img_h}"
        label_dict[class_id].append(label_str)
    # 按class_id顺序写入标签文件
    os.makedirs(save_dir, exist_ok=True)
    label_file = os.path.join(save_dir, os.path.basename(img_name).split('.')[0] + '.txt')
    with open(label_file, 'w') as f:
        for class_id in sorted(label_dict.keys()):
            f.write('\n'.join(label_dict[class_id]))
            f.write('\n')