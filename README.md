# Spatio-Temporal-Anonymized
> Eliminating sensitive information from monitoring data

## Framework
```
.
├── modules
│   ├── models
│   │   └── detect
│   │       └── detect.py
│   └── weights
│       └── detect
│           └── detect.onnx
├── utils
│       └── detect 
└── infer.py
```

## Pretrained Model
pretrained model are stored in ./modules/weights with following arrangement.

## Environment
- Python == 3.8
- CUDA == 11.8
- onnxruntime-gpu== 1.18
- opencv
- numpy

## Usage

​	Run the project by follow command:

```
python infer_folder.py  --model model.onnx --input_folder ./img --output_folder ./out --conf_thres 0.5 --iou_thres 0.5 [--save_img]
```

- `--model`:  Input your ONNX model path.
- `--input_folder`:  Path to the folder containing input images or single image.
- `--output_folder`:  Path to the folder where the results will be saved.
- `--conf_thres`:  Confidence threshold
- `--iou_thres`:  NMS IoU threshold
- `--save_img`:  Flag to determine whether to save img options are `True` or `False`, default is `False`.
