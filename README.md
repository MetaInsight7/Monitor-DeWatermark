## Language
- [English](README.md)
- [中文](README_zh.md)

<div align="center">
<h1>Monitor-DeWatermark</h1>
</div>
<h3>An innovative and efficient watermark removal method designed to accurately identify and elegantly remove watermarks from images</h3>

<h2>🌄 Framework</h2>

```
├── configs
├── docs
├── modules
│   ├── models
│   │   └── detect
│   │   └── inpainting
│   └── weights
│       └── detect
│       └── inpainting
├── utils
│       └── img_tools.py
├── out
├── test_img
├── 
└── main.py
```
`configs/` : Save the parameters of the model 

`modules/` : Under the folder is the `core code` of this project👌

`test_img/` : Under the folder are some test images

`out/` : Under the folder is the output picture, i. e. the item after the watermark is removed

<h2>📝 Introduction</h2>
The watermark removal process is divided into two steps:

 - The first step is [Watermark Detection](./out/detect), with the main code located in the `modules/models/detect/` folder, the weights file is saved in the `modules/weights/detect/` folder.

 - The second step is [Watermark Removal](./out/inpaint), with the main code located in the `modules/models/inpainting/` folder. Since the weight file will be updated periodically... No upload.

[🤗 Detect weight download (verification code: k1gv)](https://pan.baidu.com/s/1c7Wa6tDIE0UgP55cmXmT4Q?pwd=k1gv)

[🤗 Inpaint weight download (verification code: 7urm)](https://pan.baidu.com/s/1QLX6S5ssMDLsUsYslHgDng?pwd=7urm)

<h2>📝 Usage</h2>

<details open>
<summary style="font-size: 20px;">Show</summary>
<h3>Demo1<h3>
<img src="./docs/demo2.gif" />

<h3>Demo2<h3>
<img src="./docs/demo1.gif" />
</details>

<details open>
<summary style="font-size: 20px;">Installation</summary>

Requires `python>=3.8`
```bash
pip install -r requirements.txt 
```
</details>

<details open>
<summary style="font-size: 20px;">Run</summary>

Run the project with the following command:
```bash
python main.py --input_folder '/path/to/your/input_folder' --output_folder '/path/to/your/output_folder' --config './configs/default.yaml'

```
- '/path/to/your/input_folder' change the watermark folder ✔️
- '/path/to/your/output_folder' change the output folder🔜

</details>


