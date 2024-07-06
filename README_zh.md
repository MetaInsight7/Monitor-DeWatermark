## Language
- [English](README.md)
- [中文](README_zh.md)

<div align="center">
<h1>穿越时空的匿名 </h1>
</div>
<h3>一种创新且高效的去水印方法，旨在精确识别并优雅去除图片中的水印</h3>

<h2>🌄 框架</h2>

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
在 `configs/` 文件夹下保存模型的参数

在 `modules/` 文件夹下是本项目 `核心代码`👌

在 `test_img/` 文件夹下是几张测试图片

在 `out/` 文件夹下是几张测试图片



<h2>📝 简介</h2>
本项目将去除水印分成两步

 - 第一步[识别水印](./out/detect)
 ，主要代码在 `modules/models/detect` 文件夹下，权重文件保存在 `modules/weights/detect`文件夹下。
 
 - 第二步[去除水印](./out/inpaint)
 ，主要代码在 `modules/models/inpainting` 文件夹下，权重文件保存在 `modules/weights/inpainting`文件夹下，由于权重文件将进行周期性的更新...✈️，没有上传，可以在[这里🤗](https://huggingface.co/gityihang/inpaint/upload/main)下载，并放入该目录下。
<h2>📝 使用</h2>

<details open>
<summary style="font-size: 20px;">展示</summary>
<h3>效果演示1<h3>
<img src="./docs/demo2.gif" />

<h3>效果演示2<h3>
<img src="./docs/demo1.gif" />
</details>


<details open>
<summary style="font-size: 20px;">安装</summary>

要求 `python>=3.8`
```bash
conda create -n sta python==3.8
conda activate sta
python -m pip install -r requirements.txt 
```
</details>

<details open>
<summary style="font-size: 20px;">运行</summary>

通过以下命令运行项目:
```bash
python main.py --input_folder '/path/to/your/input_folder' --output_folder '/path/to/your/output_folder' --config './configs/default.yaml'
```

- 将'/path/to/your/input_folder'切换成所需要去水印的文件夹✔️

- 将'/path/to/your/output_folder'切换成保存的文件夹🔜

</details>


