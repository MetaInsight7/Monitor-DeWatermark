## Language
- [English](README.md)
- [中文](README_zh.md)

<div align="center">
<h1>去除水印 </h1>
</div>
<h3>一种创新且高效的去水印方法，旨在精确识别并优雅去除图片中的水印</h3>

<h2>📝 简介</h2>
将去除水印分成两步

 - 第一步[识别水印](https://github.com/MetaInsight7/Spatio-Temporal-Anonymized.git)
 ，主要代码在 `detection/` 文件夹下,详细请参见 [README.md](./detection/README.md)

 - 第二部[去除水印](https://github.com/geekyutao/Inpaint-Anything.git)
 ，主要代码在 `inpaint/` 文件夹下,详细请参见[README.md](./inpaint/README.md)
 <h2>📝 使用</h2>

<details open>
<summary>安装</summary>

要求 `python>=3.8`
```bash
python -m pip install torch torchvision torchaudio
python -m pip install -r requirements.txt 
```
</details>

<details open>
<summary>运行</summary>

通过以下命令运行项目:
```bash
python main.py --input_folder '/path/to/your/input_folder' --output_folder '/path/to/your/output_folder'

```
- 将'/path/to/your/input_folder'切换成所需要去水印的文件夹
- 将'/path/to/your/output_folder'切换成保存的文件夹

</details>


<details open>
<summary>展示</summary>
<h3>效果演示1<h3>
<img src="./assets/demo1.gif" />

<h3>效果演示2<h3>
<img src="./assets/demo2.gif" />
</details>