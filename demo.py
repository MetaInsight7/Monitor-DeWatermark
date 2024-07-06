import gradio as gr
from tqdm import tqdm
import argparse
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


def main():
    detect = YOLOv8(config)
    inpaint = Inpanting(config)
    
    def process_image(input_image):
        # task1：Use target detection to draw a box
        detect_result = detect(input_image)
        boxes = detect_result['box']
        # task2：Restore the picture using inpaint
        img_inpainted = inpaint(input_image, boxes)
        return img_inpainted
    
    gr.Interface(
        fn=process_image,
        inputs=gr.Image(type="filepath", label="Input Image"),
        outputs=gr.Image(label="Output Image"),
        title="🔥🔥Image Inpainting and Watermark Removal",
        description="<h3>Upload an image, and the app will detect any watermarks and remove them using inpainting.</h3> \
                    <h3>✨ Team </h3>\
                    🔥 detect：卢克珊 </br>\
                    ⚡️  Inpanting：鲍一航 </br>\
                    🚀 Framwork：胥子元 </br>\
                    ⚙️ Leader：吴金儒",
    ).launch(server_name="0.0.0.0")

if __name__ == "__main__":
    main()
