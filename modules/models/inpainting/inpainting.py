from torch import nn
import os
import torch
import cv2
import numpy as np
from omegaconf import OmegaConf
from .model.lama_model import DefaultInpaintingTrainingModule
from .utils.tools import load_img_to_array, dilate_mask, move_to_device, pad_tensor_to_modulo, save_array_to_img


class Inpanting(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_result = config.inpaint.save_result
        self.output_folder = os.path.join(config.output_folder, "inpaint")
        self.model = self.load_model(config).to(self.device) 
        # # 假设 model 是一个 PyTorch Lightning 模型实例
        # # 保存模型权重
        torch.save(self.model.state_dict(), './bbb.pth')
        

        self.model.eval()
    
    def load_model(self, config):
        kind = config.training_model.kind
        kwargs = dict(config.training_model)
        kwargs.pop('kind')
        if kind == 'default':
            model = DefaultInpaintingTrainingModule(config, **kwargs)
        else:
            raise ValueError(f'Unknown trainer module {kind}')
        
        state = torch.load(config.inpaint.model_path, map_location='cpu')
        model.load_state_dict(state['state_dict'], strict=False)
        return model

    def mask_img(self, image, boxes):
        image = cv2.imread(image)
        height, width = image.shape[:2]

        mask = np.zeros((height, width), dtype=np.uint8)
        for box in boxes:
            top_left = (box[0], box[1]) 
            bottom_right = (box[0]+box[2],box[1]+box[3])  
            cv2.rectangle(mask, top_left, bottom_right, 255, -1)
        return mask
    
    @torch.no_grad()
    def inpaint_img_with_lama(
            self,
            img: np.ndarray,
            mask: np.ndarray,
            mod=8,
    ):

        assert len(mask.shape) == 2
        if np.max(mask) == 1:
            mask = mask * 255
        img = torch.from_numpy(img).float().div(255.)
        mask = torch.from_numpy(mask).float()

        batch = {}
        batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
        batch['mask'] = mask[None, None]
        unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
        batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
        batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
        batch = move_to_device(batch, self.device)
        batch['mask'] = (batch['mask'] > 0) * 1

        batch = self.model(batch)
        cur_res = batch[self.config.out_key][0].permute(1, 2, 0)
        cur_res = cur_res.detach().cpu().numpy()

        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        return cur_res
    
    def infer(self, img, boxes):
        mask = self.mask_img(img, boxes)
        img_path = img
        img = load_img_to_array(img)

        if self.config.dilate_kernel_size is not None:
            mask = dilate_mask(mask, self.config.dilate_kernel_size)
        img_inpainted = self.inpaint_img_with_lama(img, mask)
        if self.save_result:
            os.makedirs(self.output_folder, exist_ok=True)
            save_array_to_img(img_inpainted, os.path.join(self.output_folder, os.path.basename(img_path)))
        
        return img_inpainted




