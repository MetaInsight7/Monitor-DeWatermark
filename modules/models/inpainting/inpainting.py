from torch import nn
import os
import torch
import cv2
import numpy as np
from .model.lama_model import DefaultInpaintingTrainingModule
from .utils.tools import dilate_mask, move_to_device, pad_tensor_to_modulo


class Inpanting(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_result = config.inpaint.save_result
        self.output_folder = os.path.join(config.output_folder, "inpaint")
        self.model = self.load_model(config).to(self.device) 
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
        model.load_state_dict(state, strict=False)
        return model

    def mask_img(self, img, boxes):
        height, width = img.shape[:2]

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
    
    def forward(self, img_path, boxes):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = self.mask_img(img, boxes)
        if self.config.dilate_kernel_size is not None:
            mask = dilate_mask(mask, self.config.dilate_kernel_size)
        img_inpainted = self.inpaint_img_with_lama(img, mask)
        if self.save_result:
            os.makedirs(self.output_folder, exist_ok=True)
            cv2.imwrite(os.path.join(self.output_folder, os.path.basename(img_path)), cv2.cvtColor(img_inpainted, cv2.COLOR_RGB2BGR) )
        
        return img_inpainted




