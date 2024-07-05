from torch import nn
import torch
import torch.nn.functional as F
from .generator import make_generator


class DefaultInpaintingTrainingModule(nn.Module):
    def __init__(self, config, *args, concat_mask=True, add_noise_kwargs=None, noise_fill_hole=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config
        self.concat_mask = concat_mask
        self.add_noise_kwargs = add_noise_kwargs
        self.noise_fill_hole = noise_fill_hole

        self.generator = make_generator(config, **self.config.generator)
    
    def make_multiscale_noise(self, base_tensor, scales=6, scale_mode='bilinear'):
        batch_size, _, height, width = base_tensor.shape
        cur_height, cur_width = height, width
        result = []
        align_corners = False if scale_mode in ('bilinear', 'bicubic') else None
        for _ in range(scales):
            cur_sample = torch.randn(batch_size, 1, cur_height, cur_width, device=base_tensor.device)
            cur_sample_scaled = F.interpolate(cur_sample, size=(height, width), mode=scale_mode, align_corners=align_corners)
            result.append(cur_sample_scaled)
            cur_height //= 2
            cur_width //= 2
        return torch.cat(result, dim=1)
    

    def forward(self, batch):
        img = batch['image']
        mask = batch['mask']
        
        masked_img = img * (1 - mask)

        if self.add_noise_kwargs is not None:
            noise = self.make_multiscale_noise(masked_img, **self.add_noise_kwargs)
            if self.noise_fill_hole:
                masked_img = masked_img + mask * noise[:, :masked_img.shape[1]]
            masked_img = torch.cat([masked_img, noise], dim=1)

        if self.concat_mask:
            masked_img = torch.cat([masked_img, mask], dim=1)

        batch['predicted_image'] = self.generator(masked_img)
        batch['inpainted'] = mask * batch['predicted_image'] + (1 - mask) * batch['image']

        return batch