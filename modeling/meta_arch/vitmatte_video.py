import torch
import torch.nn as nn
from kornia.morphology import erosion
from ..decoder.detail_capture import Detail_Capture
from ..backbone.backbone import Backbone


class ViTMatte(nn.Module):
    def __init__(self,
                 *,
                 backbone: Backbone,
                 pixel_mean,
                 pixel_std,
                 input_format,
                 size_divisibility,
                 decoder: Detail_Capture,
                 freeze_backbone = False
                 ):
        super(ViTMatte, self).__init__()
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        self.backbone = backbone
        self.input_format = input_format
        self.size_divisibility = size_divisibility
        self.decoder = decoder
        self.register_buffer(
            "pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        self.recurrent = decoder.recurrent
    

    @property
    def device(self):
        return self.pixel_mean.device


    def forward(self, batched_inputs):
        # Composite and move T axis to batch
        images, targets, H, W, B, T = self.preprocess_inputs(batched_inputs)
        features = self.backbone(images)
        trimap = images[:, 3:4, :H, :W].unflatten(0, (B, T))
        # Retrieve T axis
        if self.recurrent:
            outputs, rec = self.decoder(features.unflatten(0, (B, T)), images.unflatten(0, (B, T)))
            return {'pha': outputs['phas'][:, :, :H, :W].unflatten(0, (B, T)), 'trimap': trimap, 'rec': rec}
        else:
            outputs = self.decoder(features, images)
            return {'pha': outputs['phas'][:, :, :H, :W].unflatten(0, (B, T)), 'trimap': trimap}


    def preprocess_inputs(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """   
        images = batched_inputs["image"].to(self.device)
        trimap = batched_inputs['trimap'].to(self.device)
        B, T = images.shape[:2]
        images = images.flatten(0, 1)
        trimap = trimap.flatten(0, 1)

        if 'fg' in batched_inputs.keys():
            trimap[trimap < 85] = 0
            trimap[trimap >= 170] = 1
            trimap[trimap >= 85] = 0.5

        images = torch.cat((images, trimap), dim=1)
        
        B_, C, H, W = images.shape
        if images.shape[-1]%32!=0 or images.shape[-2]%32!=0:
            new_H = (32-images.shape[-2]%32) + H
            new_W = (32-images.shape[-1]%32) + W
            new_images = torch.zeros((images.shape[0], images.shape[1], new_H, new_W)).to(self.device)
            new_images[:,:,:H,:W] = images[:,:,:,:]
            del images
            images = new_images

        if "alpha" in batched_inputs:
            phas = batched_inputs["alpha"].to(self.device)
        else:
            phas = None

        return images, dict(phas=phas), H, W, B, T