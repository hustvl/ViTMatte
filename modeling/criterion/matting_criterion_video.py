import torch
import torch.nn as nn
import torch.nn.functional as F

class MattingCriterion(nn.Module):
    def __init__(self,
                 *,
                 losses,
                 ):
        super(MattingCriterion, self).__init__()
        self.losses = losses


    def loss_gradient_penalty(self, sample_map, preds, targets):
        preds = preds['phas'].flatten(0, 1)
        targets = targets['phas'].flatten(0, 1)
        device = preds.device

        #sample_map for unknown area
        if torch.sum(sample_map) == 0:
            scale = 0
        else:
            scale = 1 / torch.sum(sample_map)

        #gradient in x
        sobel_x_kernel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).type(dtype=preds.type()).to(device)
        delta_pred_x = F.conv2d(preds, weight=sobel_x_kernel, padding=1)
        delta_gt_x = F.conv2d(targets, weight=sobel_x_kernel, padding=1)

        #gradient in y 
        sobel_y_kernel = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).type(dtype=preds.type()).to(device)
        delta_pred_y = F.conv2d(preds, weight=sobel_y_kernel, padding=1)
        delta_gt_y = F.conv2d(targets, weight=sobel_y_kernel, padding=1)

        #loss
        loss = (F.l1_loss(
                delta_pred_x * sample_map,
                delta_gt_x * sample_map,
                reduction='sum'
            ) * scale + \
                F.l1_loss(
                delta_pred_y * sample_map,
                delta_gt_y * sample_map,
                reduction='sum'
            ) * scale + \
            0.01 * torch.sum(torch.abs(delta_pred_x * sample_map)) * scale +  \
            0.01 * torch.sum(torch.abs(delta_pred_y * sample_map)) * scale)

        return dict(loss_gradient_penalty=loss)


    def loss_pha_laplacian(self, preds, targets):
        assert 'phas' in preds and 'phas' in targets
        loss = laplacian_loss(preds['phas'].flatten(0, 1), targets['phas'].flatten(0, 1))

        return dict(loss_pha_laplacian=loss)


    def unknown_l1_loss(self, sample_map, preds, targets):
        if torch.sum(sample_map) == 0:
            scale = 0
        else:
            scale = 1 / torch.sum(sample_map)
        # scale = 1

        loss = F.l1_loss(
            preds['phas'].flatten(0, 1) * sample_map,
            targets['phas'].flatten(0, 1) * sample_map,
            reduction='sum'
        ) * scale
        return dict(unknown_l1_loss=loss)


    def known_l1_loss(self, sample_map, preds, targets):
        new_sample_map = torch.zeros_like(sample_map)
        new_sample_map[sample_map==0] = 1
        new_sample_map = new_sample_map.to(sample_map.device)
        
        if torch.sum(new_sample_map) == 0:
            scale = 0
        else:
            scale = 1 / torch.sum(new_sample_map)
        # scale = 1

        loss = F.l1_loss(
            preds['phas'].flatten(0, 1) * new_sample_map,
            targets['phas'].flatten(0, 1) * new_sample_map,
            reduction='sum'
        ) * scale
        return dict(known_l1_loss=loss)
    
    
    def coherence_loss(self, preds, targets):
        # breakpoint()
        pred_pha = preds['phas']
        true_pha = targets['phas']

        loss = F.mse_loss(pred_pha[:, 1:] - pred_pha[:, :-1],
                            true_pha[:, 1:] - true_pha[:, :-1]) * 5
        return dict(coherence_loss=loss)


    def forward(self, sample_map, preds, targets):
        sample_map = sample_map.flatten(0, 1)
        losses = dict()
        for k in self.losses:
            if k=='unknown_l1_loss' or k=='known_l1_loss' or k=='loss_gradient_penalty':
                losses.update(getattr(self, k)(sample_map, preds, targets))
            else:
                # breakpoint()
                losses.update(getattr(self, k)(preds, targets))
        return losses


#-----------------Laplacian Loss-------------------------#
def laplacian_loss(pred, true, max_levels=5):
    kernel = gauss_kernel(device=pred.device, dtype=pred.dtype)
    pred_pyramid = laplacian_pyramid(pred, kernel, max_levels)
    true_pyramid = laplacian_pyramid(true, kernel, max_levels)
    loss = 0
    for level in range(max_levels):
        loss += (2 ** level) * F.l1_loss(pred_pyramid[level], true_pyramid[level])
    return loss / max_levels

def laplacian_pyramid(img, kernel, max_levels):
    current = img
    pyramid = []
    for _ in range(max_levels):
        current = crop_to_even_size(current)
        down = downsample(current, kernel)
        up = upsample(down, kernel)
        diff = current - up
        pyramid.append(diff)
        current = down
    return pyramid

def gauss_kernel(device='cpu', dtype=torch.float32):
    kernel = torch.tensor([[1,  4,  6,  4, 1],
                        [4, 16, 24, 16, 4],
                        [6, 24, 36, 24, 6],
                        [4, 16, 24, 16, 4],
                        [1,  4,  6,  4, 1]], device=device, dtype=dtype)
    kernel /= 256
    kernel = kernel[None, None, :, :]
    return kernel

def gauss_convolution(img, kernel):
    B, C, H, W = img.shape
    img = img.reshape(B * C, 1, H, W)
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    img = F.conv2d(img, kernel)
    img = img.reshape(B, C, H, W)
    return img

def downsample(img, kernel):
    img = gauss_convolution(img, kernel)
    img = img[:, :, ::2, ::2]
    return img

def upsample(img, kernel):
    B, C, H, W = img.shape
    out = torch.zeros((B, C, H * 2, W * 2), device=img.device, dtype=img.dtype)
    out[:, :, ::2, ::2] = img * 4
    out = gauss_convolution(out, kernel)
    return out

def crop_to_even_size(img):
    H, W = img.shape[2:]
    H = H - H % 2
    W = W - W % 2
    return img[:, :, :H, :W]