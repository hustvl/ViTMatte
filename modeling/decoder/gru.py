from torch import nn, Tensor
import torch
from typing import Optional


class ConvGRU(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 bidirectional: Optional[bool] = False):
        super().__init__()
        self.channels = channels
        self.ih = self.zero_module(nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding),
            nn.Sigmoid()
        ))
        self.hh = self.zero_module(nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size, padding=padding),
            nn.Tanh()
        ))
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.ih_rev = self.zero_module(nn.Sequential(
                nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding),
                nn.Sigmoid()
            ))
            self.hh_rev = self.zero_module(nn.Sequential(
                nn.Conv2d(channels * 2, channels, kernel_size, padding=padding),
                nn.Tanh()
            ))


    @staticmethod
    def zero_module(module):
        """
        Zero out the parameters of a module and return it.
        """
        for p in module.parameters():
            p.detach().zero_()
        return module
        

    def forward_single_frame(self, x, h):
        r, z = self.ih(torch.cat([x, h], dim=1)).split(self.channels, dim=1)
        c = self.hh(torch.cat([x, r * h], dim=1))
        h = (1 - z) * h + z * c
        return h, h 
    

    def forward_time_series(self, x, h):
        o = []
        for xt in x.unbind(dim=1):
            ot, h = self.forward_single_frame(xt, h)
            o.append(ot)
        o = torch.stack(o, dim=1)
        return o, h


    def backward_single_frame(self, x, h):
        r, z = self.ih_rev(torch.cat([x, h], dim=1)).split(self.channels, dim=1)
        c = self.hh_rev(torch.cat([x, r * h], dim=1))
        h = (1 - z) * h + z * c
        return h, h 
    

    def backward_time_series(self, x, h):
        o = []
        frames = x.unbind(dim=1) # Read frames in inverse order
        for i in range(len(frames)-1,-1,-1):
            xt = frames[i]
            ot, h = self.backward_single_frame(xt, h)
            o.append(ot)
        o = torch.stack(o, dim=1)
        return o, h


    def forward(self, x, h: Optional[Tensor]):
        if h is None:
            h = torch.zeros((x.size(0), x.size(-3), x.size(-2), x.size(-1)),
                            device=x.device, dtype=x.dtype)
        
        if x.ndim == 5:
            """
            Option 1 

            forward = self.forward_time_series(x,h_for)
            backward = self.backward_time_series(x,h_back)
            return forward, backward
            """
            """
            Option 2

            forward = self.forward_time_series(x,h_for)
            x = torch.flip(x, dim=(1,))
            backward = self.forward_time_series(x,h_back)
            return forward, backward
            """
            if self.bidirectional:
                forward_o, forward_h = self.forward_time_series(x,h)
                backward_o, backward_h = self.backward_time_series(x, forward_h)
                return (forward_o + backward_o) / 2, forward_h
            return self.forward_time_series(x, h)
        else:
            if self.bidirectional:
                forward_o, forward_h = self.forward_single_frame(x,h)
                backward_o, backward_h = self.backward_single_frame(x, forward_h)
                return (forward_o + backward_o) / 2, forward_h
            return self.forward_single_frame(x, h)