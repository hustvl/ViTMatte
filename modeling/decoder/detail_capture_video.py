import torch
from torch import nn
from torch.nn import functional as F
from .gru import ConvGRU


class Basic_Conv3x3(nn.Module):
    """
    Basic convolution layers including: Conv3x3, BatchNorm2d, ReLU layers.
    """
    def __init__(
        self,
        in_chans,
        out_chans,
        stride=2,
        padding=1,
        recurrent=False
    ):
        super().__init__()
        if recurrent:
            self.gru = ConvGRU(in_chans)
        self.conv = nn.Conv2d(in_chans, out_chans, 3, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(True)
        self.recurrent = recurrent


    def forward(self, x, h = None):
        if self.recurrent:
            y, h = self.gru(x, h)  # Recurrent network initialized at zero
            x = x + y  # Residual connection
            B, T = x.shape[:2]
            x = x.flatten(0, 1)
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            x = x.unflatten(0, (B, T))
            return x, h
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            return x


class ConvStream(nn.Module):
    """
    Simple ConvStream containing a series of basic conv3x3 layers to extract detail features.
    """
    def __init__(
        self,
        in_chans = 4,
        out_chans = [48, 96, 192],
        recurrent = False
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        
        self.conv_chans = out_chans.copy()
        self.conv_chans.insert(0, in_chans)
        
        for i in range(len(self.conv_chans)-1):
            in_chan_ = self.conv_chans[i]
            out_chan_ = self.conv_chans[i+1]
            self.convs.append(
                Basic_Conv3x3(in_chan_, out_chan_, recurrent=recurrent)
            )
        self.recurrent = recurrent

    
    def forward(self, x, rec=None):
        if self.recurrent:
            if rec is None:
                rec = {'D'+str(i+1): None for i in range(len(self.convs))}
            out_dict = {'D0': x}
            for i in range(len(self.convs)):
                name_ = 'D'+str(i+1)
                x, rec_ = self.convs[i](x, rec[name_])
                out_dict[name_] = x
                rec[name_] = rec_
            
            return out_dict, rec
        else:
            out_dict = {'D0': x}
            for i in range(len(self.convs)):
                x = self.convs[i](x)
                name_ = 'D'+str(i+1)
                out_dict[name_] = x
            
            return out_dict


class Fusion_Block(nn.Module):
    """
    Simple fusion block to fuse feature from ConvStream and Plain Vision Transformer.
    """
    def __init__(
        self,
        in_chans,
        out_chans,
        recurrent = False
    ):
        super().__init__()
        self.conv = Basic_Conv3x3(in_chans, out_chans, stride=1, padding=1, recurrent = recurrent)
        self.recurrent = recurrent


    def forward(self, x, D, rec=None):
        if self.recurrent:
            B, T = x.shape[:2]
            x = x.flatten(0, 1)
            D = D.flatten(0, 1)
            F_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            out = torch.cat([D, F_up], dim=1)
            out = out.unflatten(0, (B, T))
            out, rec = self.conv(out, rec)
            return out, rec
        else:
            F_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            out = torch.cat([D, F_up], dim=1)
            out = self.conv(out)
            return out    


class Matting_Head(nn.Module):
    """
    Simple Matting Head, containing only conv3x3 and conv1x1 layers.
    """
    def __init__(
        self,
        in_chans = 32,
        mid_chans = 16,
    ):
        super().__init__()
        self.matting_convs = nn.Sequential(
            nn.Conv2d(in_chans, mid_chans, 3, 1, 1),
            nn.BatchNorm2d(mid_chans),
            nn.ReLU(True),
            nn.Conv2d(mid_chans, 1, 1, 1, 0)
            )


    def forward(self, x):
        x = self.matting_convs(x)

        return x


class Detail_Capture(nn.Module):
    """
    Simple and Lightweight Detail Capture Module for ViT Matting.
    """
    def __init__(
        self,
        in_chans = 384,
        img_chans=4,
        convstream_out = [48, 96, 192],
        fusion_out = [256, 128, 64, 32],
        recurrent = False
    ):
        super().__init__()
        assert len(fusion_out) == len(convstream_out) + 1

        self.convstream = ConvStream(in_chans = img_chans, recurrent = recurrent)
        self.conv_chans = self.convstream.conv_chans

        self.fusion_blks = nn.ModuleList()
        self.fus_channs = fusion_out.copy()
        self.fus_channs.insert(0, in_chans)
        for i in range(len(self.fus_channs)-1):
            self.fusion_blks.append(
                Fusion_Block(
                    in_chans = self.fus_channs[i] + self.conv_chans[-(i+1)],
                    out_chans = self.fus_channs[i+1],
                    recurrent = recurrent
                )
            )

        self.matting_head = Matting_Head(
            in_chans = fusion_out[-1],
        )
        
        self.recurrent = recurrent


    def forward(self, features, images, rec=None):
        if self.recurrent:
            if rec is None:
                rec = {'D'+str(len(self.fusion_blks)-i-1): None for i in range(len(self.fusion_blks))}
                rec['convstream'] = None
            detail_features, rec_conv = self.convstream(images, rec['convstream'])
            rec['convstream'] = rec_conv
            for i in range(len(self.fusion_blks)):
                d_name_ = 'D'+str(len(self.fusion_blks)-i-1)
                features, rec_fus = self.fusion_blks[i](features, detail_features[d_name_], rec[d_name_])
                rec[d_name_] = rec_fus

            features = features.flatten(0, 1)
            phas = torch.sigmoid(self.matting_head(features))
            return {'phas': phas}, rec
        else:
            detail_features = self.convstream(images)
            for i in range(len(self.fusion_blks)):
                d_name_ = 'D'+str(len(self.fusion_blks)-i-1)
                features = self.fusion_blks[i](features, detail_features[d_name_])
            
            phas = torch.sigmoid(self.matting_head(features))
            return {'phas': phas}