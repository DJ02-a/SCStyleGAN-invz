import torch
import torch.nn as nn

from lib.blocks import ConvBlock, ResBlock

class Spatial_E(nn.Module):
    def __init__(self, sketch_ch=1, semantic_ch=19):
        super(Spatial_E, self).__init__()

        self.Sketch_E = nn.Sequential(ConvBlock(sketch_ch, 32, stride=1), ConvBlock(32, 64))       # 512 -> 256
        self.Semantic_E = nn.Sequential(ConvBlock(semantic_ch, 32, stride=1), ConvBlock(32, 64))    # 512 -> 256 

        self.Spatial_E = nn.Sequential(
            ConvBlock(128, 256),              # 256 -> 128
            ConvBlock(256, 512),             # 128 -> 64
            ConvBlock(512, 512),             # 64  -> 32
        )

        # 40-Resblock
        self.Resblk_40 = nn.Sequential(*[ResBlock(512, 512) for _ in range(40)])

        # 5-Resblock
        self.Resblk_5 = nn.Sequential(*[ResBlock(512, 512) for _ in range(5)])
        self.to_rgb = ConvBlock(512, 3, stride=1)

    def forward(self, sketch, semantic):
        spatial_sketch = self.Sketch_E(sketch)
        spatial_semantic = self.Semantic_E(semantic)

        mid_spatial = torch.cat((spatial_sketch, spatial_semantic),dim=1)
        mid_spatial_ = self.Spatial_E(mid_spatial)

        spatial_feature = self.Resblk_40(mid_spatial_)
        spatial_img = self.Resblk_5(mid_spatial_)
        spatial_img_ = self.to_rgb(spatial_img)

        return spatial_feature, spatial_img_