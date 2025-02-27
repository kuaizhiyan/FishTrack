# 修改后的PartEnhancer类
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch.nn import init

class PartEnhancer(BaseModule):
    def __init__(self, in_channels, groups=8, reduction=16, use_fc=True):
        super().__init__()
        self.groups = groups
        self.group_channels = in_channels // groups
        self.use_fc = use_fc

        # 改进的空间注意力 (增加3x3卷积)
        if use_fc:
            self.spatial_att = nn.Sequential(
                nn.Conv2d(in_channels, in_channels//4, 1),
                nn.BatchNorm2d(in_channels//4),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels//4, groups, 3, padding=1)  # 3x3卷积
            )
        else:
            self.spatial_att = nn.Sequential(
                nn.Conv2d(in_channels, groups, 3, padding=1),
                nn.BatchNorm2d(groups)
            )

        # 通道注意力改进 (添加BN层)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//reduction, 1, bias=False),
            nn.BatchNorm2d(in_channels//reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        g = self.groups
        
        # 通道注意力优先
        channel_weight = self.channel_att(x)  # [B,C,1,1]
        x = x * channel_weight  # 先增强通道维度

        # 空间注意力
        spatial_att = self.spatial_att(x)  # [B,g,H,W]
        spatial_weight = torch.sigmoid(spatial_att)
        
        # 分组处理
        x_grouped = x.view(B, g, -1, H, W)  # [B,g,C/g,H,W]
        out = x_grouped * spatial_weight.unsqueeze(2)  # [B,g,C/g,H,W]
        out = out.view(B, C, H, W)
        
        return out + x, spatial_att  # 残差连接

class GlobalEnhancer(BaseModule):
    def __init__(self, g, method="avg_max"):
        super().__init__()
        self.method = method
        
        if method == "conv":
            self.conv = nn.Sequential(
                nn.Conv2d(g, g, 3, padding=1, groups=g, bias=False),
                nn.BatchNorm2d(g),
                nn.ReLU(),
                nn.Conv2d(g, 1, 1)
            )
        elif method == "avg_max":
            self.conv = nn.Sequential(
                nn.Conv2d(2, 1, 3, padding=1),
                nn.BatchNorm2d(1)
            )
        
        # 初始化最后层权重为0
        nn.init.constant_(self.conv[-1].weight, 0)
        nn.init.constant_(self.conv[-1].bias, -2.0)  # 初始输出偏负值

    def forward(self, x, feature_map):
        if self.method == "avg_max":
            avg_pool = x.mean(dim=1, keepdim=True)
            max_pool = x.amax(dim=1, keepdim=True)
            x = torch.cat([avg_pool, max_pool], dim=1)
        
        attn = torch.sigmoid(self.conv(x))  # [B,1,H,W]
        return feature_map * (1 + attn)  # 增强重要区域

class GroupInteraction(nn.Module):
    def __init__(self, channels, groups):
        super().__init__()
        self.groups = groups
        self.dwc = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//8, 1),
            nn.ReLU(),
            nn.Conv2d(channels//8, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        identity = x
        x = self.dwc(x)
        gate = self.gate(x)
        return identity + x * gate

class MPE_ds(BaseModule):
    def __init__(self, in_channels, groups=4, reduction=8, use_fc=True, global_method="avg_max"):
        super().__init__()
        self.part_enhancer = PartEnhancer(in_channels, groups, reduction, use_fc)
        self.global_enhancer = GlobalEnhancer(groups, global_method)
        self.group_interaction = GroupInteraction(in_channels, groups)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out, spatial_att = self.part_enhancer(x)
        out = self.global_enhancer(spatial_att, out)
        # out = self.group_interaction(out)
        return out