import torch
import torch.nn as nn

class PartEnhancer(nn.Module):
    def __init__(self, in_channels, groups=8, reduction=16, use_fc=False):
        super(PartEnhancer, self).__init__()
        assert in_channels % groups == 0, "in_channels must be divisible by groups"
        self.groups = groups
        self.group_channels = in_channels // groups  # 每组的通道数
        self.use_fc = use_fc

        # **分组空间注意力**
        if use_fc:
            # 1x1 Conv + FC（更强表达能力）
            self.spatial_att = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 4, groups, kernel_size=1)
            )
        else:
            # 只有 1x1 Conv（更高效）
            self.spatial_att = nn.Conv2d(in_channels, groups, kernel_size=1, stride=1, padding=0)

        # **通道注意力**
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global Avg Pooling
        self.channel_fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.channel_fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        
        # **Sigmoid 归一化**
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        g = self.groups
        c_per_group = C // g  # 每组通道数

        # **1️⃣ 分组空间注意力**
        att_map = self.spatial_att(x)  # [B, g, H, W]
        middle_result = att_map
        att_map = self.sigmoid(att_map)  # 归一化到 [0,1]

        # **2️⃣ 适配维度**
        att_map = att_map.view(B, g, 1, H, W)  # [B, g, 1, H, W]
        x_grouped = x.view(B, g, c_per_group, H, W)  # [B, g, c/g, H, W]

        # **3️⃣ 逐组加权**
        out = x_grouped * (1 + att_map)  # [B, g, c/g, H, W] * [B, g, 1, H, W]
        out = out.view(B, C, H, W)  # 变回 [B, C, H, W]

        # **4️⃣ 通道注意力**
        avg_pooled = self.avg_pool(out)  # [B, C, 1, 1]
        channel_weight = self.channel_fc1(avg_pooled)  # 1x1 Conv
        channel_weight = self.relu(channel_weight)  
        channel_weight = self.channel_fc2(channel_weight)  # 1x1 Conv
        channel_weight = self.sigmoid(channel_weight)  # 归一化

        # **5️⃣ 作用通道注意力**
        out = out * (1 + channel_weight)  # [B, C, H, W] * [B, C, 1, 1]

        return out, middle_result
    
    
class GlobalEnhancer(nn.Module):
    def __init__(self, g, method="conv"):
        """
        g: 组的数量
        method: "conv"（默认） 或 "avg_max"
        """
        super(GlobalEnhancer, self).__init__()
        self.method = method

        if method == "conv":
            self.conv = nn.Conv2d(g, 1, kernel_size=1, bias=True)
        elif method == "avg_max":
            self.conv = nn.Conv2d(2, 1, kernel_size=1, bias=True)  # 用于 avg + max 融合
        else:
            raise ValueError(f"Unsupported method: {method}, choose 'conv' or 'avg_max'")

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, feature_map):
        """
        x: [bs, g, h, w] - 组注意力图
        feature_map: [bs, c, h, w] - 原始特征图
        """
        if self.method == "conv":
            attn = self.conv(x)  # [bs, g, h, w] -> [bs, 1, h, w]
        elif self.method == "avg_max":
            avg_pool = torch.mean(x, dim=1, keepdim=True)  # [bs, 1, h, w]
            max_pool, _ = torch.max(x, dim=1, keepdim=True)  # [bs, 1, h, w]
            fusion = torch.cat([avg_pool, max_pool], dim=1)  # [bs, 2, h, w]
            attn = self.conv(fusion)  # [bs, 2, h, w] -> [bs, 1, h, w]

        attn = self.sigmoid(attn)  # 归一化
        return feature_map * (1 + attn)  # 避免过度衰减


class MPE(nn.Module):
    def __init__(self, in_channels, groups=8, reduction=8, use_fc=False, global_method="conv"):
         """
        Multi-Part Enhancer (MPE) 模块，结合局部分组注意力 (PartEnhancer) 和全局增强模块 (GlobalEnhancer)。
        主要用于增强输入特征的表达能力。

        结构：
        1. **PartEnhancer**: 进行分组的空间注意力和通道注意力增强，每个组独立计算注意力权重。
        2. **GlobalEnhancer**: 在组级别融合注意力信息，可选择使用 `conv` 或 `avg_max` 方法。

        参数:
        ----------
        in_channels : int
            输入特征的通道数 (C)。
        groups : int, 默认 8
            分组数 (G)，需要保证 `in_channels % groups == 0`。
        reduction : int, 默认 8
            通道注意力中的降维比率 (r)。
        use_fc : bool, 默认 False
            是否在 `PartEnhancer` 中使用 FC 层来计算空间注意力。
        global_method : str, 默认 "conv"
            选择 `GlobalEnhancer` 的方法：
            - `"conv"`: 使用 `1x1 Conv` 进行组间融合 (默认)。
            - `"avg_max"`: 使用 `avg + max pooling` 计算全局注意力。
        """
        super(MPE, self).__init__()
        self.part_enhancer = PartEnhancer(in_channels, groups=groups, reduction=reduction, use_fc=use_fc)
        self.global_enhancer = GlobalEnhancer(g=groups, method=global_method)

    def forward(self, x):
        out, spatial_att = self.part_enhancer(x)
        out = self.global_enhancer(spatial_att, out)
        return out

    

if __name__=='__main__':
    # **测试**
    B, C, H, W = 2, 64, 32, 32  # 2 个 batch，64 通道，32x32 分辨率
    x = torch.randn(B, C, H, W)  # 随机输入特征图

    model = MPE(C, groups=8,reduction=8)
    out = model(x)

    print(out.shape)  # 期望输出: [B, C, H, W]