import torch
import torch.nn as nn

class MPE(nn.Module):
    def __init__(self, in_channels, g=8):
        """
        Args:
            in_channels: int 输入通道维度, 同等维度输出
            g: int=8,分组的个数
        
        
        """
        super(MPE, self).__init__()
        self.in_channels = in_channels
        self.g = g

        # 1x1卷积层，用于生成空间权重
        self.conv1x1 = nn.Conv2d(in_channels // g, 1, kernel_size=1, stride=1, padding=0)

        # 用于生成通道注意力
        self.channel_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化

        # 通道降维层
        self.channel_fc1 = nn.Conv2d(in_channels // g, in_channels // 16, kernel_size=1)
        # 用于控制降维后的通道数
        self.channel_fc2 = nn.Conv2d(in_channels // 16, in_channels // g, kernel_size=1)

        # 全局通道 spatial fc
        self.spatial_fc1 = nn.Conv2d(g, g, kernel_size=1)
        self.spatial_fc2 = nn.Conv2d(g, 1, kernel_size=1)
        
    def forward(self, x):
        # 输入x: [bs, c, h, w]
        bs, c, h, w = x.shape

        # 按组划分输入
        x = x.view(bs, self.g, c // self.g, h, w)

        # 局部通道
        # 通道注意力：全局平均池化
        channel_att = self.channel_pool(x.view(bs * self.g, c // self.g, h, w))  # [bs * g, c/g, 1, 1]
        channel_att = self.channel_fc1(channel_att)  # [bs * g, c/g, 1, 1]
        channel_att = self.channel_fc2(channel_att).view(bs, self.g, c // self.g)  # [bs, g, c/g]
        channel_att = torch.sigmoid(channel_att)    # [bs, g, c/g]

        # 空间注意力：通过1x1卷积生成
        spatial_att = self.conv1x1(x.view(bs * self.g, c // self.g, h, w))  # [bs * g, 1, h, w]
        local_spatial_att = torch.sigmoid(spatial_att.view(bs, self.g, 1, h, w))  # [bs, g, 1, h, w] 局部分组内空间注意力
        
        # 全局通道,搜集 局部通道中的 spatial_att
        global_spatial_att = self.spatial_fc1(spatial_att.view(bs, self.g, h, w)) # [bs, g, h, w]
        global_spatial_att = self.spatial_fc2(global_spatial_att)   # [bs, 1, h, w]
        global_spatial_att = torch.sigmoid(global_spatial_att).unsqueeze(1)      # [bs, 1, 1, h, w]

        # 合并通道和空间权重
        out = x * channel_att.view(bs, self.g, c // self.g, 1, 1) * local_spatial_att * global_spatial_att # [bs, g, c/g, h, w]

        # 恢复原来的形状
        out = out.view(bs, c, h, w) # [bs, c, h, w]

        return out

# 测试一下这个实现
x = torch.randn(4, 64, 32, 32)  # 输入张量，batch size=4，通道数=64，大小为32x32
attn = MPE(64, g=8)
output = attn(x)
print(output.shape)  # 期望的输出大小应该是 [4, 64, 32, 32]
