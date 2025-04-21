import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

class DCWF(nn.Module):
    """Deformable Convolutional With Fusion module"""
    
    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, padding=1):
        super().__init__()
        out_channels = out_channels or in_channels
        
        # 可变形卷积参数
        self.offset_conv = nn.Conv2d(
            in_channels, 
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        
        # 可变形卷积权重
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))
        
        # 特征融合部分
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 1),
            nn.SiLU(),
            nn.Conv2d(out_channels // 2, out_channels, 1)
        )
        
        # 初始化参数
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def forward(self, x):
        # 输入可以是单个张量或张量列表
        if isinstance(x, (list, tuple)):
            x = torch.cat(x, dim=1)
        
        # 计算偏移量
        offset = self.offset_conv(x)
        
        # 可变形卷积
        deform_feat = deform_conv2d(
            x, 
            offset, 
            self.weight, 
            self.bias, 
            stride=self.stride,
            padding=self.padding,
        )
        
        # 特征融合
        fusion_feat = self.fusion_conv(x)
        
        # 融合结果
        return deform_feat + fusion_feat
from ultralytics.nn.modules import Conv, C2f, Bottleneck

def replace_concat_with_dcwf(model):
    """
    递归替换模型中的所有Concat操作为DCWF模块
    """
    for name, module in model.named_children():
        if isinstance(module, nn.ModuleList):
            for i, m in enumerate(module):
                if isinstance(m, (C2f, Bottleneck)):
                    replace_concat_in_c2f(m)
        elif hasattr(module, 'forward_concat'):  # 检测是否包含concat操作
            # 获取输入通道数(假设是两个相同大小的特征图拼接)
            in_channels = module.forward_concat[0].in_channels * 2
            setattr(model, name, DCWF(in_channels))
        else:
            # 递归处理子模块
            replace_concat_with_dcwf(module)

def replace_concat_in_c2f(c2f_module):
    """
    替换C2f模块内部的concat操作为DCWF
    """
    if not hasattr(c2f_module, 'cv2'):
        return
    
    # 计算DCWF的输入通道数
    n = len(c2f_module.m)
    in_channels = (2 + n) * c2f_module.c
    
    # 替换最后的concat+conv为DCWF
    c2f_module.cv2 = DCWF(in_channels, c2f_module.cv2.conv.out_channels)
class C2f_DCWF(nn.Module):
    """C2f module with DCWF instead of concat"""
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = DCWF((2 + n) * self.c, c2)  # 使用DCWF替换原来的concat+conv
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        
    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(y)  # DCWF可以直接处理列表输入