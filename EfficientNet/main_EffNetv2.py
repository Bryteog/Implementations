import torch
import torch.nn as nn
from torch import Tensor

# Conv Block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups = 1, activation = True, bias = False):
        super().__init__()
        """ if k = 1 -> p = 0, k = 3 -> p = 1, k = 5 -> p = 2"""
        padding = kernel_size // 2
        self.c1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = bias, groups = groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU() if activation else nn.Identity()
        
    def forward(self, x):
        x = self.c1(x)
        x = self.bn(x)
        x = self.silu(x)
        return x
    
    
# Squeeze and Excitation Block
class Sq_ExBlock(nn.Module):
    def __init__(self, in_channels, r = 24):
        super().__init__()
        C = in_channels
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fully_con1 = nn.Linear(C, C // r, bias = False)
        self.fully_con2 = nn.Linear(C // r, C, bias = False)
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """shape : [N, C, H, W]"""
        f = self.globalpool(x)
        f = torch.flatten(f, 1)
        f = self.silu(self.fully_con1(f))
        f = self.sigmoid(self.fully_con2(f))
        f = f[:, :, None, None]
        """f shape = [N, C, 1, 1]"""
        scale = x * f
        return scale
    

# Stochastic Depth for dropout    
class StochasticDepth(nn.Module):
    def __init__(self, p : float = 0.5):
        super().__init__()
        self.p = p
        
    def forward(self, x:Tensor) -> Tensor:
        mask_shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
        mask = torch.empty(mask_shape).bernoulli_(self.p) / self.p
        
        if self.training:
            x = mask * x
        return x
    
    
# MBConv Block
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion, r):
        super().__init__()
        expansion_c = in_channels * expansion
        self.add = in_channels == out_channels and stride == 1
        self.mbconv1 = ConvBlock(in_channels, expansion_c, 1, 1,) if expansion > 1 else nn.Identity()
        self.mbconv2 = ConvBlock(expansion_c, expansion_c, kernel_size, stride, expansion_c)
        self.Sqz_Exp = Sq_ExBlock(expansion_c, r)
        self.mbconv3 = ConvBlock(expansion_c, out_channels, 1, 1, activation = False)
        self.stochastic_depth = StochasticDepth()
        
    def forward(self, x):
        f = self.mbconv1(x)
        f = self.mbconv2(f)
        f = self.Sqz_Exp(f)
        f = self.mbconv3(f)
        
        if self.add:
            f = x + f
        f = self.stochastic_depth(f)
        return f

    
# Classifier
class Classifier(nn.Module):
    def __init__(self, in_channels, classes, p):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.full = nn.Linear(in_channels, classes)
        self.dropout = nn.Dropout(p)
        
    def forward(self, x):
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        return self.full(x)
 
    
# EfficientNet Model Architectures
class EfficientNet(nn.Module):
    def __init__(self, stages, phis, in_channels = 3, classes = 1000):
        super().__init__()
        """Parameters"""
        phi, resolution, p = phis
        self.calc_coeff(phi)
        
        """Network"""
        self.network = nn.ModuleList([])
        self.channels = []
        
        """Stage 1 Conv3x3 """
        f, c, l, k, s, expansion = stages[0]
        self.add_layer(3, f, c, l, k, s)
        
        """Stages 2-8 MBConv"""
        for i in range(1, len(stages) - 1):
            if i == 1:
                r = 4
            else:
                r = 24
                
            f, c, l, k, s, expansion = stages[i]
            self.add_layer(self.channels[-1], f, c, l, k, s, expansion, r)
        
        """Conv1x1 + Classifier"""
        f, c, l, k, s, expansion = stages[-1]
        self.add_layer(self.channels[-1], f, c, l, k, s)
        self.network.append(Classifier(self.channels[-1], classes, p))
        
        
    """def forward(self, x):
        for F in self.network:
            x = F(x)
            return x"""
        
    def forward(self, x):
        # To display the Stages and shapes
        i = 1
        for F in self.network:
            in_feat, h, w = x.shape[1:]
            
            x = F(x)
            if in_feat != x.shape[1] and i < 10:
                print("Stage {} -> ".format(i), [x.shape[1], h, w])
                i += 1
        return x
    
    def add_layer(self, in_channels, f, c, l, k, s, *args):
        c, l = self.update_feat(c, l)
        if l == 1:
            self.network.append(f(in_channels, c, k, s, *args))
        else:
            """First layer with stride 1"""
            self.network.append(f(in_channels, c, k, 1, *args))
            """Another layer with stride 1"""
            for _ in range(1-2):
                self.network.append(f(c, c, k, 1, *args))
                
                """Final layer with stride 1 or 2"""
            self.network.append(f(c, c, k, s, *args))
            
        self.channels.append(c)
        
    def calc_coeff(self, phi, alpha = 1.2, beta = 1.1):
        self.d = alpha ** phi
        self.w = beta ** phi
        
    def update_feat(self, c, l):
        width = int(c * self.w)
        depth = int(l * self.d)
        return width, depth
    


stages = [
    # [Operator(f), Channels(c), Layers(l), Kernel(k), Stride(s), Expansion(exp)]
    [ConvBlock, 32, 1, 3, 2, 1],
    [MBConv, 16, 1, 3, 1, 1],
    [MBConv, 24, 2, 3, 2, 6],
    [MBConv, 40, 2, 5, 2, 6],
    [MBConv, 80, 3, 3, 2, 6],
    [MBConv, 112, 3, 5, 1, 6],
    [MBConv, 192, 4, 5, 2, 6],
    [MBConv, 320, 1, 3, 1, 6],
    [ConvBlock, 1280, 1, 1, 1, 0]
]

phis = {
    # BN : (phi, resolution, dropout)
    "B0" : (0, 224, 0.2),
    "B1" : (0.5, 240, 0.2),
    "B2" : (1, 260, 0.3),
    "B3" : (2, 300, 0.3),
    "B4" : (3, 380, 0.4),
    "B5" : (4, 456, 0.4),
    "B6" : (5, 528, 0.5),
    "B7" : (6, 600, 0.5),
}

ver = "B0"
_, res, _ = phis[ver]
e = EfficientNet(stages, phis[ver])
print("Output: ", e(torch.rand(1, 3, res, res)).shape)