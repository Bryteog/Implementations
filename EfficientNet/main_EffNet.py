import torch
import torch.nn as nn
from math import ceil


stages = [
    # [Channels(c), Layers(l), Kernel(k), Stride(s), Expansion(exp)]
    [32, 1, 3, 2, 1],
    [16, 1, 3, 1, 1],
    [24, 2, 3, 2, 6],
    [40, 2, 5, 2, 6],
    [80, 3, 3, 2, 6],
    [112, 3, 5, 1, 6],
    [192, 4, 5, 2, 6],
    [320, 1, 3, 1, 6],
    [1280, 1, 1, 1, 0]
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


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups = 1):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups = groups)
        self.b_norm = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()
        
    def forward(self, x):
        return self.silu(self.b_norm(self.cnn(x)))
    

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduce_dim):
        super(SqueezeExcitation, self).__init__()
        self.SqEx = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduce_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduce_dim, in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return x * self.SqEx(x)
    
    
class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expand_ratio, reduction = 4, survival_prob = 0.8, bias = False):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = 0.8
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduce_dim = int(in_channels / reduction)
        
        if self.expand:
            self.expand_conv = CNNBlock(in_channels, hidden_dim, kernel_size = 3, stride = 1, padding = 1)
            
        self.conv = nn.Sequential(
            CNNBlock(hidden_dim, hidden_dim, kernel_size, stride, padding, groups = hidden_dim),
            SqueezeExcitation(hidden_dim, reduce_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias = False),
            nn.BatchNorm2d(out_channels),
        )
        
    def stochastic_depth(self, x):
        if not self.training:
            return x
        
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device = x.device) < self.survival_prob
        return torch.div(x, self.survival_prob) * binary_tensor
    
    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs
        
        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)
        

class EfficientNet(nn.Module):
    def __init__(self, version, num_classes):
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = ceil(1280 * width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes),
        )
            
    def calculate_factors(self, version, alpha = 1.2, beta = 1.1):
        phi, res, drop_rate = phis[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return width_factor, depth_factor, drop_rate
    
    def create_features(self, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        features = [CNNBlock(3, channels, 3, stride = 2, padding = 1)]
        in_channels = channels
        
        for expand_ratio, channels, repeats, stride, kernel_size in stages:
            out_channels = 4 * ceil(int(channels * width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)
            
            for layer in range(layers_repeats):
                features.append(InvertedResidualBlock(
                    in_channels, out_channels, expand_ratio = expand_ratio, stride = stride if layer == 0 else 1, kernel_size = kernel_size, padding = kernel_size // 2
                ))
                in_channels = out_channels
                
        features.append(
            CNNBlock(in_channels, last_channels, kernel_size = 1 ,stride = 1, padding = 0)
        )
        
        return nn.Sequential(*features)
    
    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1))
    
def test():
    device = "cpu"
    version = "B0"
    phi, res, drop_rate = phis[version]
    num_examples, num_classes = 4, 10
    x = torch.randn((num_examples, 3, res, res)).to(device)
    model = EfficientNet(version = version, num_classes = num_classes).to(device)
    
    print(model(x).shape)
    
test()