import torch
import torch.nn as nn

# Vgg-16 Architecture
# Input(1x3x224x224) -> Conv(64, (3x3) s=1, p=1) -> Conv(64, (3x3) s=1, p=1) -> AvgPool(s=2, p=2) -> Conv(128, (3x3) s=1, p=1) -> Conv(128, (3x3) s=1, p=1)
# -> AvgPool(s=2, p=2) -> Conv(256, (3x3) s=1, p=1) -> Conv(256, (3x3) s=1, p=1) -> Conv(256, (3x3) s=1, p=1) -> AvgPool(s=2, p=2) -> Conv(512, (3x3) s=1, p=1)
# -> Conv(512, (3x3) s=1, p=1) -> Conv(512, (3x3) s=1, p=1) -> AvgPool(s=2, p=2) -> Conv(512, (3x3) s=1, p=1) -> Conv(512, (3x3) s=1, p=1) -> Conv(512, (3x3) s=1, p=1)
# -> AvgPool(s=2, p=2) -> Linear 4096 -> Linear 4096 -> Linear 1000 -> Softmax

class VGGNet(nn.Module):
    def __init__(self, in_channels = 3 , num_classes = 1000):
        super(VGGNet, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv6 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv7 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv8 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.linear1 = nn.Linear(512 * 7 * 7, 4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, num_classes)
        
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv6(x))
        x = self.pool(x)
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv8(x))
        x = self.pool(x)
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv8(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        #x = nn.Dropout(0.5)
        x = self.relu(self.linear2(x))
        #x = nn.Dropout(0.5)
        x = self.linear3(x)
        
        return x
    
x = torch.randn(1, 3, 224, 224)
model = VGGNet(in_channels = 3, num_classes = 1000)
print(model(x).shape)