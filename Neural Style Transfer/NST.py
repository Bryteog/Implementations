import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

model = models.vgg19(weights = True).features

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ['0', '5', '10', '19', '28 ']
        self.model = models.vgg19(pretrained = True).features[:29]
    
    def forward(self, x):
        features = []
        
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features
    
def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)

device = "cuda" if torch.cuda.is_available() else "cpu"
image_size = 512

loader = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
)

content_image = load_image("/content/sample_data/flower.jpg")
style_image = load_image("/content/sample_data/style2.jpg")

   
#generated_image = torch.randn(content_image.shape, device = device, requires_grad = True)
generated_image = content_image.clone().requires_grad_(True)
model = VGG().to(device).eval()

# Hyperparameter
total_steps = 6000
learning_rate = 0.001
alpha = 1
beta = 0.01
optimizer = optim.Adam([generated_image], lr = learning_rate)

for step in range(total_steps):
    generated_features = model(generated_image)
    content_features = model(content_image)
    style_features = model(style_image)
    
    style_loss = content_loss = 0
    
    for gen_feature, content_feature, style_feature in zip(
        generated_features, content_features, style_features
    ):
        batch_size, channel, height, width = gen_feature.shape
        content_loss += torch.mean((gen_feature - content_feature) ** 2)
        
        # Compute Gram Matrix
        G = gen_feature.view(channel, height * width).mm(
            gen_feature.view(channel, height * width).t()
        )
        
        A = style_feature.view(channel, height * width).mm(
            style_feature.view(channel, height * width).t()
        )
        
        style_loss += torch.mean((G - A) ** 2)
        
    total_loss = alpha * content_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        print(total_loss)
        save_image(generated_image, "/content/sample_data/generated_image.png")