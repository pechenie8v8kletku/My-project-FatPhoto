import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)
x = torch.randn(1, 3, 224, 224)
print(model.features)
print(model.classifier)
encoder = model.features
out=encoder(x)
print(out.shape)
