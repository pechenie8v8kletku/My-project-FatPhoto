
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)
encoder = model.features
class FatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=encoder
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.Batch1=nn.LayerNorm(1280*4)
        self.conv1=nn.SiLU(inplace=True)
        self.fc2=nn.Linear(in_features=1280*4,out_features=500,bias=True)
        self.Batch2 = nn.LayerNorm(500)
        self.conv2 = nn.SiLU(inplace=True)
        self.fc3 = nn.Linear(in_features=500, out_features=1, bias=True)


    def forward(self, x):


        B,N, C, H, W = x.shape
        x=x.reshape(B*N,C,H,W)
        x=self.encoder(x)


        x=self.avg(x)
        _,C,H,W=x.shape
        x=x.reshape(B,N*C,H,W)
        x = x.flatten(1)

        x=self.Batch1(x)

        x=self.conv1(x)
        x=self.fc2(x)
        x=self.Batch2(x)
        x=self.conv2(x)
        x=self.fc3(x)
        #print(x.shape)


        return x
'''
x = torch.randn(10,4, 3, 224, 224)
model=FatModel()
z=model(x)
print(z)
'''