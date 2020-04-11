import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        
        #Preparation Layer
        self.preplayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1,  stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) 

        #Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1,  stride=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        ) 
        self.resblock1 = nn.Sequential(
          nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1, bias=False),
          nn.BatchNorm2d(128),
          nn.ReLU(),
          nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1, bias=False),
          nn.BatchNorm2d(128),
          nn.ReLU()
        )

        #Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1,  stride=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ) 

        #Layer 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1,  stride=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.resblock2 = nn.Sequential(
          nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1, bias=False),
          nn.BatchNorm2d(512),
          nn.ReLU(),
          nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1, bias=False),
          nn.BatchNorm2d(512),
          nn.ReLU()
        )
        #MaxPool
        self.pool = nn.MaxPool2d(4,4)

        #Fully Connected layer
        self.linear = nn.Linear(in_features = 512, out_features = 10, bias=False)
    
    def forward(self, x):
      x = self.preplayer(x)
      x = self.layer1(x)
      x += self.resblock1(x)
      x = self.layer2(x)
      x = self.layer3(x)
      x += self.resblock2(x)
      x = self.pool(x)
      x = x.view(x.size(0), -1)
      x = self.linear(x)
      return F.log_softmax(x.view(-1,10), dim=-1)

