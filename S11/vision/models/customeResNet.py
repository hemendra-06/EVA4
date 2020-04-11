import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class BasicBlock(nn.Module):
  expansion = 1
  def __init__(self, in_planes, planes, stride=1):
      super(BasicBlock, self).__init__()
      self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
      self.bn1 = nn.BatchNorm2d(planes)
      self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
      self.bn2 = nn.BatchNorm2d(planes)

      self.shortcut = nn.Sequential()
      if stride != 1 or in_planes != self.expansion*planes:
          self.shortcut = nn.Sequential(
              nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=1, padding=1, bias=False),
              nn.BatchNorm2d(self.expansion*planes)
          )

  def forward(self, x):
      out = F.relu(self.bn1(self.conv1(x)))
      out = F.relu((self.bn2(self.conv2(out))))
      out += x
      return out

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
        self.pool = nn.MaxPool2d(4,4)
        self.linear = nn.Linear(512, 10)
    
    def forward(self, x):
      x = self.preplayer(x)
      x = self.layer1(x)
      x += self.resblock1(x)
      x = self.layer2(x)
      x = self.layer3(x)
      x += self.resblock2(x)
      x = self.pool(x)
      x = self.linear(x)
      # x =F.log_softmax(x, dim=-1)
      return x

