## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        self.t1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
        
        self.t2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        
        self.t3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
        
        self.t3b = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        
        self.t4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
        
        self.t4b = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        
        self.t5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                                        
        self.head = nn.Linear(512, 68*2)
            
                
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        out1 = self.t1(x)
        out2 = self.t2(out1)
        out3 = self.t3(out2)
        out4 = self.t3b(out3)
        out5 = self.t4(out4)
        out6 = self.t4b(out5)
        out7 = self.t5(out6)
        
        out8 = self.avgpool(out7)
        out8 = out8.view(out8.size(0), -1)
        
        out9 = self.head(out8)        
                
        return out9
