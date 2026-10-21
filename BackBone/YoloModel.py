import torch
import torch.nn as nn

class CnnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, out_padding):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size, stride, out_padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.cnn(x)))

class YoloBackBone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            CnnBlock(3, 32, 3, 1,1),
            nn.MaxPool2d(kernel_size=(2,2)),
            CnnBlock(3, 32, 3, 1,1),
            nn.MaxPool2d(kernel_size=(2,2)),
            CnnBlock(3, 32, 3, 1,1),
            nn.MaxPool2d(kernel_size=(2,2)),
        )
    def forward(self, x):
        return self.layers(x)

class YoloHead(nn.Module):
    def __init__(self, grid_size, num_classes, num_anchors):
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.detector = nn.Conv2d(128, num_anchors*(5 + num_classes), kernel_size=1)

    def forward(self, x):
        return self.detector(x).permute(0, 2, 3, 1).contiguous()

# assembling the yolo model now

class Yolo(nn.Module):
    def __init__(self, grid_size=7, num_classes=20, num_anchors=3):
        super().__init__()
        self.backBone = YoloBackBone()
        self.head = YoloHead(grid_size, num_classes, num_anchors)

    def forward(self,x):
        featureMap = self.backBone(x)
        predictions = self.head(featureMap)
        return predictions

    
'''
example usage of mdoel creation
model = YOLO(grid_size, num_classes, num_anchors)
print(model)
'''
        
        


        
