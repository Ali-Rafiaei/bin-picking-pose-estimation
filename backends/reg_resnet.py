import torch
import torch.nn as nn
from torchvision.models import resnet50

class ResNet50(nn.Module):
    def __init__(self, num_keypoints=4, input_include_depth=False):
        super(ResNet50, self).__init__()
        self.resnet50 = resnet50(weights=None)
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-2])

        if input_include_depth:
            old_conv = self.resnet50[0]
            new_conv = nn.Conv2d(4, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride, padding=old_conv.padding, bias=False)
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                new_conv.weight[:, 3:] = 0  # depth channel init to zero
            self.resnet50[0] = new_conv

        self.regressor = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, num_keypoints, kernel_size=3, padding=1)
        )


    def forward(self, x):
        im_size = x.size()
        x = self.resnet50(x)
        x = self.regressor(x)
        x = nn.functional.interpolate(x, size=(im_size[2], im_size[3]), mode='bilinear',
                                                     align_corners=False)


        return x