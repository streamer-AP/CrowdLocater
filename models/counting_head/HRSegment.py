import torch
import torch.nn as nn
import torch.nn.functional as F


class HRSegMent(nn.Module):
    def __init__(self):
        super(HRSegMent, self).__init__()
        self.conv1 =nn.Sequential(
            nn.Conv2d(128,128,1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256,128,1),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(512,128,1),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(1024,128,1),
            nn.ReLU(inplace=True),
        )
        self.out1 = nn.Sequential(            
            nn.Conv2d(512, 256, 3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.ReLU(inplace=True),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x1, x2 , x3, x4 = x
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4= self.conv4(x4)
        
        x4 = F.interpolate(x4, scale_factor=8)
        x3=  F.interpolate(x3, scale_factor=4)
        x2 = F.interpolate(x2, scale_factor=2)

        z=torch.cat([x1,x2,x3,x4],dim=1)

        out1 = self.out1(z)
        out_dict = {}
        out_dict["predict_counting_map"] = out1
        return out_dict


def build_counting_head(args):

    return HRSegMent()

if __name__ == "__main__":
    model = HRSegMent()
    x1 = torch.randn(1,96,64,64)
    x2 = torch.randn(1,192,32,32)
    x3 = torch.randn(1,384,16,16)
    x4 = torch.randn(1,768,8,8)
    x = [x1,x2,x3,x4]
    out = model(x)
    print(out["predict_counting_map"].shape)
