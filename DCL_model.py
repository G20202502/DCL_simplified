import torch, torchvision
from torch import nn
from torchvision import models

class DCL_Network(nn.Module):
    def __init__(self, args, is_train: bool):
        super(DCL_Network, self).__init__()
        self.is_train = is_train
        self.num_class = 200
        
        self.backbone = getattr(models, 'resnet50')()
        if args.resnet50_path != None:
            self.backbone.load_state_dict(torch.load(args.resnet50_path))
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.avgpool = nn.AdaptiveAvgPool2d(output_size = 1)
        self.cls_linear = nn.Linear(in_features = 2048, out_features = self.num_class, bias = False)
        if is_train:
            self.adv_linear = nn.Linear(in_features = 2048, out_features = 2*self.num_class, bias = False)
            self.con_conv = nn.Conv2d(in_channels = 2048, out_channels = 1, kernel_size = 1, stride = 1, padding = 0, bias = True)
            self.con_avgpool = nn.AvgPool2d(kernel_size = 2, stride = 2)
    
    def forward(self, x):
        x = self.backbone(x)
        
        if self.is_train:
            con_out = self.con_conv(x)
            con_out = self.con_avgpool(con_out)
            con_out = torch.tanh(con_out)
            con_out = con_out.view(con_out.size(0), -1)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.is_train:
            adv_out = self.adv_linear(x)

        x = self.cls_linear(x)

        return x, adv_out, con_out