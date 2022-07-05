import math
import torch.nn as nn


__all__ = ['VGG', 'cvgg19_6', 'cvgg19_5', 'cvgg19_4', 'cvgg19_3', 'cvgg19_2', 'cvgg19_1', 'cvgg11_1', 'cvgg11_2', 'cvgg11_3', 'cvgg19_8', 'cvgg16_3', 'cvgg16_3_plus_hrank']


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),  # 112 or 512
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A1': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 'M'],
    'A2': [64, 'M', 128, 'M', 256, 256, 'M', 512, 'M', 512, 'M'],
    'A3': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'A_2': [64, 'M', 128, 'M', 256, 256, 256, 256, 'M', 512, 'M', 512, 'M'],
    'D_0': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 'M', 512, 'M'],
    'D_1': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 112, 112, 'M', 112, 'M'],
    'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 'M', 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 'M', 512, 'M'],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 'M'],
    'F': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

###  fig.8 ###
def cvgg19_1(num_classes):
    return VGG(make_layers(cfg['F'], batch_norm=True), num_classes=num_classes)

def cvgg19_2(num_classes):
    return VGG(make_layers(cfg['E'], batch_norm=True), num_classes=num_classes)

def cvgg19_3(num_classes):
    return VGG(make_layers(cfg['D'], batch_norm=True), num_classes=num_classes)

def cvgg19_4(num_classes):
    return VGG(make_layers(cfg['C'], batch_norm=True), num_classes=num_classes)

def cvgg19_5(num_classes):
    return VGG(make_layers(cfg['B'], batch_norm=True), num_classes=num_classes)

def cvgg19_6(num_classes):
    return VGG(make_layers(cfg['A'], batch_norm=True), num_classes=num_classes)


###  fig.9 ###
def cvgg11_1(num_classes):
    return VGG(make_layers(cfg['A1'], batch_norm=True), num_classes=num_classes)

def cvgg11_2(num_classes):
    return VGG(make_layers(cfg['A2'], batch_norm=True), num_classes=num_classes)

def cvgg11_3(num_classes):
    return VGG(make_layers(cfg['A3'], batch_norm=True), num_classes=num_classes)


###  tab.1 ###
def cvgg19_8(num_classes):  # for pruning
    return VGG(make_layers(cfg['A_2'], batch_norm=True), num_classes=num_classes)

def cvgg16_3(num_classes):  # for pruning
    return VGG(make_layers(cfg['D_0'], batch_norm=True), num_classes=num_classes)

def cvgg16_3_plus_hrank(num_classes): # combining layer pruning and filter pruning
    return VGG(make_layers(cfg['D_1'], batch_norm=True), num_classes=num_classes)





