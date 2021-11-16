import math
import torch.nn as nn


__all__ = ['VGG', 'cvgg19_13', 'cvgg19_14', 'cvgg19_15', 'cvgg19_16', 'cvgg19_17', 'cvgg19_18']


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
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
    'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 'M', 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 'M', 512, 'M'],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 'M'],
    'F': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'G': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],  # original VGG19
}


def cvgg19_18(num_classes):
    return VGG(make_layers(cfg['F'], batch_norm=True), num_classes=num_classes)

def cvgg19_17(num_classes):
    return VGG(make_layers(cfg['E'], batch_norm=True), num_classes=num_classes)

def cvgg19_16(num_classes):
    return VGG(make_layers(cfg['D'], batch_norm=True), num_classes=num_classes)

def cvgg19_15(num_classes):
    return VGG(make_layers(cfg['C'], batch_norm=True), num_classes=num_classes)

def cvgg19_14(num_classes):
    return VGG(make_layers(cfg['B'], batch_norm=True), num_classes=num_classes)

def cvgg19_13(num_classes):
    return VGG(make_layers(cfg['A'], batch_norm=True), num_classes=num_classes)


if __name__ == "__main__":
    import torch
    input = torch.randn((2, 3, 32, 32))
    model = cvgg19_13(10)
    print(model)
    output = model(input)
    print(output)




