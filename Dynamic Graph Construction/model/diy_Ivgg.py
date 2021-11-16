import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, cast


__all__ = ['VGG', 'Ivgg19_13', 'Ivgg19_14', 'Ivgg19_15', 'Ivgg19_16', 'Ivgg19_17', 'Ivgg19_18']


class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 'M', 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 'M', 512, 'M'],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 'M'],
    'F': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'G': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],  # original VGG19
}


def _vgg(cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model

def Ivgg19_13(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg('A', True, pretrained, progress, **kwargs)

def Ivgg19_14(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg('B', True, pretrained, progress, **kwargs)

def Ivgg19_15(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg('C', True, pretrained, progress, **kwargs)


def Ivgg19_16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg('D', True, pretrained, progress, **kwargs)


def Ivgg19_17(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg('E', True, pretrained, progress, **kwargs)


def Ivgg19_18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg('F', True, pretrained, progress, **kwargs)



if __name__ == "__main__":
    import  torch
    input = torch.randn((2, 3, 224, 224))
    model = Ivgg19_13(num_classes=50)
    print(model)
    output = model(input)
    print(output.shape)
