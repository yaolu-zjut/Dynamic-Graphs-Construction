from model.VGG_cifar import *
from model.VGG_ImageNet import *

cfgs = {
    'cResNet18': [2, 2, 2, 2],
    'cResNet34': [3, 4, 6, 3],
    'cResNet50': [3, 4, 6, 3],
    'cResNet101': [3, 4, 23, 3],
    'cResNet152': [3, 8, 36, 3],
    'Iresnet18': [2, 2, 2, 2],
    'Iresnet34': [3, 4, 6, 3],
    'Iresnet50': [3, 4, 6, 3],
    'Iresnet101': [3, 4, 23, 3],
    'Iresnet152': [3, 8, 36, 3],
    'Ivgg11': ['features.0', 'features.4', 'features.8', 'features.11', 'features.15', 'features.18', 'features.22', 'features.25'],
    'Ivgg13': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.21', 'features.24', 'features.28', 'features.31'],
    'Ivgg16': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20', 'features.24', 'features.27', 'features.30', 'features.34', 'features.37', 'features.40'],
    'Ivgg19': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17','features.20', 'features.23', 'features.27', 'features.30', 'features.33', 'features.36', 'features.40', 'features.43', 'features.46', 'features.49'],
    'cvgg11_bn': ['features.0', 'features.4', 'features.8', 'features.11', 'features.15', 'features.18', 'features.22',
               'features.25'],
    'cvgg13_bn': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.21',
               'features.24', 'features.28', 'features.31'],
    'cvgg16_bn': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20',
               'features.24', 'features.27', 'features.30', 'features.34', 'features.37', 'features.40'],
    'cvgg19_bn': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20',
               'features.23', 'features.27', 'features.30', 'features.33', 'features.36', 'features.40', 'features.43',
               'features.46', 'features.49'],
}

def get_inner_feature_for_resnet(model, hook, arch):
    cfg = cfgs[arch]
    print('cfg:', cfg)
    handle = model.conv1.register_forward_hook(hook)
    # handle.remove()  # free memory
    for i in range(cfg[0]):
        handle = model.layer1[i].register_forward_hook(hook)

    for i in range(cfg[1]):
        handle = model.layer2[i].register_forward_hook(hook)

    for i in range(cfg[2]):
        handle = model.layer3[i].register_forward_hook(hook)

    for i in range(cfg[3]):
        handle = model.layer4[i].register_forward_hook(hook)


def get_inner_feature_for_vgg(model, hook, arch):
    cfg = cfgs[arch]
    print('cfg:', cfg)
    count = 0
    for idx, m in enumerate(model.named_modules()):
        name, module = m[0], m[1]
        if count < len(cfg):
            if name == cfg[count]:
                print(module)
                handle = module.register_forward_hook(hook)
                count += 1
        else:
            break


if __name__ == "__main__":
    # demo
    import torch
    input = torch.randn((2, 3, 32, 32))
    inter_feature = []
    model = cvgg19_bn(10)
    # model = cvgg16_bn(10)
    # model = cvgg13_bn(10)
    # model = cvgg11_bn(10)
    print(model)


    def hook(module, input, output):
        inter_feature.append(output.clone().detach())
    get_inner_feature_for_vgg(model, hook, 'cvgg19')
    model(input)
