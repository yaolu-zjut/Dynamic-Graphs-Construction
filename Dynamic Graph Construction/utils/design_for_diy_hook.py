from model.diy_cvgg import *
from model.diy_Ivgg import *


cfgs = {
    'cResNet152_62': [3, 8, 6, 3],
    'cResNet152_74': [3, 8, 10, 3],
    'cResNet152_89': [3, 8, 15, 3],
    'cResNet152_104': [3, 8, 20, 3],
    'cResNet152_119': [3, 8, 25, 3],
    'cResNet152_137': [3, 8, 31, 3],
    'Iresnet152_62': [3, 8, 6, 3],
    'Iresnet152_74': [3, 8, 10, 3],
    'Iresnet152_89': [3, 8, 15, 3],
    'Iresnet152_104': [3, 8, 20, 3],
    'Iresnet152_119': [3, 8, 25, 3],
    'Iresnet152_137': [3, 8, 31, 3],
    'cvgg19_18': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20',
                 'features.23', 'features.27', 'features.30', 'features.33', 'features.36', 'features.40',
                 'features.43', 'features.46'],
    'cvgg19_17': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20',
                 'features.23', 'features.27', 'features.30', 'features.33', 'features.36', 'features.40',
                 'features.43'],
    'cvgg19_16': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20',
                 'features.23', 'features.27', 'features.30', 'features.33', 'features.36', 'features.40'],
    'cvgg19_15': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20',
                 'features.23', 'features.27', 'features.30', 'features.33', 'features.37'],
    'cvgg19_14': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20',
                 'features.23', 'features.27', 'features.30', 'features.34'],
    'cvgg19_13': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20',
                 'features.23', 'features.27', 'features.31'],
    'Ivgg19_18': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20',
                 'features.23', 'features.27', 'features.30', 'features.33', 'features.36', 'features.40',
                 'features.43', 'features.46'],
    'Ivgg19_17': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20',
                 'features.23', 'features.27', 'features.30', 'features.33', 'features.36', 'features.40',
                 'features.43'],
    'Ivgg19_16': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20',
                 'features.23', 'features.27', 'features.30', 'features.33', 'features.36', 'features.40'],
    'Ivgg19_15': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20',
                 'features.23', 'features.27', 'features.30', 'features.33', 'features.37'],
    'Ivgg19_14': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20',
                 'features.23', 'features.27', 'features.30', 'features.34'],
    'Ivgg19_13': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20',
                 'features.23', 'features.27', 'features.31'],
}

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


if __name__ == "__main__":
    # demo
    import torch
    input = torch.randn((2, 3, 32, 32))
    # input = torch.randn((2, 3, 224, 224))
    inter_feature = []
    # model = cvgg19_1_bn(10)
    # model = cvgg19_2_bn(10)
    # model = cvgg19_3_bn(10)
    # model = cvgg19_4_bn(10)
    # model = cvgg19_5_bn(10)
    model = cvgg19_14(num_classes=10)
    print(model)


    # def hook(module, input, output):
    #     inter_feature.append(output.clone().detach())
    # get_inner_feature_for_vgg(model, hook, 'cvgg19_6')
    out = model(input)
    print(out.shape)
