from args import args

cfgs = {
    'cResNet18': [2, 2, 2, 2],
    'cResNet34': [3, 4, 6, 3],
    'cResNet50': [3, 4, 6, 3],
    'cResNet101': [3, 4, 23, 3],
    'cResNet152': [3, 8, 36, 3],
    'resnet20': [3, 3, 3],
    'resnet32': [5, 5, 5],
    'resnet44': [7, 7, 7],
    'resnet56': [9, 9, 9],
    'resnet110': [18, 18, 18],
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
    ###  pruning ###
    'cResNet152_30': [3, 8, 6, 3],
    'cResNet152_26': [3, 8, 10, 3],
    'cResNet152_21': [3, 8, 15, 3],
    'cResNet152_16': [3, 8, 20, 3],
    'cResNet152_11': [3, 8, 25, 3],
    'cResNet152_5': [3, 8, 31, 3],
    'Iresnet152_30': [3, 8, 6, 3],
    'Iresnet152_26': [3, 8, 10, 3],
    'Iresnet152_21': [3, 8, 15, 3],
    'Iresnet152_16': [3, 8, 20, 3],
    'Iresnet152_11': [3, 8, 25, 3],
    'Iresnet152_5': [3, 8, 31, 3],
    'cvgg11_1': ['features.0', 'features.4', 'features.8', 'features.11', 'features.15', 'features.18', 'features.22'],
    'cvgg11_2': ['features.0', 'features.4', 'features.8', 'features.11', 'features.15', 'features.19'],
    'cvgg11_3': ['features.0', 'features.4', 'features.8', 'features.12', 'features.16'],
    'cvgg19_1': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20',
                 'features.23', 'features.27', 'features.30', 'features.33', 'features.36', 'features.40',
                 'features.43', 'features.46'],
    'cvgg19_2': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20',
                 'features.23', 'features.27', 'features.30', 'features.33', 'features.36', 'features.40',
                 'features.43'],
    'cvgg19_3': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20',
                 'features.23', 'features.27', 'features.30', 'features.33', 'features.36', 'features.40'],
    'cvgg19_4': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20',
                 'features.23', 'features.27', 'features.30', 'features.33', 'features.37'],
    'cvgg19_5': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20',
                 'features.23', 'features.27', 'features.30', 'features.34'],
    'cvgg19_6': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20',
                 'features.23', 'features.27', 'features.31'],
}

def get_inner_feature_for_resnet(model, hook, arch):
    handle_list = []
    cfg = cfgs[arch]
    if args.multigpu is not None:
        for i in range(len(cfg)):
            cfg[i] = 'module.' + cfg[i]
    print('cfg:', cfg)
    handle = model.conv1.register_forward_hook(hook)  # here!!!
    handle_list.append(handle)
    # handle.remove()  # free memory
    for i in range(cfg[0]):
        handle = model.layer1[i].register_forward_hook(hook)
        handle_list.append(handle)
    for i in range(cfg[1]):
        handle = model.layer2[i].register_forward_hook(hook)
        handle_list.append(handle)
    for i in range(cfg[2]):
        handle = model.layer3[i].register_forward_hook(hook)
        handle_list.append(handle)
    for i in range(cfg[3]):
        handle = model.layer4[i].register_forward_hook(hook)
        handle_list.append(handle)
    return handle_list


def get_inner_feature_for_smallresnet(model, hook, arch):
    handle_list = []
    cfg = cfgs[arch]
    if args.multigpu is not None:
        for i in range(len(cfg)):
            cfg[i] = 'module.' + cfg[i]
    print('cfg:', cfg)
    handle = model.conv1.register_forward_hook(hook)
    handle_list.append(handle)
    # handle.remove()  # free memory
    for i in range(cfg[0]):
        handle = model.layer1[i].register_forward_hook(hook)
        handle_list.append(handle)
    for i in range(cfg[1]):
        handle = model.layer2[i].register_forward_hook(hook)
        handle_list.append(handle)
    for i in range(cfg[2]):
        handle = model.layer3[i].register_forward_hook(hook)
        handle_list.append(handle)
    return handle_list


def get_inner_feature_for_vgg(model, hook, arch):
    cfg = cfgs[arch]
    if args.multigpu is not None:
        for i in range(len(cfg)):
            cfg[i] = 'module.' + cfg[i]

    handle_list = []
    print('cfg:', cfg)
    count = 0
    for idx, m in enumerate(model.named_modules()):
        name, module = m[0], m[1]
        if count < len(cfg):
            if name == cfg[count]:
                print(module)
                handle = module.register_forward_hook(hook)
                handle_list.append(handle)
                count += 1
        else:
            break
    return handle_list
