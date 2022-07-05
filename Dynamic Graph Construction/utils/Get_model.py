from model.ResNet_ImageNet import *
from model.ResNet_cifar import *
from model.VGG_ImageNet import *
from model.VGG_cifar import *
from model.diy_Iresnet import *
from model.diy_cresnet import *
from model.diy_cvgg import *
import torch
from model.samll_resnet import *


def get_model(args):
    print(f"=> Getting {args.arch}")
    if args.arch == 'cResNet18':
        model = cResNet18(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/Interpretable_CNN/pretrained_model/cResNet18/cifar10/scores_5.pt', map_location='cuda:%d' % args.gpu)
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Interpretable_CNN/pretrained_model/cResNet18/cifar100/scores_5.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cResNet34':
        model = cResNet34(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/Interpretable_CNN/pretrained_model/cResNet34/cifar10/scores_5.pt', map_location='cuda:%d' % args.gpu)
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Interpretable_CNN/pretrained_model/cResNet34/cifar100/scores_5.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cResNet50':
        model = cResNet50(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/Interpretable_CNN/pretrained_model/cResNet50/cifar10/scores_5.pt', map_location='cuda:%d' % args.gpu)
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Interpretable_CNN/pretrained_model/cResNet50/cifar100/scores_5.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cResNet101':
        model = cResNet101(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/Interpretable_CNN/pretrained_model/cResNet101/cifar10/scores_5.pt', map_location='cuda:%d' % args.gpu)
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Interpretable_CNN/pretrained_model/cResNet101/cifar100/scores_5.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cResNet152':
        model = cResNet152(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/Interpretable_CNN/pretrained_model/cResNet152/cifar10/scores_5.pt', map_location='cuda:%d' % args.gpu)
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Interpretable_CNN/pretrained_model/cResNet152/cifar100/scores_5.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Iresnet18':
        # You can also use the pretrained model in torchvision to calculate the modularity
        model = Iresnet18(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_model/Iresnet18/imagenet_dali/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Iresnet34':
        # You can also use the pretrained model in torchvision to calculate the modularity
        model = Iresnet34(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_model/Iresnet34/imagenet_dali/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Iresnet50':
        # You can also use the pretrained model in torchvision to calculate the modularity
        model = Iresnet50(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_model/Iresnet50/imagenet_dali/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Iresnet101':
        # You can also use the pretrained model in torchvision to calculate the modularity
        model = Iresnet101(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_model/Iresnet101/imagenet_dali/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Iresnet152':
        # You can also use the pretrained model in torchvision to calculate the modularity
        model = Iresnet152(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_model/Iresnet152/imagenet_dali/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Ivgg11':
        # You can also use the pretrained model in torchvision to calculate the modularity
        model = vgg11_bn(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_model/Ivgg11/imagenet_dali/myscores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Ivgg13':
        # You can also use the pretrained model in torchvision to calculate the modularity
        model = vgg13_bn(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_model/Ivgg13/imagenet_dali/myscores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Ivgg16':
        # You can also use the pretrained model in torchvision to calculate the modularity
        model = vgg16_bn(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_model/Ivgg16/imagenet_dali/myscores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Ivgg19':
        # You can also use the pretrained model in torchvision to calculate the modularity
        model = vgg19_bn(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_model/Ivgg19/imagenet_dali/myscores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cvgg11_bn':
        model = cvgg11_bn(num_classes=args.num_classes, batch_norm=True)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg11_bn/cifar10/scores.pt', map_location='cuda:%d' % args.gpu)
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg11_bn/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cvgg13_bn':
        model = cvgg13_bn(num_classes=args.num_classes, batch_norm=True)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg13_bn/cifar10/scores.pt', map_location='cuda:%d' % args.gpu)
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg13_bn/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cvgg16_bn':
        model = cvgg16_bn(num_classes=args.num_classes, batch_norm=True)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg16_bn/cifar10/scores.pt', map_location='cuda:%d' % args.gpu)
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg16_bn/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cvgg19_bn':
        model = cvgg19_bn(num_classes=args.num_classes, batch_norm=True)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg19_bn/cifar10/scores.pt', map_location='cuda:%d' % args.gpu)
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg19_bn/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'resnet20':
        model = resnet20(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet20.th', map_location='cuda:%d' % args.gpu)
                ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
    elif args.arch == 'resnet32':
        model = resnet32(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet32.th', map_location='cuda:%d' % args.gpu)
                ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
    elif args.arch == 'resnet44':
        model = resnet44(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet44.th', map_location='cuda:%d' % args.gpu)
                ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
    elif args.arch == 'resnet56':
        model = resnet56(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet56.th', map_location='cuda:%d' % args.gpu)
                ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
    elif args.arch == 'resnet110':
        model = resnet110(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet110.th', map_location='cuda:%d' % args.gpu)
                ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}

    ### pruning ###
    elif args.arch == 'cvgg19_6':
        model = cvgg19_6(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_diy_model/cvgg19_13/cifar10/new_scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cvgg19_5':
        model = cvgg19_5(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_diy_model/cvgg19_14/cifar10/new_scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cvgg19_4':
        model = cvgg19_4(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_diy_model/cvgg19_15/cifar10/new_scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cvgg19_3':
        model = cvgg19_3(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_diy_model/cvgg19_16/cifar10/new_scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cvgg19_2':
        model = cvgg19_2(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_diy_model/cvgg19_17/cifar10/new_scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cvgg19_1':
        model = cvgg19_1(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_diy_model/cvgg19_18/cifar10/new_scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cvgg11_1':
        model = cvgg11_1(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_diy_model/cvgg11_10/cifar10/new_scores.pt')
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_diy_model/cvgg11_10/cifar100/new_scores.pt')
    elif args.arch == 'cvgg11_2':
        model = cvgg11_2(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_diy_model/cvgg11_9/cifar10/new_scores.pt')
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_diy_model/cvgg11_9/cifar100/new_scores.pt')
    elif args.arch == 'cvgg11_3':
        model = cvgg11_3(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_diy_model/cvgg11_8/cifar10/new_scores.pt')
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_diy_model/cvgg11_8/cifar100/new_scores.pt')
    elif args.arch == 'cvgg19_8':
        model = cvgg19_8(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_diy_model/cvgg19_13_2/cifar10/new_scores.pt')
    elif args.arch == 'cvgg16_3':
        model = cvgg16_3(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_diy_model/cvgg16_3/cifar10/new_scores.pt')
    elif args.arch == 'cvgg16_3_plus_hrank':
        model = cvgg16_3_plus_hrank(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_diy_model/cvgg16_3_V1/cifar10/new_scores1.pt')
    elif args.arch == 'cResNet152_30':
        model = cResNet152_30(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/cResNet152_1/cifar10/new_scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cResNet152_26':
        model = cResNet152_26(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/cResNet152_2/cifar10/new_scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cResNet152_21':
        model = cResNet152_21(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/cResNet152_3/cifar10/new_scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cResNet152_16':
        model = cResNet152_16(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/cResNet152_4/cifar10/new_scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cResNet152_11':
        model = cResNet152_11(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/cResNet152_5/cifar10/new_scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cResNet152_5':
        model = cResNet152_5(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/cResNet152_6/cifar10/new_scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Iresnet152_30':
        model = Iresnet152_30(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/Iresnet152/imagenet_dali/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Iresnet152_26':
        model = Iresnet152_26(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/Iresnet152_3/imagenet_dali/new_scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Iresnet152_21':
        model = Iresnet152_21(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/Iresnet152_5/imagenet_dali/new_scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Iresnet152_16':
        model = Iresnet152_16(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/Iresnet152_7/imagenet_dali/new_scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Iresnet152_11':
        model = Iresnet152_11(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/Iresnet152_9/imagenet_dali/new_scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Iresnet152_5':
        model = Iresnet152_5(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/Iresnet152_11/imagenet_dali/new_scores.pt', map_location='cuda:%d' % args.gpu)
    else:
        assert "the model has not prepared"

    if args.pretrained:
        model.load_state_dict(ckpt)
    return model