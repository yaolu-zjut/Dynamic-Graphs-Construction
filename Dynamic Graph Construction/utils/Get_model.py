from model.ResNet_ImageNet import *
from model.ResNet_cifar import *
from model.VGG_ImageNet import *
from model.VGG_cifar import *
import torch


def get_model(args):
    # Note that you can train your own models using train.py
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
    elif args.arch == 'Iresnet18':  # note that finetune needs 1000 classes & pretrained needs less
        model = Iresnet18(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_model/Iresnet18/imagenet_dali/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Iresnet34':  # note that finetune needs 1000 classes & pretrained needs less
        model = Iresnet34(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_model/Iresnet34/imagenet_dali/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Iresnet50':  # note that finetune needs 1000 classes & pretrained needs less
        model = Iresnet50(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_model/Iresnet50/imagenet_dali/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Iresnet101':  # note that finetune needs 1000 classes & pretrained needs less
        model = Iresnet101(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_model/Iresnet101/imagenet_dali/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Iresnet152':  # note that finetune needs 1000 classes & pretrained needs less
        model = Iresnet152(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_model/Iresnet152/imagenet_dali/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Ivgg11':  # note that finetune needs 1000 classes & pretrained needs less
        model = vgg11_bn(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_model/Ivgg11/imagenet_dali/myscores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Ivgg13': # note that finetune needs 1000 classes & pretrained needs less
        model = vgg13_bn(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_model/Ivgg13/imagenet_dali/myscores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Ivgg16':  # note that finetune needs 1000 classes & pretrained needs less
        model = vgg16_bn(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_model/Ivgg16/imagenet_dali/myscores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Ivgg19':  # note that finetune needs 1000 classes & pretrained needs less
        model = vgg19_bn(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_model/Ivgg19/imagenet_dali/myscores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cvgg11_bn':
        model = cvgg11_bn(num_classes=args.num_classes, batch_norm=True)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/CVPR2022/pretrained_model/cvgg11_bn/cifar10/scores.pt', map_location='cuda:%d' % args.gpu)
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/CVPR2022/pretrained_model/cvgg11_bn/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cvgg13_bn':
        model = cvgg13_bn(num_classes=args.num_classes, batch_norm=True)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/CVPR2022/pretrained_model/cvgg13_bn/cifar10/scores.pt', map_location='cuda:%d' % args.gpu)
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/CVPR2022/pretrained_model/cvgg13_bn/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cvgg16_bn':
        model = cvgg16_bn(num_classes=args.num_classes, batch_norm=True)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/CVPR2022/pretrained_model/cvgg16_bn/cifar10/scores.pt', map_location='cuda:%d' % args.gpu)
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/CVPR2022/pretrained_model/cvgg16_bn/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cvgg19_bn':
        model = cvgg19_bn(num_classes=args.num_classes, batch_norm=True)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/CVPR2022/pretrained_model/cvgg19_bn/cifar10/scores.pt', map_location='cuda:%d' % args.gpu)
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/CVPR2022/pretrained_model/cvgg19_bn/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    else:
        assert "the model has not prepared"

    if args.pretrained:
        model.load_state_dict(ckpt)
    return model