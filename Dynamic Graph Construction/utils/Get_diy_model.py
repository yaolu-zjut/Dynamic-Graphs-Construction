from model.diy_Iresnet import *
from model.diy_Ivgg import *
from model.diy_cresnet import *
from model.diy_cvgg import *
import torch

def get_model(args):
    # Note that you can train your own models using train_diy_model.py
    print(f"=> Getting {args.arch}")
    if args.arch == 'cvgg19_13':
        # eval('model = %s(num_classes=args.num_classes)' %args.arch)
        model = cvgg19_13(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/cvgg19_bn_1/cifar10/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cvgg19_14':
        model = cvgg19_14(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/cvgg19_bn_2/cifar10/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cvgg19_15':
        model = cvgg19_15(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/cvgg19_bn_3/cifar10/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cvgg19_16':
        model = cvgg19_16(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/cvgg19_bn_4/cifar10/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cvgg19_17':
        model = cvgg19_17(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/cvgg19_bn_5/cifar10/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cvgg19_18':
        model = cvgg19_18(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/cvgg19_bn_6/cifar10/scores.pt', map_location='cuda:%d' % args.gpu)

    elif args.arch == 'cResNet152_62':
        model = cResNet152_62(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/cResNet152_1/cifar10/new_scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cResNet152_74':
        model = cResNet152_74(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/cResNet152_2/cifar10/new_scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cResNet152_89':
        model = cResNet152_89(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/cResNet152_3/cifar10/new_scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cResNet152_104':
        model = cResNet152_104(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/cResNet152_4/cifar10/new_scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cResNet152_119':
        model = cResNet152_119(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/cResNet152_5/cifar10/new_scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cResNet152_137':
        model = cResNet152_137(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/cResNet152_6/cifar10/new_scores.pt', map_location='cuda:%d' % args.gpu)

    elif args.arch == 'Ivgg19_13':
        model = Ivgg19_13(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/Ivgg19_1/imagenet_dali/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Ivgg19_14':
        model = Ivgg19_14(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/Ivgg19_2/imagenet_dali/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Ivgg19_15':
        model = Ivgg19_15(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/Ivgg19_3/imagenet_dali/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Ivgg19_16':
        model = Ivgg19_16(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/Ivgg19_4/imagenet_dali/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Ivgg19_17':
        model = Ivgg19_17(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/Ivgg19_5/imagenet_dali/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Ivgg19_18':
        model = Ivgg19_18(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/Ivgg19_6/imagenet_dali/scores.pt', map_location='cuda:%d' % args.gpu)

    elif args.arch == 'Iresnet152_62':
        model = Iresnet152_62(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/Iresnet152/imagenet_dali/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Iresnet152_74':
        model = Iresnet152_74(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/Iresnet152_3/imagenet_dali/new_scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Iresnet152_89':
        model = Iresnet152_89(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/Iresnet152_5/imagenet_dali/new_scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Iresnet152_104':
        model = Iresnet152_104(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/Iresnet152_7/imagenet_dali/new_scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Iresnet152_119':
        model = Iresnet152_119(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/Iresnet152_9/imagenet_dali/new_scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'Iresnet152_137':
        model = Iresnet152_137(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/CVPR2022/pretrained_diy_model/Iresnet152_11/imagenet_dali/new_scores.pt', map_location='cuda:%d' % args.gpu)
    else:
        assert "the model has not prepared"

    if args.pretrained:
        model.load_state_dict(ckpt)
    return model