from Data.ImageNet_dali import ImageNetDali
from Data.load_data import *


def get_dataset(args):
    print(f"=> Getting {args.set} dataset")
    if args.set == 'imagenet_dali':
        dataset = ImageNetDali()
    elif args.set == 'cifar10':
        dataset = CIFAR10()  # for normal training
    elif args.set == 'cifar100':
        dataset = CIFAR100()  # for normal training
    else:
        print('no required dataset')
    return dataset