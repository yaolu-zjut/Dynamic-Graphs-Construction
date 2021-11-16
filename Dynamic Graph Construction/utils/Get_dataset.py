from Data.ImageNet_dali import ImageNetDali
from Data.load_data import *


def get_dataset(args):
    print(f"=> Getting {args.set} dataset")
    if args.set == 'imagenet_dali':
        dataset = ImageNetDali()
    elif args.set == 'cifar10':
        dataset = CIFAR10()
    elif args.set == 'cifar100':
        dataset = CIFAR100()
    return dataset