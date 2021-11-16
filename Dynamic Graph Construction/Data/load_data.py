import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from args import args


# load data

class CIFAR10:

    def __init__(self):
        super(CIFAR10, self).__init__()

        data_root = '/public/MountData/dataset/cifar10'
        use_cuda = torch.cuda.is_available()
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                    ),
                ]
            ),
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])]),
        )
        self.val_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True
        )


class MNIST:

    def __init__(self):
        super(MNIST, self).__init__()
        use_cuda = torch.cuda.is_available()

        train_dataset = torchvision.datasets.MNIST('./dataset', train=True, download=True,
                                                   transform=transforms.Compose([transforms.ToTensor(),
                                                                                 transforms.Normalize((0.1307,),
                                                                                                      (0.3081,))])),

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

        test_dataset = torchvision.datasets.MNIST('./dataset', train=False, download=True,
                                                  transform=transforms.Compose([transforms.ToTensor(),
                                                                                transforms.Normalize((0.1307,),
                                                                                                     (0.3081,))]), )

        self.val_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
            

class CIFAR100:
    def __init__(self):
        super(CIFAR100, self).__init__()

        data_root = '/public/MountData/dataset/cifar100'
        use_cuda = torch.cuda.is_available()

        # Data Normalization code
        # normalize = transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]], std=[n / 255. for n in [68.2, 65.4, 70.4]]) # original
        normalize = transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])

        train_dataset = torchvision.datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True
        )

        test_dataset = torchvision.datasets.CIFAR100(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )
        self.val_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True
        )
