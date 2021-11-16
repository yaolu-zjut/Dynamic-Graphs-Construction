import os
import sys
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from args import args
import datetime
from model.ResNet_cifar import *
from trainer.amp_trainer_dali import train_ImageNet, validate_ImageNet
from trainer.trainer import validate, train
from utils.Get_dataset import get_dataset
from utils.Get_diy_model import get_model
from utils.utils import set_random_seed, set_gpu, Logger, get_logger


def main():
    print(args)
    sys.stdout = Logger('print process.log', sys.stdout)

    if args.random_seed is not None:
        set_random_seed(args.random_seed)

    main_worker(args)


def main_worker(args):
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.isdir('pretrained_diy_model/' + args.arch + '/' + args.set):
        os.mkdir('pretrained_diy_model/' + args.arch + '/' + args.set)
    logger = get_logger('pretrained_diy_model/' + args.arch + '/' + args.set + '/logger' + now + '.log')
    logger.info(args.arch)
    logger.info(args.set)
    logger.info(args.batch_size)
    logger.info(args.weight_decay)
    logger.info(args.lr)
    logger.info(args.epochs)
    logger.info(args.lr_decay_step)
    logger.info(args.num_classes)

    model = get_model(args)
    model = set_gpu(args, model)
    logger.info(model)
    criterion = nn.CrossEntropyLoss().cuda()
    data = get_dataset(args)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    # create recorder
    args.start_epoch = args.start_epoch or 0

    # Start training
    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()
        if args.set == 'imagenet_dali':
            # for imagenet
            train_acc1, train_acc5 = train_ImageNet(data.train_loader, model, criterion, optimizer, epoch, args)
            acc1, acc5 = validate_ImageNet(data.val_loader, model, criterion, args)
        else:
            #  for small datasets
            train_acc1, train_acc5 = train(data.train_loader, model, criterion, optimizer, epoch, args)
            acc1, acc5 = validate(data.val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)
        save = ((epoch % args.save_every) == 0) and args.save_every > 0
        if is_best or save or epoch == args.epochs - 1:
            if is_best:
                torch.save(model.state_dict(), 'pretrained_diy_model/' + args.arch + '/' + args.set + "/new_scores.pt")
                logger.info(best_acc1)


if __name__ == "__main__":
    # setup: python train_diy_model.py --gpu 2 --arch cResNet152_1  --set cifar10  --lr 0.01 --batch_size 256 --weight_decay 0.005 --epochs 300 --lr_decay_step 100,200  --num_classes 10
    main()
