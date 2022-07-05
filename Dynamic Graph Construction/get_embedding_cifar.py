import datetime
import os
import torch
import tqdm
from args import args
from draw_networkx import directly_return_undirected_weighted_network
from trainer.amp_trainer_dali import validate_ImageNet
from trainer.trainer import validate
from utils.Get_model import get_model
from utils.calculate_similarity import calculate_cosine_similarity_matrix
from utils.design_for_hook import get_inner_feature_for_resnet, get_inner_feature_for_vgg, \
    get_inner_feature_for_smallresnet
from utils.Get_dataset import get_dataset
from utils.network_indicators import calculate_modularity
from utils.utils import set_gpu, get_logger

'''
# setup up:
python get_embedding_cifar.py --gpu 0 --arch cResNet18 --set cifar10 --num_classes 10 --batch_size 500 --pretrained 
'''

def plot_mean_std_picture(total_modularity_list):
    mean = []
    std = []
    x = []
    modular = []
    for kk in range(len(total_modularity_list[00])):  # layer numbers
        for jj in range(len(total_modularity_list)):  # repeat times
            modular.append(total_modularity_list[jj][kk])

        # print(modular)
        mean.append(torch.mean(torch.tensor(modular)))
        std.append(torch.std(torch.tensor(modular)))
        modular = []
        x.append(kk+1)

    return x, mean, std

def main():
    assert args.pretrained, 'this program needs pretrained model'
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.isdir('experiment_data/' + args.arch + '/' + args.set):
        os.mkdir('experiment_data/' + args.arch + '/' + args.set)
    logger = get_logger('experiment_data/' + args.arch + '/' + args.set + '/logger' + now + '.log')
    logger.info(args)

    model = get_model(args)
    model = set_gpu(args, model)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    data = get_dataset(args)
    model.eval()
    batch_count = 0
    if args.evaluate:
        if args.set in ['cifar10', 'cifar100']:
            acc1, acc5 = validate(data.val_loader, model, criterion, args)
        else:
            acc1, acc5 = validate_ImageNet(data.val_loader, model, criterion, args)

        logger.info(acc1)

    inter_feature = []
    modularity_list = []
    total_modularity_list = []
    def hook(module, input, output):
        inter_feature.append(output.clone().detach())

    with torch.no_grad():
        for i, data in tqdm.tqdm(
                enumerate(data.val_loader), ascii=True, total=len(data.val_loader)
        ):
            batch_count += 1
            if args.set == 'imagenet_dali':
                images = data[0]["data"].cuda(non_blocking=True)
                target = data[0]["label"].squeeze().long().cuda(non_blocking=True)
            else:
                images, target = data[0].cuda(args.gpu, non_blocking=True), data[1].cuda(args.gpu, non_blocking=True)

            if args.arch in ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']:
                handle_list = get_inner_feature_for_smallresnet(model, hook, args.arch)
            elif args.arch in ['cResNet18', 'cResNet34', 'cResNet50', 'cResNet101', 'cResNet152', 'Iresnet18', 'Iresnet34', 'Iresnet50',
                               'Iresnet101', 'Iresnet152', 'cResNet152_30', 'cResNet152_26', 'cResNet152_21', 'cResNet152_16', 'cResNet152_11',
                               'cResNet152_5', 'Iresnet152_30', 'Iresnet152_26', 'Iresnet152_21', 'Iresnet152_16', 'Iresnet152_11', 'Iresnet152_5']:
                handle_list = get_inner_feature_for_resnet(model, hook, args.arch)
            else:
                handle_list = get_inner_feature_for_vgg(model, hook, args.arch)

            output = model(images)

            for m in range(len(inter_feature)):
                print('-'*50)
                print(m)
                if len(inter_feature[m].shape) != 2:
                    inter_feature[m] = inter_feature[m].reshape(args.batch_size, -1)
                    print(inter_feature[m].shape)

                similarity_matrix, edges_list = calculate_cosine_similarity_matrix(inter_feature[m], args.topk)
                undirected_weighted_network, undirected_adj = directly_return_undirected_weighted_network(edges_list)
                modularity = calculate_modularity(undirected_weighted_network, target, class_num=args.num_classes)
                modularity_list.append(modularity)

            logger.info(modularity_list)
            total_modularity_list.append(modularity_list)
            modularity_list = []
            inter_feature = []
            for i in range(len(handle_list)):
                handle_list[i].remove()

            if batch_count == 5:
                break

    _, mean, _ = plot_mean_std_picture(total_modularity_list)
    logger.info('CKA: {}'.format(mean))


if __name__ == "__main__":
    main()
