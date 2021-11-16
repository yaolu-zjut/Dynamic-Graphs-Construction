import datetime
import os
import tqdm
from args import args
from draw_networkx import directly_return_undirected_weighted_network
from model.ResNet_cifar import *
from trainer.amp_trainer_dali import validate_ImageNet
from utils.Get_diy_model import get_model
from utils.calculate_similarity import calculate_similarity, calculate_distance_to_similarity
from utils.Get_dataset import get_dataset
from utils.design_for_diy_hook import get_inner_feature_for_vgg, get_inner_feature_for_resnet
from utils.network_indicators import calculate_modularity
from utils.utils import set_gpu, get_logger
import matplotlib.pyplot as plt

'''
# setup up:
python python get_diy_embedding_imagenet.py --gpu 1 --arch Iresnet152_21 --set imagenet_dali --num_classes 50 --batch_size 500 --pretrained --evaluate
'''

topk = 3

def main():
    assert args.pretrained, 'this program needs pretrained model'
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.isdir('save_fig/' + args.arch + '/' + args.set):
        os.mkdir('save_fig/' + args.arch + '/' + args.set)
    logger = get_logger('save_fig/' + args.arch + '/' + args.set + '/logger' + now + '.log')
    logger.info(args)
    logger.info(topk)
    model = get_model(args)
    logger.info(model)
    model = set_gpu(args, model)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    data = get_dataset(args)
    model.eval()
    if args.evaluate:
        acc1, acc5 = validate_ImageNet(data.val_loader, model, criterion, args)  # checked
        logger.info(acc1)

    inter_feature = []
    modularity_list = []
    oritarget_list = []
    def hook(module, input, output):
        inter_feature.append(output.clone().detach())

    with torch.no_grad():
        for i, data in tqdm.tqdm(
                enumerate(data.val_loader), ascii=True, total=len(data.val_loader)
        ):
            images = data[0]["data"].cuda(non_blocking=True)
            target = data[0]["label"].squeeze().long().cuda(non_blocking=True)
            get_inner_feature_for_resnet(model, hook, args.arch)
            # get_inner_feature_for_vgg(model, hook, args.arch)
            output = model(images)
            for i in range(args.num_classes):  # class number
                a = [j for j, x in enumerate(target) if x == i]
                oritarget_list.append(a)

            print('oritarget_list:', oritarget_list)

            # to check nodes number and adj
            topk1 = (1, 5)
            maxk = max(topk1)
            _, pred = output.topk(maxk, 1, True, True)
            whole_label = torch.cat((target.unsqueeze(1), pred), 1)  # concat true & pred
            print('whole_label:', whole_label)

            for m in range(len(inter_feature)):
                print('-'*50)
                print(m)
                if len(inter_feature[m].shape) != 2:
                    inter_feature[m] = inter_feature[m].reshape(args.batch_size, -1)

                # # note that use Pearson will make edges less than expectation
                similarity_matrix, edges_list = calculate_similarity(inter_feature[m], topk, 'cosine')
                # similarity_matrix, edges_list = calculate_distance_to_similarity(inter_feature[m], topk, similarity_function='cos_distance_softmax')
                undirected_weighted_network, undirected_adj = directly_return_undirected_weighted_network(edges_list)
                modularity = calculate_modularity(undirected_weighted_network, target, class_num=args.num_classes)
                modularity_list.append(modularity)

            logger.info(modularity_list)
            break
    plt.show()


if __name__ == "__main__":
    main()