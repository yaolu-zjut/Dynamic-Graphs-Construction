import datetime
import os
import torch
from args import args
from draw_networkx import directly_return_undirected_weighted_network
from trainer.trainer import validate
from utils.Get_model import get_model
from utils.Get_dataset import get_dataset
from utils.calculate_similarity import calculate_cosine_similarity_matrix
from utils.design_for_hook import get_inner_feature_for_resnet
from utils.network_indicators import calculate_modularity
from utils.utils import set_gpu, get_logger, accuracy
'''
# setup up:
python explore_adversarial_attack.py --gpu 3 --arch cResNet18 --set cifar10 --num_classes 10 --batch_size 500 --pretrained --attack PGD
'''

topk = 3

def main():
    assert args.pretrained, 'this program needs pretrained model'
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.isdir('save_AS_fig/' + args.arch + '/' + args.set):
        os.mkdir('save_AS_fig/' + args.arch + '/' + args.set)
    logger = get_logger('save_AS_fig/' + args.arch + '/' + args.set + '/logger' + now + '.log')
    logger.info(args)
    logger.info('topk')
    logger.info(topk)
    print(args.attack)

    model = get_model(args)
    model = set_gpu(args, model)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    data = get_dataset(args)
    model.eval()
    if args.evaluate:
        acc1, acc5 = validate(data.val_loader, model, criterion, args)
        logger.info(acc1)

    inter_feature = []
    modularity_list = []
    total_modularity_list = []
    oritarget_list = []
    def hook(module, input, output):
        inter_feature.append(output.clone().detach())

    checkpoint = torch.load('save_AS_fig/save_checkpoint_%s_%s_%s.pt' % (args.attack, args.arch, args.set))
    logger.info(checkpoint['attack mode'])
    logger.info(checkpoint['arch'])
    count_list = checkpoint['the index of adversarial sample']
    logger.info(count_list)
    print(len(count_list))
    num_before = 0
    batch_count = 0
    num_after = 500

    print(checkpoint['images'].shape)
    print(checkpoint['target'].shape)
    adv_images, target = checkpoint['images'].cuda(args.gpu, non_blocking=True), checkpoint['target'].cuda(args.gpu, non_blocking=True)

    for ll in range(5):  # You can control this number to repeat the experiment, you need 5 * batch AS in total.
        batch_count += 1
        part_adv_images = adv_images[num_before:num_after, :, :, :]
        part_target = target[num_before:num_after]
        handle_list = get_inner_feature_for_resnet(model, hook, args.arch)
        adv_output = model(part_adv_images)
        adv_acc1, _ = accuracy(adv_output, part_target, topk=(1, 5))
        logger.info('The accuracy of adversarial sample')
        logger.info(adv_acc1)

        for i in range(args.num_classes):  # class number
            a = [j for j, x in enumerate(part_target) if x == i]
            oritarget_list.append(a)

        for m in range(len(inter_feature)):
            print('-'*50)
            print(m)
            if len(inter_feature[m].shape) != 2:
                inter_feature[m] = inter_feature[m].reshape(args.batch_size, -1)

            similarity_matrix, edges_list = calculate_cosine_similarity_matrix(inter_feature[m], args.topk)
            undirected_weighted_network, undirected_adj = directly_return_undirected_weighted_network(edges_list)
            modularity = calculate_modularity(undirected_weighted_network, part_target, class_num=args.num_classes)
            modularity_list.append(modularity)

        logger.info(modularity_list)

        total_modularity_list.append(modularity_list)
        num_before += 500
        num_after += 500
        modularity_list = []
        oritarget_list = []
        inter_feature = []
        for i in range(len(handle_list)):
            handle_list[i].remove()

        if batch_count == 5:
            break


if __name__ == "__main__":
    main()

