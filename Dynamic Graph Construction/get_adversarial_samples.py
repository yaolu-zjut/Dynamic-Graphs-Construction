import datetime
import os
import torch
import tqdm
from advertorch.attacks import GradientSignAttack, PGDAttack, CarliniWagnerL2Attack, JacobianSaliencyMapAttack, \
    SinglePixelAttack, LocalSearchAttack
from args import args
from trainer.trainer import validate
from utils.Get_model import get_model
from utils.Get_dataset import get_dataset
from utils.utils import set_gpu, get_logger, accuracy
import matplotlib.pyplot as plt
'''
# setup up:
python get_adversarial_samples.py --gpu 3 --arch cResNet18 --set cifar10 --num_classes 10 --batch_size 1 --pretrained --attack PGD
'''

topk = 3

def main():
    assert args.pretrained, 'this program needs pretrained model'
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.isdir('save_AS_fig/' + args.arch + '/' + args.set):
        os.mkdir('save_AS_fig/' + args.arch + '/' + args.set)
    logger = get_logger('save_AS_fig/' + args.arch + '/' + args.set + '/logger' + now + '.log')
    logger.info(args)
    count = -1
    count_list = []

    model = get_model(args)
    logger.info(model)
    model = set_gpu(args, model)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    data = get_dataset(args)
    model.eval()
    if args.evaluate:
        acc1, acc5 = validate(data.val_loader, model, criterion, args)
        logger.info(acc1)

    if args.attack == 'FGSM':  # checked
        attack = GradientSignAttack(model, targeted=False)
    elif args.attack == 'PGD':  # checked
        attack = PGDAttack(model, targeted=False)
    elif args.attack == 'CW':
        attack = CarliniWagnerL2Attack(model, num_classes=10, targeted=False, max_iterations=200)
    elif args.attack == 'JSMA': # checked
        attack = JacobianSaliencyMapAttack(model, num_classes=10)
        attack.targeted = False
    elif args.attack == 'OnePixel':
        attack = SinglePixelAttack(model, targeted=False)
    elif args.attack == 'LocalSearch':
        attack = LocalSearchAttack(model, targeted=False)

    num_of_adversample = 100
    image_save = torch.zeros((num_of_adversample, 3, 32, 32))
    target_save = torch.zeros(num_of_adversample, 5)
    groundtrue_save = torch.zeros(num_of_adversample)

    for i, data in tqdm.tqdm(
            enumerate(data.val_loader), ascii=True, total=len(data.val_loader)
    ):
        aa = count
        images, target = data[0].cuda(args.gpu, non_blocking=True), data[1].cuda(args.gpu, non_blocking=True)
        image_save, groundtrue_save, count = samples_filter(images, target, model, count, image_save, target_save, groundtrue_save, attack)
        if aa != count:
            count_list.append(i)
        if count == num_of_adversample - 1:
            break

    adv_output = model(image_save.cuda())
    adv_acc1, adv_acc5 = accuracy(adv_output, groundtrue_save.cuda(), topk=(1, 5))
    print(adv_acc1)

    save_checkpoint = {'attack mode': args.attack,
                       'arch': args.arch,
                       'images': image_save,
                       'target': groundtrue_save,
                       'the index of adversarial sample': count_list}
    torch.save(save_checkpoint, 'save_AS_fig/save_checkpoint_%s_%s_%s.pt' % (args.attack, args.arch, args.set))
    plt.show()


def samples_filter(images, target, model, count, image_save, target_save, groundtrue_save, attack):
    '''

    Args:
        images: input images (Only support single image input)
        target: the targets of input images (Only support single target)
        model: pre-trained model
        count: Count the number of images
        image_save: the matrix to save the disturbed images
        target_save: the matrix to save the predicted labels
        groundtrue_save: the matrix to save the ground true label
        attack: Attack method

    Returns: image_save, groundtrue_save, count

    '''
    output = model(images)  # For original images

    topk1 = (1, 5)
    maxk = max(topk1)
    _, pred = output.topk(maxk, 1, True, True)
    if pred[0][0] == target:
        adv_images = attack(images, target)
        adv_output = model(adv_images)
        _, adv_pred = adv_output.topk(maxk, 1, True, True)
        if adv_pred[0][0] != target:  # batch_size should be 1
            print(adv_pred[0][0], target)
            count += 1
            image_save[count] = adv_images
            target_save[count] = pred
            groundtrue_save[count] = target

    return image_save, groundtrue_save, count


if __name__ == "__main__":
    main()
