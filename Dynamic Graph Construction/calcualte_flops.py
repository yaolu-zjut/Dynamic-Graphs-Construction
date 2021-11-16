import torch
import argparse
from thop import profile

from model.VGG_ImageNet import vgg19_bn
from model.VGG_cifar import cvgg19_bn
# from utils.Get_diy_model import get_model
from utils.Get_model import get_model

parser = argparse.ArgumentParser(description='Calculating flops and params')

parser.add_argument(
    '--input_image_size',
    type=int,
    default=32,
    help='The input_image_size')
parser.add_argument("--gpu", default=None, type=int, help="Which GPU to use for training")
parser.add_argument("--arch", default=None, type=str, help="arch")
parser.add_argument("--pretrained", action="store_true", help="use pre-trained model")
parser.add_argument("--num_classes", default=10, type=int, help="number of class")
parser.add_argument("--finetune", action="store_true", help="finetune pre-trained model")
parser.add_argument("--set", help="name of dataset", type=str, default='cifar10')
args = parser.parse_args()
torch.cuda.set_device(args.gpu)
# model = torch.load('trained_model/VGG16_cifar10_random_seed_1234/best_pruning_model.pth').cuda(args.gpu)
model = get_model(args).cuda()
# model = cvgg19_bn(num_classes=args.num_classes).cuda()
# model = vgg19_bn(num_classes=args.num_classes).cuda()
# ckpt = torch.load('/public/ly/CVPR2022/pretrained_model/Ivgg19/imagenet_dali/myscores.pt', map_location='cuda:%d' % args.gpu)
# model.load_state_dict(ckpt)
model.eval()

# calculate model size
input_image_size = args.input_image_size
input_image = torch.randn(1, 3, input_image_size, input_image_size).cuda()
flops, params = profile(model, inputs=(input_image,))

print('Params: %.2f' % (params))
print('Flops: %.2f' % (flops))
