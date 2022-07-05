import argparse
import sys

global args
args = None

parser = argparse.ArgumentParser(description="PyTorch Robust Pruning", epilog="End of Parameters")
parser.add_argument("--set", help="name of dataset", type=str, default='cifar10', choices=["cifar10", "cifar100", 'imagenet_dali'])
parser.add_argument("--nonlinearity", default="relu", help="Nonlinearity used by initialization")
parser.add_argument("--pretrained", action="store_true", help="use pre-trained model")
parser.add_argument("--finetune", action="store_true", help="finetune pre-trained model")
parser.add_argument("--arch", metavar="ARCH", default='cResNet18', help="model architecture")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
parser.add_argument("--evaluate", dest="evaluate", action="store_true", help="evaluate model on validation set")
parser.add_argument("--epochs", default=150, type=int, metavar="N", help="number of total epochs to run")
parser.add_argument("--num_classes", default=10, type=int, help="number of class")
parser.add_argument("--random_seed", default=0, type=int, help="random seed")
parser.add_argument("--topk", default=3, type=int, help="topk nodes to set connections")
parser.add_argument("--print-freq", default=100, type=int, metavar="N", help="print frequency (default: 10)")
parser.add_argument("--batch_size", default=500, type=int, help="batch_size")
parser.add_argument("--log-dir", help="Where to save the runs. If None use ./runs", default=None)
parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
parser.add_argument("--multigpu", default=None, type=lambda x: [int(a) for a in x.split(",")],
                    help="Which GPUs to use for multigpu training")
parser.add_argument("--gpu", default=0, type=int, help="Which GPU to use for training")
parser.add_argument('--log_root', default='log',
                    help='the directory to save the logs or other imformations (e.g. images)')
parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum(defalt: 0.9)")
parser.add_argument('--weight_decay', type=float, default=0.005, help="SGD weight decay(defalt: 1e-4)")
parser.add_argument("--save_every", default=-1, type=int, help="Save every epochs")
parser.add_argument('--lr_decay_step', default='50,100', type=str, help='learning rate')
parser.add_argument("--attack", metavar="attack mode", default='CW', choices=["CW", "FGSM", 'PGD', "JSMA", "OnePixel", 'LocalSearch'], help="Adversarial attack mode")
parser.add_argument("--perturbation_ratio", default=0.1, type=float, help="Perturbation ratio of images")
args = parser.parse_args()
