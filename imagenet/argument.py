import argparse


def str2bool(v):
    """Cast string to boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def ipc_epoch(ipc, factor, nclass=10, bound=-1):
    """Calculating training epochs for ImageNet
    """
    factor = max(factor, 1)
    ipc *= factor**2

    if ipc == 1:
        epoch = 3000
    elif ipc <= 10:
        epoch = 2000
    elif ipc <= 50:
        epoch = 1500
    elif ipc <= 200:
        epoch = 1000
    elif ipc <= 500:
        epoch = 500
    else:
        epoch = 300

    if nclass == 100:
        epoch = int((2 / 3) * epoch)
        epoch = epoch - (epoch % 100)

    return epoch


parser = argparse.ArgumentParser(description='')
# Dataset
parser.add_argument('--imagenet_dir', default='/ssd_data/imagenet/', type=str)
parser.add_argument('--nclass', default=10, type=int, help='number of classes in trianing dataset')
parser.add_argument('--size', default=224, type=int, help='resolution of data')

# Network
parser.add_argument('-n',
                    '--net_type',
                    default='resnet_ap',
                    type=str,
                    help='network type: resnet, resnet_ap, efficientnet')
parser.add_argument('--depth', default=10, type=int, help='depth of the network. 10/18 for resnet')

# Testing
parser.add_argument(
    '--epochs',
    default=300,
    type=int,
    help='number of training epochs. This will automatically defined by ipc_epoch function.')
parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size for training')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('-j', '--workers', default=8, type=int, help='number of data loading workers')
parser.add_argument('--print-freq',
                    '-p',
                    default=50,
                    type=int,
                    help='print frequency (default: 50)')
parser.add_argument('--mixup',
                    default='cut',
                    type=str,
                    choices=('vanilla', 'cut'),
                    help='mixup choice for evaluation')
parser.add_argument('--beta', default=1.0, type=float, help='mixup beta distribution')
parser.add_argument('--mix_p', default=1.0, type=float, help='mixup probability')

parser.add_argument('--repeat', default=1, type=int, help='number of test repetetion')

# Condensed data
parser.add_argument('-s',
                    '--slct_type',
                    type=str,
                    default='idc',
                    help='data condensation type (idc, random, herding)')
parser.add_argument('-i', '--ipc', type=int, default=10, help='number of condensed data per class')
parser.add_argument('-f',
                    '--factor',
                    type=int,
                    default=1,
                    help='multi-formation factor. (1 for IDC-I, 3 for IDC)')

parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

args.size = 224
if args.nclass >= 100:
    args.lr = 0.1
    args.weight_decay = 1e-4
    args.batch_size = max(128, args.batch_size)
else:
    args.lr = 0.01
    args.weight_decay = 5e-4
    args.batch_size = max(64, args.batch_size)

args.epochs = ipc_epoch(args.ipc, args.factor, args.nclass)
args.epoch_print_freq = args.epochs // 4
