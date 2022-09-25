# Codes borrowed from https://github.com/snu-mllab/Efficient-Dataset-Condensation

import os
import numpy as np
from train import train
from data import ImageFolder, MultiEpochsDataLoader
from data import transform_imagenet


def load_val(args):
    """Load ImageNet valid loader
    """

    valdir = os.path.join(args.imagenet_dir, 'val')

    _, test_transform = transform_imagenet(from_tensor=False, size=args.size)
    val_dataset = ImageFolder(valdir, test_transform, nclass=args.nclass, load_memory=False)

    print(len(val_dataset))
    val_loader = MultiEpochsDataLoader(val_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       persistent_workers=True,
                                       num_workers=4)

    return val_loader


def test_data(args, train_loader, val_loader, model, repeat=1, num_val=4):
    """Train neural networks on condensed data
    """

    args.epoch_print_freq = args.epochs // num_val

    best_acc_l = []
    acc_l = []
    for _ in range(repeat):
        best_acc, acc = train(args, model, train_loader, val_loader)
        best_acc_l.append(best_acc)
        acc_l.append(acc)
    print(f'Repeat {repeat} => Best, last acc: {np.mean(best_acc_l):.1f} {np.mean(acc_l):.1f}\n')


if __name__ == '__main__':
    from argument import args
    import torch.backends.cudnn as cudnn
    from models.load import load_model
    from loader_idc import load_dataloader
    cudnn.benchmark = True

    val_loader = load_val(args)
    train_loader = load_dataloader(args)
    model = load_model(args.net_type, args.depth, nclass=args.nclass).cuda()
    test_data(args, train_loader, val_loader, repeat=args.repeat, model=model)
