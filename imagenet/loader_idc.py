# Codes borrowed from https://github.com/snu-mllab/Efficient-Dataset-Condensation

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from data import transform_imagenet, MultiEpochsDataLoader
from data import ImageFolder, TensorDataset

DATA_PATH = "/storage/janghyun/results/publication/icml22/results"


def return_data_path(args):
    if args.slct_type == 'idc':
        if args.factor > 1:
            init = 'mix'
        else:
            init = 'random'

        if args.nclass == 10:
            name = f'imagenet10/resnet10apin_grad_l1_ely10_nd500_cut_factor{args.factor}_{init}'
        elif args.nclass == 100:
            name = f'imagenet100/resnet10apin_grad_l1_pt5_nd500_cut_nlr0.1_wd0.0001_factor{args.factor}_lr0.001_b_real128_{init}'

    path = f'{name}_ipc{args.ipc}'

    return os.path.join(DATA_PATH, path)


def decode_zoom(img, target, factor, size=-1):
    if size == -1:
        size = img.shape[-1]
    resize = nn.Upsample(size=size, mode='bilinear')

    h = img.shape[-1]
    remained = h % factor
    if remained > 0:
        img = F.pad(img, pad=(0, factor - remained, 0, factor - remained), value=0.5)
    s_crop = ceil(h / factor)
    n_crop = factor**2

    cropped = []
    for i in range(factor):
        for j in range(factor):
            h_loc = i * s_crop
            w_loc = j * s_crop
            cropped.append(img[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
    cropped = torch.cat(cropped)
    data_dec = resize(cropped)
    target_dec = torch.cat([target for _ in range(n_crop)])

    return data_dec, target_dec


def decode_zoom_multi(img, target, factor_max):
    data_multi = []
    target_multi = []
    for factor in range(1, factor_max + 1):
        decoded = decode_zoom(img, target, factor)
        data_multi.append(decoded[0])
        target_multi.append(decoded[1])

    return torch.cat(data_multi), torch.cat(target_multi)


def decode_fn(data, target, factor):
    if factor > 1:
        data, target = decode_zoom(data, target, factor)

    return data, target


def decode(args, data, target):
    """Decoding function for IDC
    """
    data_dec = []
    target_dec = []
    ipc = len(data) // args.nclass
    for c in range(args.nclass):
        idx_from = ipc * c
        idx_to = ipc * (c + 1)
        data_ = data[idx_from:idx_to].detach()
        target_ = target[idx_from:idx_to].detach()
        data_, target_ = decode_fn(data_, target_, args.factor)
        data_dec.append(data_)
        target_dec.append(target_)

    data_dec = torch.cat(data_dec)
    target_dec = torch.cat(target_dec)

    return data_dec, target_dec


def load_dataloader(args):
    """Load IDC condensed data
    """

    # Load condensed dataset
    if args.slct_type == 'idc':
        path_base = return_data_path(args)
        if args.nclass == 10:
            data, target = torch.load(os.path.join(f'{path_base}', 'data.pt'))
            print(f"Load data from {path_base}")

        elif args.nclass == 100:
            nclass_sub = 20
            data_all = []
            target_all = []
            for idx in range(args.nclass // nclass_sub):
                path = f'{path_base}_{nclass_sub}_phase{idx}'
                data, target = torch.load(os.path.join(path, 'data.pt'))
                data_all.append(data)
                target_all.append(target)
                print(f"Load data from {path}")

            data = torch.cat(data_all)
            target = torch.cat(target_all)
        print("Loaded condensed data: ", data.shape)

        if args.factor > 1:
            data, target = decode(args, data, target)
        print("Decoded data: ", data.shape)

        train_transform, _ = transform_imagenet(from_tensor=True, size=args.size)
        train_dataset = TensorDataset(data, target, train_transform)

    elif args.slct_type == 'random':
        traindir = os.path.join(args.imagenet_dir, 'train')

        train_transform, _ = transform_imagenet(from_tensor=False, size=args.size)
        train_dataset = ImageFolder(traindir,
                                    train_transform,
                                    nclass=args.nclass,
                                    slct_type=args.slct_type,
                                    ipc=args.ipc,
                                    load_memory=True)
        print(f"Test random selection {args.ipc} (total {len(train_dataset)})")

    train_loader = MultiEpochsDataLoader(train_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         num_workers=args.workers,
                                         persistent_workers=True)

    return train_loader
