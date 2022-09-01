from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

DATA_PATH = "/storage/janghyun/results/publication/icml22/results"

MEANS = {'cifar': [0.4914, 0.4822, 0.4465], 'imagenet': [0.485, 0.456, 0.406]}
STDS = {'cifar': [0.2023, 0.1994, 0.2010], 'imagenet': [0.229, 0.224, 0.225]}
MEANS['cifar10'] = MEANS['cifar']
STDS['cifar10'] = STDS['cifar']
MEANS['cifar100'] = MEANS['cifar']
STDS['cifar100'] = STDS['cifar']


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


def decode_fn(data, target, factor, decode_type='uniform'):
    if factor > 1:
        if decode_type == 'multi':
            data, target = decode_zoom_multi(data, target, factor)
        else:
            data, target = decode_zoom(data, target, factor)

    return data, target


def decode(data, target, factor, nclass, decode_type='uniform'):
    data_dec = []
    target_dec = []
    ipc = len(data) // nclass
    for c in range(nclass):
        idx_from = ipc * c
        idx_to = ipc * (c + 1)
        data_ = data[idx_from:idx_to].detach()
        target_ = target[idx_from:idx_to].detach()
        data_, target_ = decode_fn(data_, target_, factor, decode_type)
        data_dec.append(data_)
        target_dec.append(target_)

    data_dec = torch.cat(data_dec)
    target_dec = torch.cat(target_dec)

    return data_dec, target_dec


class IDCDataLoader:
    def __init__(self, factor=2):
        self.factor = factor

    def load_data(self, root_dir, dataset, ipc, data_file):
        train_images, train_labels = torch.load(os.path.join(root_dir, data_file))

        train_images, train_labels = self.decode(train_images, train_labels, dataset)
        train_images = self.normalize(train_images, dataset)

        return train_images, train_labels

    @staticmethod
    def get_data_file_name(method, dataset, ipc, factor):
        if factor > 1:
            init = 'mix'
        else:
            init = 'random'

        name = f'{dataset.lower()}/conv3in_grad_mse_nd2000_cut_niter2000_factor{factor}_lr0.005_{init}'
        path = os.path.join(DATA_PATH, f'{name}_ipc{ipc}', 'data.pt')
        return path

    @staticmethod
    def normalize(images, dataset):
        means = torch.tensor(MEANS[dataset.lower()]).reshape(1, 3, 1, 1)
        stds = torch.tensor(STDS[dataset.lower()]).reshape(1, 3, 1, 1)

        images = (images - means) / stds
        return images

    def decode(self, images, target, dataset):
        if dataset.lower() == 'cifar10':
            nclass = 10
        elif dataset.lower() == 'cifar100':
            nclass = 100
        else:
            raise AssertionError("Not implemented dataset!")

        if self.factor > 1:
            images, target = decode(images, target, self.factor, nclass)

        return images, target
