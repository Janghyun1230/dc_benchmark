# Codes borrowed from https://github.com/snu-mllab/Efficient-Dataset-Condensation

import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
MEANS = {'imagenet': [0.485, 0.456, 0.406]}
STDS = {'imagenet': [0.229, 0.224, 0.225]}

CLASS100_DIR = './class100.txt'


class TensorDataset(torch.utils.data.Dataset):

    def __init__(self, images, labels, transform=None):
        # images: NxCxHxW tensor
        self.images = images.detach().cpu().float()
        self.targets = labels.detach().cpu()
        self.transform = transform

    def __getitem__(self, index):
        sample = self.images[index]
        if self.transform != None:
            sample = self.transform(sample)

        target = self.targets[index]
        return sample, target

    def __len__(self):
        return self.images.shape[0]


class ImageFolder(datasets.DatasetFolder):
    """ Dataset with subclasses
    """

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=datasets.folder.default_loader,
                 is_valid_file=None,
                 load_memory=False,
                 load_transform=None,
                 nclass=100,
                 phase=0,
                 slct_type='random',
                 ipc=-1,
                 seed=-1):
        self.extensions = IMG_EXTENSIONS if is_valid_file is None else None
        super(ImageFolder, self).__init__(root,
                                          loader,
                                          self.extensions,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)

        # Override
        if nclass < 1000:
            self.classes, self.class_to_idx = self.find_subclasses(nclass=nclass,
                                                                   phase=phase,
                                                                   seed=seed)
        else:
            self.classes, self.class_to_idx = self.find_classes(self.root)
        self.nclass = nclass
        self.samples = datasets.folder.make_dataset(self.root, self.class_to_idx, self.extensions,
                                                    is_valid_file)

        if ipc > 0:
            self.samples = self._subset(slct_type=slct_type, ipc=ipc)

        self.targets = [s[1] for s in self.samples]
        self.load_memory = load_memory
        self.load_transform = load_transform
        if self.load_memory:
            self.imgs = self._load_images(load_transform)
        else:
            self.imgs = self.samples

    def find_subclasses(self, nclass=100, phase=0, seed=-1):
        """Finds the class folders in a dataset.
        """
        classes = []
        phase = max(0, phase)
        cls_from = nclass * phase
        cls_to = nclass * (phase + 1)
        if seed == -1:
            with open(CLASS100_DIR, 'r') as f:
                class_name = f.readlines()
            for c in class_name:
                c = c.split('\n')[0]
                classes.append(c)
            classes = classes[cls_from:cls_to]
        else:
            np.random.seed(seed)
            class_indices = np.random.permutation(len(self.classes))[cls_from:cls_to]
            for i in class_indices:
                classes.append(self.classes[i])

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        assert len(classes) == nclass

        return classes, class_to_idx

    def _subset(self, slct_type='random', ipc=10):
        n = len(self.samples)
        idx_class = [[] for _ in range(self.nclass)]
        for i in range(n):
            label = self.samples[i][1]
            idx_class[label].append(i)

        min_class = np.array([len(idx_class[c]) for c in range(self.nclass)]).min()
        print("# examples in the smallest class: ", min_class)
        assert ipc < min_class

        if slct_type == 'random':
            indices = np.arange(n)
        else:
            raise AssertionError(f'selection type does not exist!')

        samples_subset = []
        idx_class_slct = [[] for _ in range(self.nclass)]
        for i in indices:
            label = self.samples[i][1]
            if len(idx_class_slct[label]) < ipc:
                idx_class_slct[label].append(i)
                samples_subset.append(self.samples[i])

            if len(samples_subset) == ipc * self.nclass:
                break

        return samples_subset

    def _load_images(self, transform=None):
        """Load images on memory
        """
        imgs = []
        for i, (path, _) in enumerate(self.samples):
            sample = self.loader(path)
            if transform != None:
                sample = transform(sample)
            imgs.append(sample)
            if i % 100 == 0:
                print(f"Image loading.. {i}/{len(self.samples)}", end='\r')

        print(" " * 50, end='\r')
        return imgs

    def __getitem__(self, index):
        if not self.load_memory:
            path = self.samples[index][0]
            sample = self.loader(path)
        else:
            sample = self.imgs[index]

        target = self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

    def __len__(self):
        return len(self.sampler)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    """Multi epochs data loader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()  # Init iterator and sampler once

        self.convert = None
        if self.dataset[0][0].dtype == torch.uint8:
            self.convert = transforms.ConvertImageDtype(torch.float)

        if self.dataset[0][0].device == torch.device('cpu'):
            self.device = 'cpu'
        else:
            self.device = 'cuda'

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        for i in range(len(self)):
            data, target = next(self.iterator)
            if self.convert != None:
                data = self.convert(data)
            yield data, target


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec, device='cpu'):
        self.alphastd = alphastd
        self.eigval = torch.tensor(eigval, device=device)
        self.eigvec = torch.tensor(eigvec, device=device)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        # make differentiable
        if len(img.shape) == 4:
            return img + rgb.view(1, 3, 1, 1).expand_as(img)
        else:
            return img + rgb.view(3, 1, 1).expand_as(img)


def transform_imagenet(size=224, from_tensor=False, normalize=True):

    # this is consistent to the preprocessing in the condensation stage.
    resize_train = [transforms.Resize(size), transforms.CenterCrop(size)]
    resize_test = [transforms.Resize(size), transforms.CenterCrop(size)]

    jittering = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
    lighting = Lighting(alphastd=0.1,
                        eigval=[0.2175, 0.0188, 0.0045],
                        eigvec=[
                            [-0.5675, 0.7192, 0.4009],
                            [-0.5808, -0.0045, -0.8140],
                            [-0.5836, -0.6948, 0.4203],
                        ])
    aug = [transforms.RandomHorizontalFlip(), jittering, lighting]

    rrc_fn = transforms.RandomResizedCrop(size, scale=(0.5, 1.0))
    aug = [rrc_fn] + aug

    if from_tensor:
        cast = []
    else:
        cast = [transforms.ToTensor()]

    if normalize:
        normal_fn = [transforms.Normalize(mean=MEANS['imagenet'], std=STDS['imagenet'])]
    else:
        normal_fn = []

    train_transform = transforms.Compose(resize_train + cast + aug + normal_fn)
    test_transform = transforms.Compose(resize_test + cast + normal_fn)

    return train_transform, test_transform
