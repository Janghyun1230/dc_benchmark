import os
import torch


class RandomDataLoader:
    @staticmethod
    def load_data(root_dir, dataset, ipc, data_file):
        image_path = data_file[0]
        label_path = data_file[1]
        image_path = os.path.join(root_dir, "random", image_path)
        label_path = os.path.join(root_dir, "random", label_path)
        training_images = torch.load(image_path)
        training_lables = torch.load(label_path).long()
        return training_images, training_lables

    @staticmethod
    def get_data_file_name(method, dataset, ipc):
        return ('%s_IPC%d_normalize_images.pt' % (dataset, ipc),
                '%s_IPC%d_normalize_labels.pt' % (dataset, ipc))
