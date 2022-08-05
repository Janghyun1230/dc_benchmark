import os
import torch

class DMDataLoader:

    @staticmethod
    def load_data(root_dir, dataset, ipc, data_file):
        data_path = os.path.join(root_dir, "DM", dataset, 'IPC' + str(ipc), data_file)
        dm_data = torch.load(data_path)
        training_data = dm_data['data']
        train_images, train_labels = training_data[-1]
        return train_images, train_labels