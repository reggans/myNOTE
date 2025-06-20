import os

import torch.utils.data
import numpy as np

class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, config):
        super(CIFAR10Dataset, self).__init__()

        self.file_path = config['file_path']
        self.domains = config['domains']
        self.img_size = config['img_size']

        self.features = []
        self.class_labels = []
        self.domain_labels = []
        self.transform = None

        for i, domain in enumerate(self.domains):
            data_path, label_path = self.get_filepaths(domain)
            data = np.load(data_path)
            data = data.astype('float32')  / 255.0
            
            self.features.append(torch.from_numpy(data))
            self.class_labels.append(torch.from_numpy(np.load(label_path)).long())
            self.domain_labels.append(torch.Tensor(i for _ in range(len(data))).long())

        self.features = torch.cat(self.features)
        self.class_labels = torch.cat(self.class_labels)
        self.domain_labels = torch.cat(self.domain_labels)

        self.dataset = torch.utils.data.TensorDataset(
            self.features,
            self.class_labels,
            self.domain_labels,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, cl, dl = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, cl, dl

    def get_filepaths(self, domain):
        if domain.startswith('original'):
            sub_path1 = 'origin'
            sub_path2 = ''
            data_filename = 'original.npy'
            label_filename = 'labels.npy'
        elif domain.startswith('test'):
            sub_path1 = 'corrupted'
            sub_path2 = 'severity-1'
            data_filename = 'test.npy'
            label_filename = 'labels.npy'
        else:
            sub_path1 = 'corrupted'
            sub_path2 = 'severity-' + domain.split('-')[1]
            data_filename = domain.split('-')[0] + '.npy'
            label_filename = 'labels.npy'

        data_path = os.path.join(self.file_path, sub_path1, sub_path2, data_filename)
        label_path = os.path.join(self.file_path, sub_path1, sub_path2, label_filename)
        return data_path, label_path