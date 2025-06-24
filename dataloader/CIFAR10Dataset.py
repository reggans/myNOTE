import os

import torch.utils.data
from numpy.random import permutation
from torchvision import transforms
import numpy as np

class CIFAR10Dataset(torch.utils.data.Dataset):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2471, 0.2435, 0.2616]
    num_classes = 10

    def __init__(self, file_path, domains, transform='src', distribution='real', dir_beta=0.1):
        super(CIFAR10Dataset, self).__init__()
        if distribution not in ['real', 'random', 'dirichlet']:
            raise NotImplementedError

        self.file_path = file_path
        self.domains = domains
        self.distribution = distribution
        self.dir_beta = dir_beta

        self.features = []
        self.class_labels = []
        self.domain_labels = []

        for i, domain in enumerate(self.domains):
            data_path, label_path = self.get_filepaths(domain)
            data = np.load(data_path)
            data = data.astype('float32')  / 255.0
            data = data.transpose((0, 3, 1, 2)) # To (B, C, H, W)

            self.features.append(torch.from_numpy(data))
            self.class_labels.append(torch.from_numpy(np.load(label_path)).long())
            self.domain_labels.append(torch.Tensor([i for _ in range(len(data))]).long())

        self.features = torch.cat(self.features)
        self.class_labels = torch.cat(self.class_labels)
        self.domain_labels = torch.cat(self.domain_labels)

        self.shuffle_distribution()

        self.dataset = torch.utils.data.TensorDataset(
            self.features,
            self.class_labels,
            self.domain_labels,
        )

        if transform == 'src':
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
        elif transform == 'tgt':
            self.transform = transforms.Compose([
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
        else:
            raise NotImplementedError

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

    def shuffle_distribution(self):
        # Originally there's a config variable n_samples, skipped here
        if self.distribution == 'real':
            return

        rng = np.random.default_rng()
        permutation = rng.permutation(len(self.class_labels))

        self.features = self.features[permutation]
        self.class_labels = self.class_labels[permutation]
        self.domain_labels = self.domain_labels[permutation]

        if self.distribution == 'dirichlet':
            numchunks = self.num_classes
            min_size = -1
            N = len(self.features)
            min_size_thresh = 10
            features = []
            cl_labels = []
            do_labels = []
            while min_size < min_size_thresh:
                idx_batch = [[] for _ in range(numchunks)]
                idx_batch_cls = [[] for _ in range(numchunks)]
                for k in range(numchunks):
                    cl_labels = self.class_labels.numpy()
                    idx_k = np.where(cl_labels == k)[0]

                    proportions = rng.dirichlet(np.repeat(self.dir_beta, numchunks))
                    proportions /= proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)
                    idx_batch = [elem + idx.tolist() for elem, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(elem) for elem in idx_batch])

                    for elem, idx in zip(idx_batch_cls, np.split(idx_k, proportions)):
                        elem.append(idx)

            for chunk in idx_batch_cls:
                cls_seq = list(range(self.num_classes))
                rng.shuffle(cls_seq)
                for cls in cls_seq:
                    idx = chunk[cls]
                    features.extend(self.features[idx])
                    cl_labels.extend(self.domain_labels[idx])
                    do_labels.extend(self.domain_labels[idx])

            self.features = torch.stack(features)
            self.domain_labels = torch.LongTensor(do_labels)
            self.class_labels = torch.LongTensor(cl_labels)