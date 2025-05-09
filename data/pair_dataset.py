import random
from torch.utils.data import Dataset
from PIL import Image
import torch


class SiameseDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.targets = dataset.targets

        if isinstance(self.targets, list):
            # Convert to tensor if needed (CIFAR-10)
            self.targets = torch.tensor(self.targets)

        self.label_to_indices = {}
        for idx, label in enumerate(self.targets):
            label = int(label)
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

    def __getitem__(self, index):
        img1, label1 = self.dataset[index]
        should_get_same_class = random.randint(0, 1)

        if should_get_same_class:
            idx2 = random.choice(self.label_to_indices[int(label1)])
        else:
            different_labels = list(set(self.label_to_indices.keys()) - {int(label1)})
            label2 = random.choice(different_labels)
            idx2 = random.choice(self.label_to_indices[label2])

        img2, label2 = self.dataset[idx2]
        label = torch.tensor([int(label1 == label2)], dtype=torch.float32)

        return img1, img2, label

    def __len__(self):
        return len(self.dataset)
