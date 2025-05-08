import torch
from torch.utils.data import Dataset
import random

class PairDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        img1, label1 = self.dataset[idx]
        should_match = random.randint(0, 1)

        while True:
            idx2 = random.randint(0, len(self.dataset) - 1)
            img2, label2 = self.dataset[idx2]
            if (label1 == label2) == should_match:
                break

        return img1, img2, torch.tensor(should_match, dtype=torch.float32)

    def __len__(self):
        return len(self.dataset)