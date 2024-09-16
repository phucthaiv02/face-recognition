import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, random_split
from PIL import Image


class SiameseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super(SiameseDataset, self).__init__()

        self.image1_paths = []
        self.image2_paths = []
        self.labels = []
        self.transform = transform
        with open(data_dir, 'r') as f:
            for line in f.readlines():
                image1, image2, label = line.strip().split()
                self.image1_paths.append(image1)
                self.image2_paths.append(image2)
                self.labels.append(int(label))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image1 = Image.open(self.image1_paths[index]).convert('RGB')
        image2 = Image.open(self.image2_paths[index]).convert('RGB')

        image1 = self.transform(image1)
        image2 = self.transform(image2)

        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.float32)

        return image1, image2, label


def get_dataset(data_dir, transform):
    dataset = SiameseDataset(data_dir, transform=transform)

    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    return train_set, val_set
