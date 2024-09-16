import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from models.model_architecture import FaceNet, SiameseNetwork
from scripts.train import train
from scripts.dataset import get_dataset

from PIL import Image

DATA_DIR = 'data/data.txt'
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_WORKERS = 8
NUM_CLASSES = 3

torch.manual_seed(12)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"{device} is available")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    train_set, val_set = get_dataset(DATA_DIR, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=BATCH_SIZE)

    facenet = FaceNet().to(device)
    siamese = SiameseNetwork().to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(siamese.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer, 'min', patience=2, factor=0.1)

    train(model=siamese,
          train_loader=train_loader,
          val_loader=val_loader,
          epochs=EPOCHS,
          criterion=criterion,
          optimizer=optimizer,
          scheduler=scheduler)
