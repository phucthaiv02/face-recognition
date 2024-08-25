import torch
from PIL import Image
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from scripts.train import train
from models.model_architecture import classifyModel

DATA_DIR = 'data/processed/'
EPOCHS = 20
BATCH_SIZE = 1
LEARNING_RATE = 0.001
NUM_WORKERS = 8
NUM_CLASSES = 3

torch.manual_seed(12)

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS)

    c = [0, 0, 0]
    for images, labels in train_loader:
        c[labels[0].numpy()] += 1
    print(c)

# model = classifyModel(num_classes=NUM_CLASSES)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# model, history = train(model, train_loader, val_loader,
#                        epochs=EPOCHS, criterion=criterion, optimizer=optimizer)
