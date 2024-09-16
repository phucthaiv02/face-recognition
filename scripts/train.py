import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(model, train_loader, val_loader, epochs, criterion, optimizer, scheduler=None):
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}/{epochs}: ', end='')
        model.train()
        train_loss = 0.0
        for idx, batch in enumerate(train_loader):
            images1 = batch[0].to(device)
            images2 = batch[1].to(device)
            labels = batch[2].to(device)
            optimizer.zero_grad()

            output = model(images1, images2)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss = val(model, val_loader, criterion)
        print(f'train_loss: {train_loss}, val_loss: {val_loss}')

        if scheduler:
            scheduler.step(val_loss)
            print(f"\tLearning rate: {scheduler.get_last_lr()}")

        torch.save(model, f'./output/checkpoints/ckp{epoch}.pth',)
    torch.save(model, f'output/model/model.pth')

    return model, history


def val(model, val_loader, criterion):
    model.eval()

    val_loss = 0.0
    for idx, batch in enumerate(val_loader):
        images1 = batch[0].to(device)
        images2 = batch[1].to(device)
        labels = batch[2].to(device)
        output = model(images1, images2)
        loss = criterion(output, labels)
        val_loss += loss.item()

    val_loss /= len(val_loader)
    return val_loss
