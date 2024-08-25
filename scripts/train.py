import torch


def train(model, train_loader, val_loader, epochs, criterion, optimizer):
    history = {'train_loss': [], 'val_loss': []}
    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}/{epochs}: ', end='')
        model.train()
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss = val(model, val_loader, criterion)
        print(f'train_loss: {train_loss}, val_loss: {val_loss}')

        torch.save(model, f'./output/checkpoints/ckp{epoch}.pth',)
    torch.save(model, f'output/model/model.pth')

    return model, history


def val(model, val_loader, criterion):
    model.eval()

    val_loss = 0.0
    for i, (images, labels) in enumerate(val_loader):
        output = model(images)
        loss = criterion(output, labels)
        val_loss += loss.item()

    val_loss /= len(val_loader)
    return val_loss
