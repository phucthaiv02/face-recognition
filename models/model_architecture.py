import torch
from torch import nn
from facenet_pytorch import InceptionResnetV1


class classifyModel(nn.Module):
    def __init__(self, num_classes):
        super(classifyModel, self).__init__()
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.fc = nn.Sequential(
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        with torch.no_grad():
            embeddings = self.resnet(x)

        output = self.fc(embeddings)

        return output
