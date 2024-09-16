import torch
from torch import nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1


class FaceNet(nn.Module):
    def __init__(self):
        super(FaceNet, self).__init__()
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

    def forward(self, x):
        with torch.no_grad():
            embeddings = self.resnet(x)

        return embeddings


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = InceptionResnetV1(pretrained='vggface2')
        self.fc = nn.Sequential(
            nn.Linear(2 * 512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input1, input2):
        with torch.no_grad():
            embedding1 = self.resnet(input1)
            embedding2 = self.resnet(input2)

        combined_embeddings = torch.cat((embedding1, embedding2), dim=1)

        output = self.fc(combined_embeddings)
        output = output.squeeze(1)
        return output
