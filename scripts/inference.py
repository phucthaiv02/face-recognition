from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

import glob
import os
# device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = [os.path.join(image_dir, img)
                            for img in os.listdir(image_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def detect_faces(image):
    device = 'cpu'
    mtcnn = MTCNN(thresholds=[0.7, 0.7, 0.8], keep_all=True, device=device)

    # Detect faces and crop them
    boxes, _ = mtcnn.detect(image)
    draw = ImageDraw.Draw(image)
    faces = []
    for idx, box in enumerate(boxes):
        bbox = list(map(int, box.tolist()))
        faces.append(image.crop(bbox))

    for idx, box in enumerate(boxes):
        bbox = list(map(int, box.tolist()))
        tbox = (bbox[0] + 5, bbox[1] + 5)
        draw.rectangle(bbox, outline="red", width=3)
        draw.text(tbox, str(idx),
                  font=ImageFont.truetype("arial.ttf", 40),
                  fill='red')
    return image, faces


def indentify_faces(model, faces):
    ref = {}
    result = 'Unknown'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    image_tensor = transform(faces.convert('RGB')).unsqueeze(0).to('cuda')
    input = image_tensor.repeat(16, 1, 1, 1)
    for key, value in ref.items():
        test_set = TestDataset(os.path.join(
            'data/processed/', key), transform=transform)

        data_loader = DataLoader(
            test_set, batch_size=16, shuffle=True, drop_last=True)
        scores = []
        for batch in data_loader:
            batch = batch.to('cuda')
            output = model(input, batch)
            scores.extend(output.detach().cpu().numpy())
        avg_score = sum(scores)/len(scores)
        if avg_score < 0.1:
            result = value

    return result
