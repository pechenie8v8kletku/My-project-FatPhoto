import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class BodyFatCollageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for label_name in os.listdir(root_dir):
            full = os.path.join(root_dir, label_name)
            if not os.path.isdir(full):
                continue
            label = float(label_name)
            for fname in os.listdir(full):
                if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.samples.append((os.path.join(full, fname), label))

    def __len__(self):
        return len(self.samples)

    def _split_into_quadrants(self, img: Image.Image):
        w, h = img.size
        w2, h2 = w // 2, h // 2

        quadrants = [
            img.crop((0,    0,    w2, h2)),
            img.crop((w2,   0,    w,  h2)),
            img.crop((0,    h2,   w2, h)),
            img.crop((w2,   h2,   w,  h)),
        ]
        return quadrants

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        collage = Image.open(img_path).convert("RGB")
        quads = self._split_into_quadrants(collage)

        if self.transform:
            quads = [self.transform(image=np.array(q))["image"] for q in quads]
        else:
            quads = [
                torch.from_numpy(np.array(q)).permute(2, 0, 1).float() / 255
                for q in quads
            ]
        batch = torch.stack(quads)

        return batch, torch.tensor(label, dtype=torch.float32)
'''
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.Resize(224, 224),

    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),

    A.RandomBrightnessContrast(p=0.3),
    A.HueSaturationValue(p=0.2),

    A.GaussNoise(p=0.2),

    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    ToTensorV2(),
])


Path="different_datasets/male_ai_generated"
Dataset=BodyFatCollageDataset(Path,train_transform)
x,y=Dataset[len(Dataset)-1]
print(y)
print(x.shape)
'''