import torch
from torch.utils.data import DataLoader, random_split

from Dataset import BodyFatCollageDataset
from model import FatModel
import cfg
import utils
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import ConcatDataset
import numpy as np
def blackout(image, **kwargs):
    return np.zeros_like(image)

train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=25, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.HueSaturationValue(p=0.2),
    A.GaussNoise(p=0.2),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    A.OneOf([
        A.Lambda(image=blackout),
        A.NoOp()
    ], p=0.25),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    ToTensorV2(),
])
MODEL_NAME = "efficientnet_b0"

def main():
    dataset1_train = BodyFatCollageDataset(cfg.DATASET_PATH, train_transform)
    dataset2_train = BodyFatCollageDataset(cfg.WOMEN, train_transform)
    dataset1_val = BodyFatCollageDataset(cfg.DATASET_PATH, val_transform)
    dataset2_val = BodyFatCollageDataset(cfg.WOMEN, val_transform)
    dataset_train_full = ConcatDataset([dataset1_train, dataset2_train])
    dataset_val_full = ConcatDataset([dataset1_val, dataset2_val])

    val_size = int(0.15 * len(dataset_train_full))
    train_size = len(dataset_train_full) - val_size

    train_ds, val_indices = random_split(dataset_train_full, [train_size, val_size])

    val_ds = torch.utils.data.Subset(dataset_val_full, val_indices.indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    model = FatModel().to(cfg.DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY
    )
    print(next(model.parameters()).device)

    loss_fn = torch.nn.L1Loss()
    if cfg.LOAD:
        utils.load_checkpoint(model,optimizer,cfg.LOAD_PATH,cfg.DEVICE)

    for epoch in range(cfg.EPOCHS):
        train_loss = utils.train_one_epoch(
            train_loader, model, optimizer, loss_fn, cfg.DEVICE
        )

        val_loss = utils.validate(
            val_loader, model, loss_fn, cfg.DEVICE
        )

        print(f"Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}")

        utils.save_epoch_checkpoint(model, optimizer,epoch,cfg.CHECKPOINT_DIR)
        utils.log_model_result(
            cfg.LOG_PATH,
            MODEL_NAME,
            epoch,
            val_loss
        )


if __name__ == "__main__":
    main()
