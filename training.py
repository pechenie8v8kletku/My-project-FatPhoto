import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

ALPHA=0.997
ST_SIZE=5
GAMMA_LR=0.1
LEARNING_RATE = (1e-3)/3
WEIGHT_DECAY=(1e-2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 400
NUM_WORKERS = 2
IMAGE_HEIGHT = 224# 1280 originally
IMAGE_WIDTH = 224# 1918 originally
PIN_MEMORY = True
LOAD_MODEL =True
TRAIN_IMG_DIR = "training_img"
TRAIN_MASK_DIR = "training_contours"
VAL_IMG_DIR = "valid_imgs"
VAL_MASK_DIR = "valid_contours"
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, y_true, y_pred):
        smooth = 1e-4
        intersection = torch.sum(y_true * y_pred)
        union = torch.sum(y_true) + torch.sum(y_pred)
        dice_coeff = 2*(intersection) / (union +smooth )
        return 1-dice_coeff
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, y_pred, y_true):
        dice = self.dice_loss(y_true, torch.sigmoid(y_pred))
        return dice*100


def train_fn(loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader)
    running_loss = 0.0
    num_batches = len(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.amp.autocast('cuda'):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

        running_loss += loss.item()

    # возвращаем среднее значение потерь за эпоху
    avg_loss = running_loss / num_batches
    return avg_loss
def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    #optimizer=torch.optim.RMSprop(model.parameters(),lr=LEARNING_RATE,alpha=ALPHA)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=1e-3/8 ,
        max_lr=(1e-3)/4,
        step_size_up=4,
        mode="triangular2"
    )

    #scheduler = StepLR(optimizer, step_size=ST_SIZE, gamma=GAMMA_LR)
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.amp.GradScaler("cuda")


    for epoch in range(NUM_EPOCHS):
        avg_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        scheduler.step(avg_loss)
        # save model
        if epoch%2==0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)
            save_predictions_as_imgs(
                val_loader, model, folder="saved_images/", device=DEVICE
            )
            check_accuracy(val_loader, model, device=DEVICE)




        # check accuracy


        # print some examples to a folder



if __name__ == "__main__":
    main()