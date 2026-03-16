import os
import torch
from tqdm import tqdm


def train_one_epoch(loader, model, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    loop = tqdm(loader, leave=False)
    for x, y in loop:
        x = x.to(device)
        y = y.unsqueeze(1).to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)


@torch.no_grad()
def validate(loader, model, loss_fn, device):
    model.eval()
    total_loss = 0


    for x, y in loader:
        x = x.to(device)
        y = y.unsqueeze(1).to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        total_loss += loss.item()


    return total_loss / len(loader)

def save_epoch_checkpoint(model, optimizer, epoch, path):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"epoch_{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, file_path)
def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])


import pandas as pd
import os
def log_model_result(csv_path, model_name, epoch, value):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=["epoch"])
    if epoch not in df["epoch"].values:
        df = pd.concat(
            [df, pd.DataFrame([{"epoch": epoch}])],
            ignore_index=True
        )
    if model_name not in df.columns:
        df[model_name] = None
    df.loc[df["epoch"] == epoch, model_name] = value
    df = df.sort_values("epoch")
    df.to_csv(csv_path, index=False)
