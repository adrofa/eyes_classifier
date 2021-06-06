from utils.support import jsn_dump, pkl_dump, pkl_load
from train.versions.augmentation import get_augmentation
from train.versions.model import get_model
from train.versions.optimizer import get_optimizer
from train.versions.criterion import get_criterion
from train.versions.scheduler import get_scheduler

import sys
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import pandas as pd
from matplotlib import pyplot as plt


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class MyDataset(Dataset):
    def __init__(self, identity_df, transform=None):
        self.transform = A.Compose([transform, ToTensorV2(p=1)])
        self.images = []
        self.labels = []
        for _, row in identity_df.iterrows():
            img = cv2.imread(str(row["cew_img"]), cv2.IMREAD_GRAYSCALE)
            img = np.expand_dims(img, 2)
            self.images.append(img)
            self.labels.append(row["label"])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        transformed = self.transform(image=self.images[idx], label=self.labels[idx])
        return transformed["image"].type(torch.FloatTensor), float(transformed["label"])


def train(model, dataloader, criterion, optimizer, logit_ths=0, device="cuda", verbose="train"):
    """Train model 1 epoch.
    Returns:
        progress_dct (dict): dct with loss and accuracy
    """
    model.train()
    with tqdm(dataloader, desc=verbose, file=sys.stdout, disable=not verbose) as iterator:
        progress = {
            "loss": 0,
            "accuracy": 0,
        }
        items_epoch = 0
        for x, y in iterator:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model.forward(x)

            loss_batch = criterion(pred, y)
            loss_batch.backward()
            optimizer.step()

            progress["loss"] += loss_batch.item() * pred.shape[0]
            progress["accuracy"] += ((pred > logit_ths) == y).sum().item()
            items_epoch += pred.shape[0]

            if verbose:
                iterator.set_postfix_str(" | ".join(
                    [f"{i}: {(progress[i] / items_epoch):.5f}" for i in progress]
                ))

    progress_dct = {i: progress[i] / items_epoch for i in progress}
    return progress_dct


def valid(model, dataloader, criterion, logit_ths=0, device="cuda", verbose="valid"):
    """Train model 1 epoch.
    Returns:
        progress_dct (dict): dct with loss and accuracy
    """
    model.eval()
    with torch.no_grad():
        with tqdm(dataloader, desc=verbose, file=sys.stdout, disable=not verbose) as iterator:
            progress = {
                "loss": 0,
                "accuracy": 0,
            }
            items_epoch = 0
            for x, y in iterator:
                x, y = x.to(device), y.to(device)
                pred = model.forward(x)

                loss_batch = criterion(pred, y)

                progress["loss"] += loss_batch.item() * pred.shape[0]
                progress["accuracy"] += ((pred > logit_ths) == y).sum().item()
                items_epoch += pred.shape[0]

                if verbose:
                    iterator.set_postfix_str(" | ".join(
                        [f"{i}: {(progress[i] / items_epoch):.5f}" for i in progress]
                    ))

    progress_dct = {i: progress[i] / items_epoch for i in progress}
    return progress_dct


def progress_chart(progress_df, chart_path):
    fig, ax1 = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('white')

    # title
    best_epoch = progress_df.loc[progress_df["valid_loss"].idxmin()]
    fig.suptitle(" | ".join([
        f"Best valid-loss: {best_epoch.valid_loss:.3}",
        f"accuracy: {best_epoch.valid_accuracy:.3}",
        f"epoch: {best_epoch.name}",
    ]))

    # loss
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.plot(progress_df["train_loss"], color="blue", label="train-loss")
    ax1.plot(progress_df["valid_loss"], color="red", label="valid-loss")
    ax1.legend(loc="center left")

    # accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('accuracy')
    ax2.plot(progress_df["train_accuracy"], color="green", label="train-accuracy")
    ax2.plot(progress_df["valid_accuracy"], color="orange", label="valid-accuracy")
    ax2.legend(loc="center right")

    fig.tight_layout()
    plt.savefig(chart_path)
    plt.close()


def main(cfg):
    results_dir = Path(cfg["output"]) / "models" / f"hypothesis-{str(cfg['hypothesis'])}" / f"fold-{str(cfg['fold'])}"
    try:
        os.makedirs(results_dir, exist_ok=True if cfg["hypothesis"] == "debug" else False)
    except:
        raise Exception(f"v{cfg['version']} fold-{str(cfg['fold'])} exists!")
    jsn_dump(cfg, results_dir / "config.json")

    seed_everything(cfg["seed"])

    # dataset_df
    dataset_df = pkl_load(Path(cfg["output"]) / "crossval_split" / "crossval_dct.pkl")
    train_df = dataset_df[cfg["fold"]]["train"]
    valid_df = dataset_df[cfg["fold"]]["valid"]

    # PyTorch Datasets initialization
    augmentation = get_augmentation(cfg["augmentation_version"])
    dataset = {
        "train": MyDataset(train_df, augmentation["train"]),
        "train_valid": MyDataset(train_df, augmentation["valid"]),
        "valid": MyDataset(valid_df, augmentation["valid"])
    }

    # PyTorch DataLoaders initialization
    dataloader = {
        "train": DataLoader(
            dataset["train"], batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["n_jobs"]
        ),
        "train_valid": DataLoader(
            dataset["train_valid"], batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["n_jobs"]
        ),
        "valid": DataLoader(
            dataset["valid"], batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["n_jobs"]
        ),
    }

    # model
    model = get_model(cfg["model_version"], cfg["model_weights"])
    model.to(cfg["device"])

    # optimizer
    optimizer = get_optimizer(cfg["optimizer_version"], model.parameters(), cfg["optimizer_weights"])

    # scheduler
    if cfg["scheduler_version"]:
        scheduler = get_scheduler(cfg["scheduler_version"], optimizer)

    # loss function
    criterion = get_criterion(cfg["criterion_version"])

    # EPOCHS
    progress = []
    loss_min = None
    accuracy_max = None
    epochs_without_improvement = 0
    for epoch in range(cfg["epoch_num"]):
        print(f"Epoch-{epoch}", file=sys.stdout)

        # train
        progress_train = train(model, dataloader["train"], criterion, optimizer, logit_ths=cfg["logit_ths"],
                               device=cfg["device"], verbose="train")
        # train loss w/o dropout
        progress_train_valid = valid(model, dataloader["train_valid"], criterion, logit_ths=cfg["logit_ths"],
                                     device=cfg["device"], verbose="train")

        # LR scheduler
        if cfg["scheduler_version"]:
            if cfg["scheduler_version"].split("_")[0] == "rop":
                scheduler.step(progress_train_valid["loss"])
            else:
                raise Exception("Unknown scheduler!")

        # validation
        progress_valid = valid(model, dataloader["valid"], criterion, logit_ths=cfg["logit_ths"],
                               device=cfg["device"], verbose="valid")

        # saving progress info
        progress_epoch = {"raw_"+i: progress_train[i] for i in progress_train}
        progress_epoch.update({"train_"+i: progress_train_valid[i] for i in progress_train_valid})
        progress_epoch.update({"valid_" + i: progress_valid[i] for i in progress_valid})
        progress.append(progress_epoch)
        progress_df = pd.DataFrame(progress)
        pkl_dump(progress_df, results_dir / "progress.pkl")
        progress_chart(progress_df, results_dir / "progress.png")

        # saving the model's weights (of the model with the lowest loss)
        if loss_min is None or progress_valid["loss"] < loss_min:
            loss_min = progress_valid["loss"]
            epochs_without_improvement = 0

            torch.save(model.state_dict(), results_dir / "model.pt")
            torch.save(optimizer.state_dict(), results_dir / "optimizer.pt")
        else:
            epochs_without_improvement += 1

        # saving the model's weights (of the model with the highest accuracy)
        if accuracy_max is None or progress_valid["accuracy"] > accuracy_max:
            accuracy_max = progress_valid["accuracy"]
            torch.save(model.state_dict(), results_dir / "model_best_accuracy.pt")
            torch.save(optimizer.state_dict(), results_dir / "model_best_accuracy.pt")

        # Logs
        print(
            "\t".join([
                f"Best valid loss: {loss_min:.5}",
                f"Best valid accuracy: {accuracy_max:.5}",
            ]),
            "\n" + "-" * 70,

            file=sys.stdout
        )

        # early stopping (by loss)
        if epochs_without_improvement >= cfg["early_stopping"]:
            print("EARLY STOPPING!")
            break


if __name__ == "__main__":
    config = {
        "hypothesis": "debug",
        "fold": 1,

        "model_version": 4,
        "model_weights": None,
        "optimizer_version": "adam_4",
        "optimizer_weights": None,
        "scheduler_version": "rop_1",

        "augmentation_version": 1,
        "criterion_version": 1,
        "logit_ths": 0,

        "output": "../output",
        "crossval_dct": "../output/crossval",

        "epoch_num": 1000,
        "early_stopping": 30,

        "device": "cuda",
        "batch_size": 800,
        "n_jobs": 4,
        "seed": 0,
    }
    main(config)
