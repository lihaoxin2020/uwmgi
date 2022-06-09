import copy
import gc
import time
from collections import defaultdict

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedGroupKFold
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset
import utils

import wandb

import warnings
warnings.filterwarnings("ignore")

from colorama import Fore, Back, Style

c_ = Fore.GREEN
sr_ = Style.RESET_ALL


class CFG:
    seed = 42
    debug = False  # set debug=False for Full Training
    exp_name = 'Baselinev2'
    comment = 'unet-efficientnet_b1-224x224-aug2-split2'
    model_name = 'Unet'
    backbone = 'efficientnet-b1'
    train_bs = 96
    valid_bs = train_bs * 2
    img_size = [224, 224]
    epochs = 15
    lr = 1e-3
    scheduler = 'CosineAnnealingLR'
    min_lr = 1e-6
    T_max = int(30000 / train_bs * epochs) + 50
    T_0 = 25
    warmup_epochs = 1
    wd = 1e-6
    n_accumulate = max(1, 32 // train_bs)
    n_fold = 5
    n_sample = None
    num_classes = 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ref: https://www.kaggle.com/code/awsaf49/uwmgi-unet-train-pytorch/#%F0%9F%9A%85-Training
def prepare_loaders(df, fold, debug=False):
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)
    if debug:
        train_df = train_df.head(32 * 5).query("empty==0")
        valid_df = valid_df.head(32 * 3).query("empty==0")
    train_dataset = dataset.BuildDataset(train_df, transforms=data_transforms['train'])
    valid_dataset = dataset.BuildDataset(valid_df, transforms=data_transforms['valid'])

    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs if not debug else 20,
                              num_workers=4, shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs if not debug else 20,
                              num_workers=4, shuffle=False, pin_memory=True)

    return train_loader, valid_loader


# ref: https://www.kaggle.com/code/awsaf49/uwmgi-unet-train-pytorch/#%F0%9F%9A%85-Training
def build_model():
    model = smp.Unet(
        encoder_name=CFG.backbone,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=CFG.num_classes,  # model output channels (number of classes in your dataset)
        activation=None,
    )
    model.to(CFG.device)
    return model


# ref: https://www.kaggle.com/code/awsaf49/uwmgi-unet-train-pytorch/#%F0%9F%9A%85-Training
def load_model(path):
    model = build_model()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


JaccardLoss = smp.losses.JaccardLoss(mode='multilabel')
DiceLoss = smp.losses.DiceLoss(mode='multilabel')
BCELoss = smp.losses.SoftBCEWithLogitsLoss()
LovaszLoss = smp.losses.LovaszLoss(mode='multilabel', per_image=False)
TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)


# ref: https://www.kaggle.com/code/awsaf49/uwmgi-unet-train-pytorch/#%F0%9F%9A%85-Training
def dice_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=(1, 0))
    return dice


# ref: https://www.kaggle.com/code/awsaf49/uwmgi-unet-train-pytorch/#%F0%9F%9A%85-Training
def iou_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)
    iou = ((inter + epsilon) / (union + epsilon)).mean(dim=(1, 0))
    return iou


# ref: https://www.kaggle.com/code/awsaf49/uwmgi-unet-train-pytorch/#%F0%9F%9A%85-Training
def criterion(y_pred, y_true):
    return 0.5 * BCELoss(y_pred, y_true) + 0.5 * TverskyLoss(y_pred, y_true)


# ref: https://www.kaggle.com/code/awsaf49/uwmgi-unet-train-pytorch/#%F0%9F%9A%85-Training
def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    scaler = amp.GradScaler()

    dataset_size = 0
    running_loss = 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')
    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        batch_size = images.size(0)

        with amp.autocast(enabled=True):
            y_pred = model(images)
            loss = criterion(y_pred, masks)
            loss = loss / CFG.n_accumulate

        scaler.scale(loss).backward()

        if (step + 1) % CFG.n_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                         lr=f'{current_lr:0.5f}',
                         gpu_mem=f'{mem:0.2f} GB')
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss


# ref: https://www.kaggle.com/code/awsaf49/uwmgi-unet-train-pytorch/#%F0%9F%9A%85-Training
@torch.no_grad()
def valid_one_epoch(model, optimizer, dataloader, device, epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    val_scores = []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')
    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        batch_size = images.size(0)

        y_pred = model(images)
        loss = criterion(y_pred, masks)

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        y_pred = nn.Sigmoid()(y_pred)
        val_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
        val_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()
        val_scores.append([val_dice, val_jaccard])

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                         lr=f'{current_lr:0.5f}',
                         gpu_memory=f'{mem:0.2f} GB')
    val_scores = np.mean(val_scores, axis=0)
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss, val_scores


# ref: https://www.kaggle.com/code/awsaf49/uwmgi-unet-train-pytorch/#%F0%9F%9A%85-Training
def run_training(model, train_loader, valid_loader, optimizer, scheduler, device, num_epochs, fold, run):
    # To automatically log gradients
    wandb.watch(model, log_freq=100)

    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_dice = -np.inf
    best_epoch = -1
    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        print(f'Epoch {epoch}/{num_epochs}', end='')
        train_loss = train_one_epoch(model, optimizer, scheduler,
                                     dataloader=train_loader,
                                     device=CFG.device, epoch=epoch)

        val_loss, val_scores = valid_one_epoch(model, optimizer, valid_loader,
                                               device=CFG.device,
                                               epoch=epoch)
        val_dice, val_jaccard = val_scores

        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(val_loss)
        history['Valid Dice'].append(val_dice)
        history['Valid Jaccard'].append(val_jaccard)

        # Log the metrics
        wandb.log({"Train Loss": train_loss,
                   "Valid Loss": val_loss,
                   "Valid Dice": val_dice,
                   "Valid Jaccard": val_jaccard,
                   "LR": scheduler.get_last_lr()[0]})

        print(f'Valid Dice: {val_dice:0.4f} | Valid Jaccard: {val_jaccard:0.4f}')

        # deep copy the model
        if val_dice >= best_dice:
            print(f"{c_}Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
            best_dice = val_dice
            best_jaccard = val_jaccard
            best_epoch = epoch
            run.summary["Best Dice"] = best_dice
            run.summary["Best Jaccard"] = best_jaccard
            run.summary["Best Epoch"] = best_epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"best_epoch-{fold:02d}.bin"
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            wandb.save(PATH)
            print(f"Model Saved{sr_}")

        last_model_wts = copy.deepcopy(model.state_dict())
        PATH = f"last_epoch-{fold:02d}.bin"
        torch.save(model.state_dict(), PATH)

        print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Score: {:.4f}".format(best_jaccard))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history


def fetch_scheduler(optimizer):
    if CFG.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.T_max,
                                                   eta_min=CFG.min_lr)
    elif CFG.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0,
                                                             eta_min=CFG.min_lr)
    elif CFG.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.1,
                                                   patience=7,
                                                   threshold=0.0001,
                                                   min_lr=CFG.min_lr,
                                                   )
    elif CFG.scheduler == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    elif CFG.scheduler is None:
        return None

    return scheduler


data_transforms = {
    "train": A.Compose([
        A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
            # A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
        ], p=0.25),
        A.CoarseDropout(max_holes=8, max_height=CFG.img_size[0] // 20, max_width=CFG.img_size[1] // 20,
                        min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
    ], p=1.0),

    "valid": A.Compose([
        A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
    ], p=1.0)
}


def main():
    utils.set_seed(CFG.seed)

    df = pd.read_csv('./train.csv')
    if CFG.n_sample is not None:
        df = df.head(CFG.n_sample)

    # df['image_path'] = df.id.map()
    df['segmentation'] = df.segmentation.fillna('')
    df['rle_len'] = df.segmentation.map(len)  # length of each rle mask

    df2 = df.groupby(['id'])['segmentation'].agg(list).to_frame().reset_index()  # rle list of each id
    df2 = df2.merge(
        df.groupby(['id'])['rle_len'].agg(sum).to_frame().reset_index())  # total length of all rles of each id

    df = df.drop(columns=['segmentation', 'class', 'rle_len'])
    df = df.groupby(['id']).head(1).reset_index(drop=True)
    df = df.merge(df2, on=['id'])
    df = utils.get_image_path("./train", df)
    df['empty'] = (df.rle_len == 0)  # empty masks

    def mask_path(ID):
        return f"./train_masks/{ID}.png"

    df['mask_path'] = df.id.map(mask_path)

    skf = StratifiedGroupKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['empty'], groups=df["case"])):
        df.loc[val_idx, 'fold'] = fold

    for fold in range(1):
        print(f'#' * 15)
        print(f'### Fold: {fold}')
        print(f'#' * 15)
        run = wandb.init(project='uw-maddison-gi-tract',
                         config={k: v for k, v in dict(vars(CFG)).items() if '__' not in k},
                         name=f"fold-{fold}|dim-{CFG.img_size[0]}x{CFG.img_size[1]}|model-{CFG.model_name}",
                         group=CFG.comment,
                         )
        train_loader, valid_loader = prepare_loaders(df, fold=fold, debug=CFG.debug)
        model = build_model()
        optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
        scheduler = fetch_scheduler(optimizer)
        model, history = run_training(model,
                                      train_loader,
                                      valid_loader,
                                      optimizer,
                                      scheduler,
                                      device=CFG.device,
                                      num_epochs=CFG.epochs,
                                      fold=fold,
                                      run=run,
                                      )
        run.finish()


if __name__ == '__main__':
    main()
