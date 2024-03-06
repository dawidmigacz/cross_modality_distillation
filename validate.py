from pytorch_resnet import resnet18
from pytorch_utils import get_datasets, val_epoch
import tqdm
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageFile
import wandb
ImageFile.LOAD_TRUNCATED_IMAGES = True

def f(*args, **kwargs):
    return torch.tensor([0])

train_loader, test_loader = get_datasets()
model = resnet18(pretrained=True)
val_epoch(test_loader, model, f)








