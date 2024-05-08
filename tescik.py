from better_resnet.calc_2_wasserstein_dist import calculate_2_wasserstein_dist

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm


array = torch.tensor([ [0, -1, 1],[0, 0, 3], [7, 8, 9], [10, 11, 12]])
array2 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print(calculate_2_wasserstein_dist(array, array2))
