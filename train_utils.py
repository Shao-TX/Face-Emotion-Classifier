#%%
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

#%%
def Get_Device():
    if(torch.cuda.is_available()):
        print("Device : GPU")
        device = torch.device("cuda:0")
    else:
        print("Device : CPU")
        device = torch.device("cpu")

    return device

#%%
# Set Random Seed
def Set_Seed(myseed = 1520):
    np.random.seed(myseed)

    torch.manual_seed(myseed)
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#%%
def Load_Dataset(IMAGE_SIZE, BATCH_SIZE):
    train_transform = transforms.Compose([
                                        #   transforms.ColorJitter(brightness=(0.5, 1.2)),                          # 隨機亮度調整
                                          transforms.RandomHorizontalFlip(p=0.5),                                 # 隨機水平翻轉
                                          transforms.RandomRotation((-40, 40)),                                   # 隨機旋轉
                                        #   transforms.RandomResizedCrop(size = IMAGE_SIZE, scale = (0.5, 1.5)),    # 隨機縮放
                                        
                                        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                        transforms.Grayscale(num_output_channels=1),
                                        transforms.ToTensor(),
                                        # transforms.Normalize((0.5, 0.5 ,0.5), (0.5, 0.5 ,0.5)) => Use RGB to train
                                        transforms.Normalize((0.5, ), (0.5, )) # => Use Gray to train
                                        ])

    valid_transform = transforms.Compose([
                                        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                        transforms.Grayscale(num_output_channels=1),
                                        transforms.ToTensor(),
                                        # transforms.Normalize((0.5, 0.5 ,0.5), (0.5, 0.5 ,0.5)) => Use RGB to train
                                        transforms.Normalize((0.5, ), (0.5, )) # => Use Gray to train
                                        ])

    train_set = ImageFolder(root = r"dataset/train", transform = train_transform)
    valid_set = ImageFolder(root = r"dataset/test", transform = valid_transform)

    train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size = BATCH_SIZE, shuffle=False, pin_memory=True)

    print(train_set.class_to_idx)

    return train_set, valid_set, train_loader, valid_loader