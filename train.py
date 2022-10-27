#%%
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import torchvision
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset
from torchvision.datasets import ImageFolder
from torchsummary import summary

# Tool
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import wandb

# Utils
from train_model import Net
from train_utils import Get_Device, Set_Seed, Load_Dataset

#%%
# train_path = "dataset/train"
# test_path = "dataset/test"

# Hyperparameter
LR = 1e-5
EPOCH = 50
IMAGE_SIZE = 48
BATCH_SIZE = 64

train_set, valid_set, train_loader, valid_loader = Load_Dataset(IMAGE_SIZE, BATCH_SIZE)

#%%
if __name__ == "__main__":
    Set_Seed()
    device = Get_Device()
    
    # Time Recording
    total_time = 0

    # Load model
    model = Net()
    model.to(device)
    # summary(model, (3, IMAGE_SIZE, IMAGE_SIZE))
    summary(model, (1, IMAGE_SIZE, IMAGE_SIZE))

    # Initial Wandb
    wandb.init(project="Face-Emotion-Classifier", name=f"lr{LR}_b{BATCH_SIZE}")
    wandb.config = config={
                           "Dataset" : "FER2013",
                           "Learning_Rate" : LR,
                           "Batch Size" : BATCH_SIZE,
                           "Epoch" : EPOCH,
                           "Augmentation" : "RandomRotation((-40, 40))",
                           "Some Case" : "Using Gray to Train"
                          }
    wandb.watch(model, log_freq=100)

    # Loss & Optimizer function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Initial Some Value
    min_loss = 1000      # 用於判定是否該存 model

    # Start Training
    for epoch in range(EPOCH):

        # Start Recording Time
        time_start = time.time()

        # Initial Loss & Accuracy
        train_loss = 0.0
        valid_loss = 0.0
        train_acc  = 0.0
        valid_acc  = 0.0

        # Train :
        model.train()

        for (data, label) in tqdm(train_loader):
            data, label = data.to(device), label.to(device)

            pred = model(data)
            loss = loss_fn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10) # 梯度剪裁
            optimizer.step()

            # Compute the accuracy for current batch.
            acc = (pred.argmax(dim=-1) == label.to(device)).float().mean()

            # Record the Loss & Accuracy
            _, pred_index = torch.max(pred, 1) # get the index of the class with the highest probability

            train_acc  += (pred_index.cpu() == label.cpu()).sum().item() # [1, 0, 1] => mean = (1 + 0 + 1)/3 = 0.66
            train_loss += loss.item()

        # Valid :
        model.eval()

        with torch.no_grad():
            for (data, label) in tqdm(valid_loader):
                data, label = data.to(device), label.to(device)

                pred = model(data)
                loss = loss_fn(pred, label)

                # Record the Loss & Accuracy
                _, pred_index = torch.max(pred, 1)

                valid_acc  += (pred_index.cpu() == label.cpu()).sum().item()
                valid_loss += loss.item()

            # Caculate the Average Accuracy & Loss
            train_avg_acc  = (train_acc/len(train_set))*100
            valid_avg_acc  = (valid_acc/len(valid_set))*100
            
            train_avg_loss = train_loss/len(train_loader)
            valid_avg_loss = valid_loss/len(valid_loader)

            # Show the Accuracy & Loss each epoch
            print('[{:03d}/{:03d}] Train Acc: {:3.2f} Loss: {:3.6f} | Val Acc: {:3.2f} loss: {:3.6f}'.format(epoch + 1, EPOCH, train_avg_acc, train_avg_loss, valid_avg_acc, valid_avg_loss))
            wandb.log({"Train Accuracy": train_avg_acc, "Train Loss" : train_avg_loss, "Valid Accuracy" : valid_avg_acc, "Valid Loss" : valid_avg_loss})

            # if the model improves, save a checkpoint at this epoch
            if(valid_avg_loss < min_loss):
                min_loss = (valid_loss/len(valid_loader))
                torch.save(model.state_dict(), 'model.pth')
                print('Saving model with loss {:.3f}'.format(min_loss))

        time_end = time.time()              # Finish Recording Time
        time_cost = time_end - time_start   # Time Spent
        total_time = total_time + time_cost # Total Time

        print("Each Epoch Cost : {:3.3f} s\n".format(time_cost))

    print("Total Cost Time : {:3.3f} s".format(total_time))

#%%