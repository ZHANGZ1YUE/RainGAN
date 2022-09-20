import os
import numpy as np
import math
import time
import random
from collections import defaultdict
from pathlib import Path
from Model import Generator

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt


def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)


##########################   Training Setting   ##############################
seed_everything(20220901)
save_path = "generator.pth"
cuda_device = 7
epoch = 50
learning_rate = 0.001


#################################   Dataset Preparation & Preprocessing   ###################################
transform = transforms.Compose([
  transforms.ToTensor()
])

class rain(Dataset):

    def __init__(self, input_dir, output_dir, transform=transform):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.input = np.load(self.input_dir)
        self.output = output = np.load(self.output_dir)
        self.transform = transform

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):   #idx就是来选取第几个的那个index
        input = self.input[idx]
        output = self.output[idx] 

        if self.transform:
            input = self.transform(input)  #torch的tensor和np实际上是反过来的，我们需要改动一下
            output = self.transform(output)

        return input, output



#####################################   Dataset Loading   ###########################################

in_dir = 'paddedinput_30mins_4ch.npy'
out_dir = 'paddedoutput_30mins_4ch.npy'
val_percent: float = 0.2 #Use 20% as validation dataset
batch_size = 16

dataset = rain(in_dir, out_dir, transform)     #创造一个dataset，用我上面自定义的dataset class

n_val = int(len(dataset) * val_percent)   #这几行就是定义多少个training 多少个validation
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

#验证一下dataset是不是我想的那样
'''
input, output = dataset[0]  #打印第0个element of dataset
print(type(input))   # --> <class 'torch.Tensor'>
print(input.size())   # --> torch.Size([4, 192, 128])
print(output.size())   # --> torch.Size([1, 192, 128])
plt.imshow(input[:,:,0]) # --> 应该打出来不对的 因为现在是channel开头了 不是channel最后了 transform的作用
'''

#Dataloader了该 Dataloader的作用就是打包Batch
loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True) #为下面的两行做准备
train_loader = DataLoader(train_set, shuffle=True, **loader_args)   #dataloader来load这个dataset，分为train和validation
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

'''
print(len(train_loader)) # -->29 因为29*16 = 464 大概就是整个training set,这个代表一个epoch走29个iteration,每一个iteration有16个batch
onebatch = next(dataloader_iter)
print(onebatch[0].size())
'''

dataloaders = {                   ########################删除
  'train': train_loader,
  'val': val_loader
}





#######################################   Start Training   #################################################

#Training function
criterion = nn.MSELoss()
torch.cuda.set_device(cuda_device)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = Generator().to(device)

def calc_loss(pred, target, metrics):
    loss = criterion(pred, target)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

def train_model(model, optimizer, num_epochs):
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']: 
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for input, output in dataloaders[phase]:
                input = input.to(device, dtype=torch.float32)  
                output = output.to(device, dtype=torch.float32)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    prediction = model(input)
                    loss = calc_loss(prediction, output, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += input.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            #if phase == 'train':
            #  scheduler.step()
            #  for param_group in optimizer.param_groups:
            #      print("LR", param_group['lr'])

            # save the model weights
            if phase == 'val' and epoch_loss < best_loss:
                print(f"saving best model to {save_path}")
                best_loss = epoch_loss
                torch.save(model.state_dict(), save_path)

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    #model.load_state_dict(torch.load(save_path))
    #return model

optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
#scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
train_model(model, optimizer, epoch)