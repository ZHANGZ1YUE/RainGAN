import numpy as np
import random

from Model import Generator, Discriminator
from dataset import raindata  

from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
import torch.nn as nn
import torch

import matplotlib.pyplot as plt



#######################################   Setting   #######################################
def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)

seed_everything(20220901)
cuda_device = 6
epochs = 500
lr_G = 0.0002
lr_D = 0.001
batch_size = 16
input_channel = 4
output_channel = 1
_lambda = 20       #scalar for the pixel-wise loss term
g_path = "generator.pth"
d_path = "discriminator.pth"

torch.cuda.set_device(cuda_device)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



#####################################   Dataset Loading   ###########################################

in_dir = 'paddedinput_30mins_4ch.npy'
out_dir = 'paddedoutput_30mins_4ch.npy'
val_percent: float = 0.2 #Use 20% as validation dataset



dataset = raindata(in_dir, out_dir)     #创造一个dataset，用我上面自定义的dataset class

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





########################################   Start Training   #################################################


generator = Generator()
discriminator = Discriminator(input_channel, output_channel)

generator.to(device)
discriminator.to(device)

loss_L1 = nn.L1Loss().cuda()
loss_binaryCrossEntropy = nn.BCELoss().cuda()

optimizer_G= torch.optim.Adam(generator.parameters(), lr= lr_G, betas=(0.5, 0.999), weight_decay = 0.00001)
optimizer_D= torch.optim.Adam(discriminator.parameters(), lr= lr_D, betas=(0.5, 0.999), weight_decay = 0.00001)

def GAN_Loss(input, target, criterion):
    if target == True:
        tmp_tensor = torch.FloatTensor(input.size()).fill_(1.0)
        labels = Variable(tmp_tensor, requires_grad=False)
    else:
        tmp_tensor = torch.FloatTensor(input.size()).fill_(0.0)
        labels = Variable(tmp_tensor, requires_grad=False)

    if torch.cuda.is_available():
        labels = labels.cuda()

    return criterion(input, labels)

def to_variable(x):
    if torch.cuda.is_available():
        x = x.to(device, dtype=torch.float32)
    return Variable(x)


total_step = len(train_loader) # For Print Log
for epoch in range(epochs):
    for i, batch in enumerate(train_loader):

        # x is the input from the dataset batch, y is the output
        x, y = batch

        # =================================== Train Discriminator ====================================#
        discriminator.zero_grad()

        real_x = to_variable(x)
        fake_y = generator(real_x)    #Use generator to generate an initial prediction, denote as fake_y
        real_y = to_variable(y)


        pred_fake = discriminator(real_x, fake_y)
        loss_D_fake = GAN_Loss(pred_fake, False, loss_binaryCrossEntropy)

        pred_real = discriminator(real_x, real_y)
        loss_D_real = GAN_Loss(pred_real, True, loss_binaryCrossEntropy)

        # Combined loss
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_D.backward(retain_graph=True)
        optimizer_D.step()

        # =================================== Train Generator =====================================#
        generator.zero_grad()

        pred_fake = discriminator(real_x, fake_y)
        loss_G_GAN = GAN_Loss(pred_fake, True, loss_binaryCrossEntropy)

        loss_G_L1 = loss_L1(fake_y, real_y)

        loss_G = loss_G_GAN + loss_G_L1 * _lambda
        loss_G.backward()
        optimizer_G.step()

        # print the log info
        if (i + 1) % 10 == 0:
            print('Epoch [%d/%d], BatchStep[%d/%d], D_Real_loss: %.4f, D_Fake_loss: %.4f, G_loss: %.4f, G_L1_loss: %.4f'
                    % (epoch + 1, epochs, i + 1, total_step, loss_D_real.item(), loss_D_fake.data.item(), loss_G_GAN.data.item(), loss_G_L1.data.item()))

  
torch.save(generator.state_dict(), g_path)
torch.save(discriminator.state_dict(), d_path)

