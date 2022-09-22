import numpy as np
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import Dataset

#################################   Dataset Preparation & Preprocessing   ###################################
transform = transforms.Compose([
  transforms.ToTensor()
])

class raindata(Dataset):      #Torch dataset class. __init__, __len__, __getitem__ are necessary

    def __init__(self, input_dir, output_dir, transform=transform):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.input = np.load(self.input_dir)
        self.output  = np.load(self.output_dir)
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








