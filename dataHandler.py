import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset

#Read JSON
DATA = [] #planes 1; no planes 0
print("Reading JSON...")
with open("planes.json") as f:
    json_data = json.load(f)['data']
    planes = json_data[:8000]
    no_planes = json_data[8000:]
    DATA.append(no_planes)
    DATA.append(planes)
print("JSON Read!")

#Functions
def listToTensor(data):
    channels = list()
    for c in range(3):
        channels.append([])
        for x in range(20): 
            channels[c].append([])
            for y in range(20):
                value = data[c*(20**2) + x*20 + y]/255
                channels[c][x].append(value)
    return torch.Tensor(channels)

def plotTensor(tensor):
    plt.imshow(tensor.permute(1, 2, 0).numpy())
    plt.show()

def tripleAugment(tensor):
    v_tensor = torch.flip(tensor,[1])
    h_tensor = torch.flip(tensor,[2])
    return [tensor,v_tensor,h_tensor]

class PlaneDataset(Dataset):
    def __init__(self,count):
        self.X_data = []
        self.Y_data = []
        for _ in range(count*3):
            self.Y_data.append(0)
            self.X_data.append(listToTensor(DATA[0].pop()))
        for _ in range(count):
            self.Y_data.extend([1]*3)
            self.X_data.extend(tripleAugment(listToTensor(DATA[1].pop())))
        self.length = len(self.Y_data)

    def __len__(self):
       return self.length

    def __getitem__(self, idx):
       return self.X_data[idx],self.Y_data[idx]

breakdown = [7500,400,100]
if sum(breakdown) != 8000:
    print("INVALID BREAKDOWN!")

trainset = PlaneDataset(breakdown[0])
testset = PlaneDataset(breakdown[1])
devest = PlaneDataset(breakdown[2])

print(len(trainset))
print(len(testset))
print(len(devest))