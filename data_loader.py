import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class Unbalanced_CIFAR10():
    def __init__(self, n_data = 5000, classes=['dog', 'cat'], 
                 proportion=0.9, n_val=50, type='train'):
        
        if type == 'train':
            self.cifar = datasets.CIFAR10('data', train=True, download=True)
        else:
            self.cifar = datasets.CIFAR10('data', train=False, download=True)
            proportion = 0.5
            n_data = 0
        
        self.transform=transforms.Compose([
            # transforms.Resize([3,32,32]),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            # transforms.Grayscale()
            # transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomVerticalFlip(0.2),
            # transforms.RandomRotation(30),
            # transforms.RandomAdjustSharpness(0.4) #,
            # transforms.ToTensor(),            
        ])
        
        n_c1 = int(np.floor(n_data * proportion))
        n_c2 = n_data - n_c1
        n_class = [n_c1-n_val, n_c2-n_val]

        self.data       = []
        self.val_data   = []
        self.labels     = []
        self.val_labels = []
        
        classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
             'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
        
        ## Debug
        # labels = 'airplane automobile bird cat deer dog frog horse ship truck'.split()
        # print(self.cifar.data)
        # print(self.cifar.class_to_idx)
        
        # first_50_samples = sorted([self.cifar[i] for i in range(50)], key=lambda x:x[1])

        # figure = plt.figure(figsize=(15,10))
        # for i in range(1,51):
        #     img = first_50_samples[i-1][0].permute(1,2,0)
        #     label = labels[first_50_samples[i-1][1]]
        #     figure.add_subplot(5,10,i)
        #     plt.title(label)
        #     plt.axis('off')
        #     plt.imshow(img)
        ## End debug
        dim1 = self.cifar.data.shape[0]
        
        if type == 'train':
            raw_data = torch.from_numpy(self.cifar.data).float()/255
            raw_labels = self.cifar.targets
        else:
            raw_data = torch.from_numpy(self.cifar.data).float()/255
            raw_labels = self.cifar.targets
        
        # print(raw_data[:5])
        # print(raw_labels[:5])
        
        for i, c in enumerate(classes):
            labels_array = np.array(raw_labels)
            locs = np.where(labels_array == classDict[c])[0]
            np.random.shuffle(locs)
            # locs = torch.from_numpy(locs)
            
            self.data.append(raw_data[locs[0:n_class[i]]])
            label_idx = labels_array[locs[0:n_class[i]]] == classDict[classes[0]]
            label_idx = torch.from_numpy(label_idx)
            self.labels.append(label_idx.float())
            
            # Make fill out validation data
            if type == 'train':
                image_val = raw_data[locs[n_class[i]:n_class[i] + n_val]]
                # Try just adding it like this
                self.val_data.append(image_val)
                
                label_val = labels_array[locs[n_class[i]:n_class[i] + n_val]] == classDict[classes[0]]
                label_val = torch.from_numpy(label_val)
                self.val_labels.append(label_val.float())
                
        self.data   = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)
        
        if type == 'train':
            self.val_data   = torch.cat(self.val_data, dim=0)
            self.val_labels = torch.cat(self.val_labels, dim=0)
            
    # Other required functions for custom dataset in DataLoader
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        x = self.transform(self.data[index].permute(2,0,1))
        y = self.labels[index]
        
        return x, y
            
def get_loader(batch_size, n_data=5000, classes=['dog', 'cat'], proportion=0.95, n_val=10, 
           type='train'):
    
    dataset = Unbalanced_CIFAR10(n_data=n_data, classes=classes, proportion=proportion, n_val=n_val)
    
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=True)
    return loader