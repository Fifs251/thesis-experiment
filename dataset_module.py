import torch
import torchvision
import torchvision.transforms as transforms
from config import ArgObj
from torch.utils.data import random_split

my_args = ArgObj()

myTransform = transforms.Compose([
    transforms.Resize((227,227)),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = torchvision.datasets.CIFAR100(root='./root', train=True,
                                              transform=myTransform, download=False)
test_val_dataset = torchvision.datasets.CIFAR100(root='./root', train=False,
                                              transform=myTransform, download=False) 

gnr = torch.Generator().manual_seed(my_args.dataset_seed)

test_dataset, val_dataset = random_split(test_val_dataset, [0.5, 0.5], generator=gnr)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=my_args.batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=my_args.batch_size, shuffle=False)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=my_args.batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = next(examples)