import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import datetime
from datetime import datetime as dtm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math
from torch.utils.data import random_split
import time
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
from IPython.display import FileLink
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import seaborn as sn

import mymodels
from config import ArgObj

my_args = ArgObj()

myTransform = transforms.Compose([
    transforms.Resize((227,227)),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

### Dataset, Loaders, Split
train_dataset = torchvision.datasets.CIFAR100(root='./root', train=True,
                                              transform=myTransform, download=False)
test_val_dataset = torchvision.datasets.CIFAR100(root='./root', train=False,
                                              transform=myTransform, download=False) 

gnr = torch.Generator().manual_seed(my_args.seed)

test_dataset, val_dataset = random_split(test_val_dataset, [0.5, 0.5], generator=gnr)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=my_args.batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=my_args.batch_size, shuffle=False)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=my_args.batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = next(examples)

criterion_mean = nn.CrossEntropyLoss(reduction='mean')

def train_epoch_cnn(model, train_loader, optimizer, args, print_reduce=False):

    model.train()  # set model to training mode (activate dropout layers for example)
    t = time.time() # we measure the needed time
    for batch_idx, (data, target) in enumerate(train_loader):  # iterate over training data
        data, target = data.to(args.device), target.to(args.device)  # move data to device (GPU) if necessary
        optimizer.zero_grad()  # reset optimizer
        output = model(data)   # forward pass: calculate output of network for input
        loss = criterion_mean(output, target)  # calculate loss
        loss.backward()  # backward pass: calculate gradients using automatic diff. and backprop.
        optimizer.step()  # udpate parameters of network using our optimizer
        cur_time = time.time()
        # print some outputs if we reached our logging intervall
        if cur_time - t > args.log_interval or batch_idx == len(train_loader)-1:
            if not print_reduce:
                print(f"[{batch_idx * len(data):.0f}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tloss: {loss.item():.6f}, took {cur_time - t:.2f}s")
            t = cur_time

def test_cnn(model, test_loader, args):

    model.eval()  # set model to inference mode (deactivate dropout layers for example)
    test_loss = 0  # init overall loss
    correct = 0
    total = 0
    
    with torch.no_grad():  # do not calculate gradients since we do not want to do updates
        
        for data, target in test_loader:  # iterate over test data
            data, target = data.to(args.device), target.to(args.device)  # move data to device 
            output = model(data) # forward pass
            # claculate loss and add it to our cumulative loss
            test_loss += criterion_mean(output, target).item()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    test_loss /= len(test_loader.dataset)  # calc mean loss
    test_acc = (100 * correct / total)
    
    print('Average eval loss: {:.4f}'.format(test_loss, len(test_loader.dataset)))
    print('Eval accuracy: {:.4f}%\n'.format(test_acc))
    return test_loss

def inference_cnn(model, data, args):
    model.eval()   # set model to inference mode
    in_shape = data.shape
    side = int((args.context - 1) / 2)
    outlen = in_shape[0] - 2 * side
    output = np.zeros((in_shape[0], args.out_num))
    data = torch.from_numpy(data[None, :, :]) 
    data = data.to(args.device) # move input to device
    with torch.no_grad(): # do not calculate gradients
        for idx in range(outlen): # iterate over input data
            # calculate output for input data (and move back from device)
            output[idx+side, :] = model(data[:, idx:(idx + args.context), :])[0, :].cpu()
    return output

def get_activation(model_name, layer_name, epoch, seed):
    def hook(inst, inp, out):
        flattened = out.flatten()
        writer.add_histogram((model_name.capitalize()+" seed#"+str(seed)+" "+layer_name), out[0], epoch)
    return hook

log_dir = "tb_logs"
writer = SummaryWriter(log_dir)

examples = iter(train_loader)
samples, labels = next(examples)

def hook_write_to_TB(model, train_loader, epoch, model_name, seed):
    handle4=model.classifier[1].register_forward_hook(get_activation(model_name, "FC1 WO", epoch, seed))
    handle1=model.classifier[2].register_forward_hook(get_activation(model_name, "FC1", epoch, seed))
    handle5=model.classifier[4].register_forward_hook(get_activation(model_name, "FC2 WO", epoch, seed))
    handle2=model.classifier[5].register_forward_hook(get_activation(model_name, "FC2", epoch, seed))
    handle3=model.classifier[6].register_forward_hook(get_activation(model_name, "FC3", epoch, seed))
    
    global samples
    global my_args

    y=model(samples.to(my_args.device))

    handle1.remove()
    handle2.remove()
    handle3.remove()
    handle4.remove()
    handle5.remove()
    
## Training
def train_cnn(smoke_test=False, args=my_args, model_arg=mymodels.AlexNet(), name="undefined", datekey="000101", save=True, save_folder="./models", print_reduce=False):
    
    if smoke_test:
        max_epochs = 2
    else:
        max_epochs = args.max_epochs
    
    torch.manual_seed(args.seed)
    device = args.device
    
    model = model_arg.to(device)
    
    from torch.optim import SGD
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_valid_loss = 9999.
    cur_patience = args.patience

    print(f"Training {name}#seed_{args.seed}")
    start_t = time.time()
    
    for epoch in range(1,max_epochs):
        print(f"Model: {name.capitalize()}")
        print(f"Epoch: {epoch}")
        print(f"Best validation loss: {best_valid_loss:.4f}")
        print(f"Patience: {cur_patience}")

        train_epoch_cnn(model, train_loader, optimizer, args, print_reduce=print_reduce)
        
        cur_valid_loss = test_cnn(model, val_loader, args)
        
        hook_write_to_TB(model, train_loader, epoch, name, args.seed)
        writer.add_scalar(("Loss/train "+name), cur_valid_loss, epoch)

        if cur_valid_loss<best_valid_loss:
            best_valid_loss=cur_valid_loss
            cur_patience=args.patience
        else:
            cur_patience-=1
            
        if cur_patience<=0:
            print(f"Best validation loss: {best_valid_loss:.4f}")
            if save:
                torch.save(model, f"{save_folder}/{datekey}_{name}#seed_{args.seed}-epoch-{str(epoch)}_FINAL")
            break
        
        if save:
            torch.save(model, f"{save_folder}/{datekey}_{name}#seed_{args.seed}-epoch-{str(epoch)}")
    
    print('Trainig took: {:.2f}s for {} epochs'.format(time.time()-start_t, epoch))
    print('Testing...')
    test_loss = test_cnn(model, test_loader, args)
    
    writer.flush()

    return model

seedlist = [751,456,894,564,483]

def experiment(rounds=5, modified_only=False):
    experiment_start = time.time()
    
    datekey = dtm.now().strftime("%y-%m-%d_%H-%M")
    
    for i in range(rounds):
        round_start = time.time()
        
        my_args = ArgObj(seed=seedlist[(i+1)])
        
        if not modified_only:
            trained_model1 = train_cnn(args=my_args, model_arg=mymodels.AlexNet(), name = "inittest", datekey=datekey, save=False)
            #trained_model2 = train_cnn(args=my_args, model_arg=mymodels.AlexNet_Tanh(), name= "modified_tanh", datekey=datekey)

        round_end = time.time()
        round_duration = round_end-round_start
        
        print(f"Round {i+1} duration: {datetime.timedelta(seconds=round_duration)}")
    
    experiment_end = time.time()
    experiment_duration = experiment_end-experiment_start
    
    print(f"Full experiment duration: {datetime.timedelta(seconds=experiment_duration)}")
    
    #cm = evaluate(trained_model1)

experiment()