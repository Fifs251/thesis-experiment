import torch
import torch.nn as nn
from torch.optim import SGD
from config import ArgObj
import numpy as np
import matplotlib.pyplot as plt

my_args = ArgObj()

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 2, bias=False)
        self.fc2 = nn.Linear(2, 2, bias=False)
        self.fc3 = nn.Linear(2, 2, bias=False)
        self.acti = nn.ReLU()
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.fc1(x)
        x = self.acti(x)
        x = self.fc2(x)
        x = self.acti(x)
        logits = self.fc3(x)
        return logits
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0, std=1)
            if module.bias is not None:
                module.bias.data.zero_()

my_args.lr = 0.1
model = SimpleNet()
optimizer = SGD(model.parameters(), lr=my_args.lr)

criterion_mean = nn.CrossEntropyLoss(reduction='mean')

nr_inputs = 5
n = int(np.floor(np.random.rand()*nr_inputs))
m = int(nr_inputs - n)

print(f"Input: {input}")


layer = model.fc2
weight_movement = np.empty([nr_inputs, 2])
max_x = 100
print_tf = True

def sameSign(x, y):
    if (x*y > 0):
        return True
    else:
        return False

max_se_nw = 0

def generate_input():
    target = np.array([1] * n + [0] * m)
    target = torch.tensor(target)
    input = torch.rand(nr_inputs, 5)
    return input, target
#input, target = generate_input()

for x in range(max_x):
    print(f"x: {x}")
    #model = SimpleNet()
    #layer = model.fc2
    input, target = generate_input()
    for i in range(nr_inputs):
        optimizer.zero_grad()
        output = model(input[i])
        loss = criterion_mean(output, target[i])
        loss.backward()
        if print_tf:
            print(f"Grads after backward in data scan {i+1}:")
            print(layer.weight.grad)
            print(layer.weight)
        w1 = layer.weight[0][0]*1
        w2 = layer.weight[0][1]*1
        optimizer.step()
        if print_tf:
            print(layer.weight)
            print(f"w1 step: {(layer.weight[0][0] - w1)}")
            print(f"w2 step: {(layer.weight[0][1] - w2)}")
        weight_movement[i][0] = layer.weight[0][0]
        weight_movement[i][1] = layer.weight[0][1]
        if print_tf:
            print('\n\n')

    se_nw = 0
    n_s_w_e = 0
    n_s = 0

    for i in range(1, nr_inputs):
        w1_up = weight_movement[i,0]-weight_movement[i-1,0]
        w2_up = weight_movement[i,1]-weight_movement[i-1,1]
        if not sameSign(w1_up, w2_up) and not (w1_up == 0 or w2_up == 0):
            se_nw+=1
        if (w1_up == 0 and w2_up != 0) or (w2_up == 0 and w1_up != 0):
            n_s_w_e += 1
        if (w1_up == 0 and w2_up != 0):
            n_s += 1

    if print_tf:
        print(f"Number of north-west or south-east movements: {se_nw}")
        print(f"Number of north, south, west or east movements: {n_s_w_e}")

    if se_nw > max_se_nw:
        max_se_nw = se_nw
    
    if n_s > 0:
        break

print('\n\n')
print(f"MAX se_nw: {max_se_nw}")

offset_x = 0
offset_y = 5

fig = plt.figure(figsize=[8,8])
plt.plot(weight_movement[:,0], weight_movement[:,1])
for i in range(nr_inputs):
    plt.annotate(i+1,
            xy=(weight_movement[i,0], weight_movement[i,1]), xycoords='data',
            xytext=(offset_x, offset_y), textcoords='offset points')
plt.xlabel("w1")
plt.ylabel("w2")
plt.savefig('figs/weights_zig_zag.png', dpi=500)