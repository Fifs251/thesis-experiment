import torch
import os

from evaluation import evaluate_iteration
from training import test_cnn
from dataset_module import test_loader
import matplotlib.pyplot as plt
import numpy as np
from config import ArgObj
import pickle
from matplotlib.ticker import FormatStrFormatter

my_args = ArgObj()

def find_model(seed, type, epoch, folder="./models"):

    for the_file in os.listdir(folder):

        if f"seed_{seed}" in the_file and type in the_file and (the_file.endswith(f"epoch-{epoch}") or the_file.endswith(f"epoch-{epoch}_FINAL")):
            file_path = os.path.join(folder, the_file)
            loaded_model = torch.load(file_path, map_location=torch.device('cpu'))

    return loaded_model

def find_max_epoch(seed, type, folder="./models"):
  for the_file in os.listdir(folder):
        if f"seed_{seed}" in the_file and type in the_file and the_file.endswith(f"_FINAL"):
          return int(the_file[-8:-6])

metrics = ["precision_macro",
            "recall_macro",
            "sub_acc",
            "super_acc",
            "ultra_acc",
            "loss"]

seedlist = my_args.seedlist

def generate_plots():
  eval_dict = {
    "orig":[],
    "tanh":[],
    "seeds":seedlist
  }

  for s in seedlist:
    for t in ['orig', 'tanh']:
      num_epochs=find_max_epoch(s, t)
      result_list=[]
      for i in range(1,num_epochs+1):
          model=find_model(s,t,i)
          
          result_list.append(evaluate_iteration(model))
          result_list[i-1]["loss"]=test_cnn(model, test_loader, my_args, print_on=False)
          print(f"seed {s} {t} epoch {i} done")
      eval_dict[t].append(result_list)
  return eval_dict

#eval_dict=generate_plots()
#print(eval_dict)

#with open('pickle/all_epoch_eval_dict.pkl', 'wb') as f:
#      pickle.dump(eval_dict, f)

with open('pickle/all_epoch_eval_dict.pkl', 'rb') as f:
        eval_dict = pickle.load(f)

def create_y_array(input, metric):
  output=[]
  for i in input:
    output.append(i[metric])
  return np.array(output)

"""
result_list = np.array([{'a':1,
                         'b':2,
                         'c':3,
                         'd':4,
                         'e':5},
                          {'a':1.1,
                           'b':2.2,
                           'c':3.3,
                           'd':4.4,
                           'e':5.5}])
num_epochs = 2

losses=[4,3]
"""

def prepare_plot(modeltype, metric):
    overall_max_epoch=0

    for seed in seedlist:
      max_epoch_one_model=find_max_epoch(seed, modeltype)
      if max_epoch_one_model>overall_max_epoch:
        overall_max_epoch=max_epoch_one_model
    
    x=range(overall_max_epoch)
    a=[]
    s=[]

    for i in x:
      metrics_on_epoch=[]

      for m in eval_dict[modeltype]:
        try:
          metrics_on_epoch.append(m[i][metric])
        except:
          pass
      
      a.append(np.average(metrics_on_epoch))
      s.append(np.std(metrics_on_epoch))

    a=np.array(a)
    s=np.array(s)
    
    ubound=a+s
    lbound=a-s
    
    return x, a, ubound, lbound

def plot_average(metric, filename='test'):
  fig, axs = plt.subplots(1, 2, sharey=True)

  x_orig, a_orig, ubound_orig, lbound_orig = prepare_plot('orig', metric)

  axs[0].plot(x_orig, a_orig, label=metric)
  axs[0].fill_between(x_orig, ubound_orig, lbound_orig, alpha=0.2)

  x_tanh, a_tanh, ubound_tanh, lbound_tanh = prepare_plot('tanh', metric)

  axs[1].plot(x_tanh, a_tanh, label=metric)
  axs[1].fill_between(x_tanh, ubound_tanh, lbound_tanh, alpha=0.2)

  axs[0].set_title('Original')
  axs[1].set_title('Tanh')

  axs[0].set_xticks(np.arange(5, max(x_orig)+1, 5))
  axs[1].set_xticks(np.arange(5, max(x_tanh)+1, 5))

  for ax in axs:
    ax.set_xlabel("Epoch")

  plt.savefig(f"figs/{filename}.png", dpi=300)

  return 0

"""
plot_average('precision_macro', 'avg_metrics-prec_mac')

plot_average('recall_macro', 'avg_metrics-rec_mac')

plot_average('sub_acc', 'avg_metrics-sub_acc')

plot_average('super_acc', 'avg_metrics-super_acc')

plot_average('ultra_acc', 'avg_metrics-ultra_acc')

plot_average('loss', 'avg_metrics-loss')
"""

def long_name(metric):
  name_dict={
    'precision_macro': 'Precision',
    'recall_macro': 'Recall',
    'sub_acc': 'Sub-class Accuracy',
    'super_acc': 'Super-class Accuracy',
    'ultra_acc': 'Ultra-class Accuracy',
    'loss': 'Loss'
  }
  return name_dict[metric]

def plot_average_all(filename='test1'):
  fig, axs = plt.subplots(1, 2, sharey=True)

  for i in range(6):

    x_orig, a_orig, ubound_orig, lbound_orig = prepare_plot('orig', metrics[i])

    axs[0].plot(x_orig, a_orig, label=long_name(metrics[i]))
    axs[0].fill_between(x_orig, ubound_orig, lbound_orig, alpha=0.2)

    x_tanh, a_tanh, ubound_tanh, lbound_tanh = prepare_plot('tanh', metrics[i])

    axs[1].plot(x_tanh, a_tanh, label=long_name(metrics[i]))
    axs[1].fill_between(x_tanh, ubound_tanh, lbound_tanh, alpha=0.2)

    axs[0].set_title('Original')
    axs[1].set_title('Tanh')

    axs[0].set_xticks(np.arange(5, max(x_orig)+1, 5))
    axs[1].set_xticks(np.arange(5, max(x_tanh)+1, 5))

  for ax in axs:
    ax.set_xlabel("Epoch")

  plt.legend(loc='upper right', bbox_to_anchor=(1, 0.4), prop={'size': 8})
  plt.savefig(f"figs/{filename}.png", dpi=500)

  return 0

plot_average_all(filename="metrics_over_epochs_ALL")

#linestyles = ['solid', 'dotted', 'dashed', 'solid', 'dashdot', 'dotted']

def plot_per_seed(metric, seed, filename='test'):
  fig, axs = plt.subplots(1, 2, sharey=True)

  x_orig=range(find_max_epoch(seed, 'orig'))
  x_tanh=range(find_max_epoch(seed, 'tanh'))

  y_orig=create_y_array(eval_dict['orig'][seedlist.index(seed)], metric)
  y_tanh=create_y_array(eval_dict['tanh'][seedlist.index(seed)], metric)

  axs[0].plot(x_orig, y_orig, label=metric)
  axs[1].plot(x_tanh, y_tanh, label=metric)

  axs[0].set_title('Original')
  axs[1].set_title('Tanh')

  axs[0].set_xticks(np.arange(5, max(x_orig)+1, 5))
  axs[1].set_xticks(np.arange(5, max(x_tanh)+1, 5))

  for ax in axs:
    ax.set_xlabel("Epoch")

  fig.suptitle(f"Seed: {seed}")
  plt.savefig(f"figs/metrics_over_epochs/{filename}.png", dpi=300)

"""
for seed in seedlist:
  plot_per_seed('precision_macro', seed, f'prec_mac-over-epochs-seed_{seed}')
  plot_per_seed('recall_macro', seed, f'rec_mac-over-epochs-seed_{seed}')
  plot_per_seed('sub_acc', seed, f'sub_acc-over-epochs-seed_{seed}')
  plot_per_seed('super_acc', seed, f'super_acc-over-epochs-seed_{seed}')
  plot_per_seed('ultra_acc', seed, f'ultra_acc-over-epochs-seed_{seed}')
  plot_per_seed('loss', seed, f'loss-over-epochs-seed_{seed}')
"""