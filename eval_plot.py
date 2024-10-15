from hierarchy import *
import numpy as np
from dataset_module import *
from trained_models import trained_models
import matplotlib.pyplot as plt 
import pickle 

with open('pickle/eval_dict.pkl', 'rb') as f:
        my_dict = pickle.load(f)

def print_evaluation():

    cnt=len(trained_models["seeds"])

    for i in range(cnt):
        print(f"Seed: {my_dict['seeds'][i]}")
        print("\t\t\tOriginal\tModified\tDiff")
        print(f"Precision (macro): \t{my_dict['orig'][i]['precision_macro']:.4f}\t\t{my_dict['mod'][i]['precision_macro']:.4f}", end="")
        print(f"\t\t{my_dict['mod'][i]['precision_macro']-my_dict['orig'][i]['precision_macro']:.4f}")
        
        print(f"Recall (macro):    \t{my_dict['orig'][i]['recall_macro']:.4f}\t\t{my_dict['mod'][i]['recall_macro']:.4f}", end="")
        print(f"\t\t{my_dict['mod'][i]['recall_macro']-my_dict['orig'][i]['recall_macro']:.4f}")
        
        print(f"Sub-class Accuracy:\t{my_dict['orig'][i]['sub_acc']:.4f}\t\t{my_dict['mod'][i]['sub_acc']:.4f}", end="")
        print(f"\t\t{my_dict['mod'][i]['sub_acc']-my_dict['orig'][i]['sub_acc']:.4f}")
        
        print(f"Super-class Accuracy:   {my_dict['orig'][i]['super_acc']:.4f}\t\t{my_dict['mod'][i]['super_acc']:.4f}", end="")
        print(f"\t\t{my_dict['mod'][i]['super_acc']-my_dict['orig'][i]['super_acc']:.4f}")
        
        print(f"Ultra-class Accuracy:   {my_dict['orig'][i]['ultra_acc']:.4f}\t\t{my_dict['mod'][i]['ultra_acc']:.4f}", end="")
        print(f"\t\t{my_dict['mod'][i]['ultra_acc']-my_dict['orig'][i]['ultra_acc']:.4f}")
        print("\n")
    
    print("Original avg diffs")
    print(f"Precision: \t{np.average([my_dict['orig'][i]['precision_macro'] for i in range(cnt)]):.4f}")

#print_evaluation()

def eval_plot_all():
    plt.clf()

    x = [1,2,3] 

    fig = plt.figure(figsize=(15, 15))
    fig.tight_layout()

    cnt=0
    rows=2
    columns=3

    for seed in my_dict['seeds']:
        #for modelid, modeltext in zip(['orig', 'mod'], ['ReLU', 'Tanh']):
        y1=[]
        y2=[]

        for i in ['sub_acc', 'super_acc', 'ultra_acc']:
            y1.append(my_dict['orig'][cnt][i])
            y2.append(my_dict['mod'][cnt][i])


        fig.add_subplot(rows, columns, cnt+1)
        plt.plot(x, y1, label = "ReLU" if cnt == 0 else "", linestyle='dashed', marker="x")
        plt.plot(x, y2, label = "Tanh" if cnt == 0 else "", linestyle='dashed', marker="x")
        plt.xticks(np.arange(1,4), ["sub", "super", "ultra"])
        plt.title(f'Seed: {seed}', fontsize=20)

        cnt+=1
        
    fig.legend(loc='lower right', bbox_to_anchor=(0.79, 0.39), fontsize=20, markerscale=1.5) 
    fig.savefig('figs/eval_all.png', dpi=500)

eval_plot_all()

def eval_plot_one():

    plt.clf()
    x = [1,2,3] 
    plt.figure()
    y1=[]
    y2=[]

    for i in ['sub_acc', 'super_acc', 'ultra_acc']:
        y1.append(my_dict['orig'][0][i])
        y2.append(my_dict['mod'][0][i])

    plt.xticks(np.arange(1,4), ["sub-class accuracy", "super-class accuracy", "ultra-class accuracy"])
    plt.plot(x, y1, label = "ReLU", linestyle='dashed', marker="x")
    plt.plot(x, y2, label = "Tanh", linestyle='dashed', marker="x")

    plt.legend() 
    plt.savefig('figs/eval_one.png', dpi=500)

eval_plot_one()