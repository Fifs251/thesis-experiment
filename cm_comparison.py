from load_cms import cms
import matplotlib.pyplot as plt
from hierarchy import *

#generator for section borders 

def section_border_gen(get_level): #squares begin in the upper left corner 
    border_list=[-0.5]

    #get_level - get superclass or ultraclass from a subclass index
    state=get_level(sorted_labels[0])
    for i in range(len(sorted_labels)):
        current=get_level(sorted_labels[i])
        #if change in super- or ultra-class at the given level - make a border
        if current!=state:
            border_list.append(i-0.5)
        state=get_level(sorted_labels[i])

#smaller than 99.5 so that the line doesn't stand out from the graph
    border_list.append(99.5)

    return border_list

#level = the borders (at a given level - super/ultra) #create 4 points to make a square #draw lines between

def add_borders(fig, axs):
    section_borders = [section_border_gen(get_super), section_border_gen(get_ultra)] 
    border_colors = ["red", "green"] 
    line_widths = [0.7, 1.5]

    for ax in axs: 
        for level, color, line_width in zip(section_borders, border_colors, line_widths): 
            for i, j in zip(level, level[1:]): 
                points = [[i,i], [i,j], [j,j], [j,i]]
                for k, l in zip(points, points[1:]):
                    ax.plot(k, l, color=color, linewidth=line_width)
                    ax.plot(points[3], points[0], color=color, linewidth=line_width)

def plot_two_CMs(types = ["orig", "mod"], idxs = [0, 0], titles = ["ReLU", "Tanh"]):

    cm1 = cms[types[0]][idxs[0]]
    cm2 = cms[types[1]][idxs[1]]

    fig, axs = plt.subplots(2, 1, figsize = (15,20)) 
    axs[0].imshow(cm1, interpolation='none', aspect='auto') #store for colorbar 
    im=axs[1].imshow(cm2, interpolation='none', aspect='auto')

    for i in range(len(axs)):
        axs[i].set_title(f"{titles[i]}", fontdict={"size": 32})

    add_borders(fig, axs)
    
    for ax in axs: 
        ax.axis('off') 
        ax.grid(True)

    fig.colorbar(im, ax=axs.ravel().tolist())

    plt.savefig('figs/CM_comparison_two.png')

#plot_two_CMs()

def plot_all_CMs():

    fig, axs = plt.subplots(5, 2, figsize = (25,50))
    fig.tight_layout()
    
    types = ["orig", "mod"]
    titles = ["ReLU", "Tanh"]

    for modeltype, title, axcol in zip(types, titles, range(2)):
        for cm, axrow in zip(cms[modeltype], range(5)):
            im=axs[axrow, axcol].imshow(cm, interpolation='none', aspect='auto')
            #if axrow==0:
                #axs[axrow, axcol].set_title(f"{title}", fontdict={"size": 40})

    add_borders(fig, axs.flat)

    for ax in axs.flat: 
        ax.axis('off') 
        ax.grid(True)

    #fig.colorbar(im, ax=axs.ravel().tolist())

    plt.savefig('figs/CM_comparison_all.png')

plot_all_CMs()