import numpy as np
import datetime
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from load_cms import cms

def evaluate_CMs():
    SSEs = np.empty((10,10))
    SSEs[:] = np.nan
    
    cnt=len(cms["seeds"])

    for i in range(cnt):
        for j in range(cnt):
            if i!=j:
                SSEs[i,j]=np.sum((cms["orig"][i]-cms["orig"][j]).to_numpy()**2)
                SSEs[i+5,j+5]=np.sum((cms["mod"][i]-cms["mod"][j]).to_numpy()**2)
            if i!=j+5:
                SSEs[i,j+5]=np.sum((cms["orig"][i]-cms["mod"][j]).to_numpy()**2)
            if i+5!=j:
                SSEs[i+5,j]=np.sum((cms["mod"][i]-cms["orig"][j]).to_numpy()**2)

    return SSEs

SSEs=evaluate_CMs()

#log_dir = "tb_logs"
#writer = SummaryWriter(log_dir)

def SSE_plot():
    ticklabs=["R1","R2","R3","R4","R5","T1","T2","T3","T4","T5"]

    for i in range(1):
        df_cm = cms["orig"][i] / np.sum(cms["orig"][i], axis=1)
        cm_fig = sn.heatmap(df_cm, annot=False).get_figure()
        cm_fig.savefig(f"figs/CM_orig_#{cms['seeds'][i]}.png")

        df_cm = pd.DataFrame(cms["mod"][i])
        cm_fig = sn.heatmap(df_cm, annot=False).get_figure()
        cm_fig.savefig(f"figs/CM_mod_#{cms['seeds'][i]}.png", dpi=500)

    fig, ax = plt.subplots(1,1)

    img = ax.imshow(SSEs)

    ax.set_xticks(range(10))
    ax.set_xticklabels(ticklabs)
    #ticklabs.reverse()
    ax.set_yticks(range(10))
    ax.set_yticklabels(ticklabs)
    plt.colorbar(img)

    fig.savefig(f"figs/SSE_plot.png", dpi=500)
    
    #writer.add_figure("SSEs", fig.colorbar(img))


SSE_plot()
print(f"Upper left quadrant mean: {np.nanmean(SSEs[:5,:5])}")
print(f"Upper right quadrant: {np.mean(SSEs[:5,5:])}")
print(f"Lower left quadrant mean: {np.mean(SSEs[5:,:5])}")
print(f"Lower right quadrant mean: {np.nanmean(SSEs[5:,5:])}")