import torch.nn as nn
from torch import max, unsqueeze
import matplotlib.pyplot as plt
import numpy as np

from dataset_module import train_loader, test_loader, val_loader
from trained_models import trained_models
from hierarchy import get_labels, sorted_labels
from config import ArgObj
from metrics_over_epochs import find_max_epoch, find_model

my_args = ArgObj()

def find_image(image_class="caterpillar", dataloader=test_loader):
    flag=False
    for data, target in dataloader:
        for i in range(len(target)):
            if get_labels(target[i]) == image_class:
                lbl = target[i].cpu()
                image = data[i]
                flag = True
                break
        if flag:
            break

    return image, lbl

def output_prob(modeltype, seed, image_class="caterpillar", dataloader=test_loader, args=my_args):
    seed_idx = args.seedlist.index(seed)
    model = trained_models[modeltype][seed_idx]

    model.eval()

    m = nn.Softmax(dim=1)

    img, lbl = find_image(image_class, dataloader)

    img = unsqueeze(img, 0)

    output = model(img)
    soft_output = m(output.data.cpu())[0]
    #predicted = max(soft_output, 0)[1]
    soft_output = [soft_output[i] for i in sorted_labels]
    text_labels = [get_labels(i) for i in sorted_labels]

    bar_colors = ["tab:red"] * 100
    bar_colors[sorted_labels.index(lbl)] = "tab:green"

    fig=plt.figure()
    plt.bar(text_labels, soft_output, color=bar_colors)
    plt.xticks(rotation=90, fontsize=3.5)
    plt.title(f"{modeltype.capitalize()}, seed: {seed}")
    plt.savefig(f"figs/output_probabilities_one_class/out_prob-{modeltype[:4]}_{seed}.png", dpi=500)

def plot_image(image_class="caterpillar", dataloader=test_loader):
    img, _ = find_image(image_class, dataloader)
    img = img / 2 + 0.5
    npimg = img.numpy()
    fig=plt.figure()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.savefig(f"figs/output_probabilities_one_class/example_image.png", dpi=500)

image_class="bear"

for i in my_args.seedlist:
    for j in ["original", "modified"]: 
        None
        #output_prob(j, i, image_class)

#plot_image(image_class)



# shape: (dataloader, modeltype, seed, num_epochs, num_batches, batch_size, [output*num_classes, target])

def collect_probs():
    m = nn.Softmax(dim=1)
    for dataloader, data_name in zip([#train_loader, test_loader, 
        val_loader], [#"train", "test", 
        "val"]):
        for modeltype in ["orig", "mod"]:
            for seed in my_args.seedlist:
                max_epochs=2#find_max_epoch(seed, modeltype)
                soft_output_save = []
                for epoch in range(1, max_epochs+1):
                    soft_output_epoch = []
                    print(f"{data_name} data, type {modeltype}, seed {seed}, epoch {epoch}")
                    model=find_model(seed, modeltype, epoch)
                    for batch_idx, (data, target) in enumerate(dataloader):
                        print(f"batch {batch_idx}")
                        output=model(data)
                        soft_output = m(output.detach().cpu()).numpy()
                        soft_output_epoch.append([soft_output, target])
                    soft_output_save.append(soft_output_epoch)
                soft_output_save = np.array(soft_output_save)
                np.save(f"output_probs/{data_name}/{modeltype}-{seed}.pkl", soft_output_save, allow_pickle=True)

collect_probs()