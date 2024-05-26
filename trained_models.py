import os
import torch
from config import ArgObj

my_args = ArgObj()

def find_final(seed, folder="./models"):
    original_model = None
    modified_model = None
    
    for the_file in os.listdir(folder):
        
        if f"seed_{seed}" in the_file and "FINAL" in the_file:
            file_path = os.path.join(folder, the_file)
            if "orig" in the_file:
                original_model = torch.load(file_path, map_location=torch.device('cpu'))
            elif "tanh" in the_file:
                modified_model = torch.load(file_path, map_location=torch.device('cpu'))
                
    if original_model == None or modified_model == None:
        return 0
    else:
        return original_model, modified_model
    
trained_models = {
    "original":[],
    "modified":[],
    "seeds":[]
}

seedlist = my_args.seedlist

for i in range(5):
    seed=seedlist[i]
    loaded_models = find_final(seed=seed)
    
    if loaded_models != 0:
        orig, mod = loaded_models
        trained_models["original"].append(orig)
        trained_models["modified"].append(mod)
        trained_models["seeds"].append(seed)

def class_max(input):
    return (torch.max(torch.exp(input), 1)[1]).data.cpu().numpy()