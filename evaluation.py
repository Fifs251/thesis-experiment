import torch
from sklearn.metrics import precision_score, recall_score
from hierarchy import *
import numpy as np
from dataset_module import *
from trained_models import trained_models
import pickle

def class_max(input):
    return (torch.max(torch.exp(input), 1)[1]).data.cpu().numpy()

def evaluate_iteration(model):
    
    results={"precision_macro":0,
            "recall_macro":0,
            "sub_acc":0,
            "super_acc":0,
            "ultra_acc":0}
    
    y_pred = np.array([])
    y_true = np.array([])
    
    for data, target in test_loader:
            model = model.to("cpu")
            
            target = target.numpy()
            
            output = model(data)
            output = class_max(output)
            
            y_pred = np.append(y_pred, output)
            y_true = np.append(y_true, target)
            
            #total += len(target)
            #correct += (predicted == target).sum().item()

    results["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    results["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    
    results["sub_acc"] = sum(y_true==y_pred)/len(y_true)
    results["super_acc"] = sum(v_create_superclass(y_true)==v_create_superclass(y_pred))/len(y_true)
    results["ultra_acc"] = sum(v_create_ultraclass(v_create_superclass(y_true))==v_create_ultraclass(v_create_superclass(y_pred)))/len(y_true)
    
    return results

#evaluate_iteration(trained_models["original"][0])

def save_evaluation():
    my_dict={"orig":[],
            "mod":[],
            "seeds":[]}
    
    cnt=len(trained_models["seeds"])

    for i in range(cnt):
        my_dict["orig"].append(evaluate_iteration(trained_models["original"][i]))
        my_dict["mod"].append(evaluate_iteration(trained_models["modified"][i]))
        my_dict["seeds"].append(trained_models["seeds"][i])
        
    with open('pickle/eval_dict.pkl', 'wb') as f:
        pickle.dump(my_dict, f)

    return my_dict

#save_evaluation()