import numpy as np
from sklearn.metrics import confusion_matrix
from dataset_module import *
from hierarchy import sorted_labels
from trained_models import trained_models, class_max
import time
import datetime
import pandas as pd

def make_CM(model):
    
    y_pred = np.array([])
    y_true = np.array([])
    
    for data, target in test_loader:
            model = model.to("cpu")
            
            target = target.numpy()
            
            output = model(data)
            output = class_max(output)
            
            y_pred = np.append(y_pred, output)
            y_true = np.append(y_true, target)
            
    cm = confusion_matrix(y_true, y_pred, labels = sorted_labels)
    
    return cm

def all_CMs():
    cms={"orig":[],
                "mod":[],
                "seeds":[]}
        
    small_start_t=time.time()
    print("Making CMs...")
    for i in range(len(trained_models["seeds"])):
        curseed=trained_models["seeds"][i]
        print(f"Seed: {curseed}")
        print("orig")
        cms["orig"].append(make_CM(trained_models["original"][i]))
        print("mod")
        cms["mod"].append(make_CM(trained_models["modified"][i]))
        cms["seeds"].append(curseed)
    small_duration=time.time()-small_start_t
    print(f"Done. CM gen duration:{datetime.timedelta(seconds=small_duration)}")
    print("\n\n")

    return cms

cms = all_CMs()

def delete_diagonal():
    for modeltype in [cms["orig"], cms["mod"]]:
        for cm in modeltype:
            for i in range(len(cm)):
                cm[i,i]=0

delete_diagonal()

#normalize, transform into dataframe and turn into pickle Rick (hilarious)

def normalize_in_df():
    for modeltype in ["orig", "mod"]:
        dict_item=cms[modeltype]
        counter=0
        for cm in dict_item:
            cm = pd.DataFrame(cm / np.sum(cm, axis=1)[:, None])
            cm.to_pickle(f"pickle/CM_{modeltype}_#{cms['seeds'][counter]}.pkl")
            counter+=1

normalize_in_df()