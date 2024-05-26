import pandas as pd
from config import ArgObj

my_args = ArgObj()

def load_CMs():
    cms={"orig":[],
            "mod":[],
            "seeds":my_args.seedlist}
    for i in range(len(cms["seeds"])):
        curseed=cms["seeds"][i]
        cms["orig"].append(pd.read_pickle(filepath_or_buffer=f"pickle/CM_orig_#{curseed}.pkl"))
        cms["mod"].append(pd.read_pickle(filepath_or_buffer=f"pickle/CM_mod_#{curseed}.pkl"))
    return cms

cms=load_CMs()