import mymodels
import torch
import math

"""
Prints mean and std of the same layer in 2 models
"""

wi_y = mymodels.AlexNet()
wi_n = mymodels.AlexNet_no_weight_init()

wi_y_l=wi_y.features[0]
wi_n_l=wi_n.features[0]

wi_y_w = wi_y_l.weight
wi_n_w = wi_n_l.weight

wi_y_b = wi_y_l.bias
wi_n_b = wi_n_l.bias

print(f"With init:\n\tWeights:\n\t\tStd:{torch.std(wi_y_w)}\n\t\tMean:{torch.mean(wi_y_w)}\n\tBias:\n\t\tStd:{torch.std(wi_y_b)}\n\t\tMean:{torch.mean(wi_y_b)}\n")
print(f"Without init:\n\tWeights:\n\t\tStd:{torch.std(wi_n_w)}\n\t\tMean:{torch.mean(wi_n_w)}\n\tBias:\n\t\tStd:{torch.std(wi_n_w)}\n\t\tMean:{torch.mean(wi_n_w)}")
print(wi_n_w.size(1))

print(1. / math.sqrt(wi_n_w.size(1)))