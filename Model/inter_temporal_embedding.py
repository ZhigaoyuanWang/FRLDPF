import argparse
import os
import torch
import random
import numpy as np
import json

def interpolation(temporal_embedding,device):
### interpolation temporal embeddings
    length = [53,52,52,52,52,53,52,12]
    tau = [0,5,4,3,2,0,6,5]
    l_index = []
    r_index = []
    tau_list = []
    for i in length:
        l_index+=list(range(i))
        if i == 53:
            r_index+=list(range(i))
        else:
            r_index+=list(range(1,i+1))
    for i in range(8):
        tau_list+=[tau[i] for j in range(length[i])]
    l_index = torch.tensor(l_index).to(device)
    r_index = torch.tensor(r_index).to(device)
    tau_list = torch.tensor(tau_list).to(device)
    inter_embeddings = ((temporal_embedding(l_index).T * (7-tau_list) + temporal_embedding(r_index).T * tau_list)/7).T
    return inter_embeddings