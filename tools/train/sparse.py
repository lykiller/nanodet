import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def global_prune(abs_weights, sparsity, sparsity_max):
    all_weights = torch.cat([w for w in abs_weights.values()])
    total = all_weights.numel()
    zero_counts = int(total * sparsity)

    thresh = all_weights.sort()[0][zero_counts]
    del all_weights

    weight_threshs = {}
    not_saturate_weights = {}
    for k, v in abs_weights.items():
        if (v < thresh).sum().item() / float(v.numel()) >= sparsity_max:
            zero_count = int(v.numel() * sparsity_max)
            weight_threshs[k] = v.sort()[0][zero_count]
            zero_counts -= zero_count
        else:
            not_saturate_weights[k] = v

    # print(weight_threshs)
    if len(not_saturate_weights) == len(abs_weights):
        for k in not_saturate_weights.keys():
            weight_threshs[k] = thresh
    elif len(not_saturate_weights) == 1:
        weight_threshs[list(not_saturate_weights.keys())[0]] = thresh
    elif len(not_saturate_weights) > 1:
        sparsity = zero_counts / sum([w.numel() for w in not_saturate_weights.values()])
        weight_threshs.update(global_prune(not_saturate_weights, sparsity, sparsity_max))
    return weight_threshs


def global_prune_init(weights, sparsity, sparsity_max):
    masks = {}
    abs_weights = {}
    for k, v in weights.items():
        if v.dim() != 4:    ###no bn
            continue
        if ((v.size(0) == 80) or (v.size(0) == 4) or (v.size(0) == 1)):    ### 2022/5/19  just for coco...
            continue
        
        abs_weights[k] = v.data.abs().view(-1)

    weight_threshs = global_prune(abs_weights, sparsity, sparsity_max)
    for k, v in abs_weights.items():
        masks[k] = (v <= weight_threshs[k]).view(weights[k].shape)        

    return masks


def sparsify(weights, masks):
    for k, mask in masks.items():
        weights[k].data[mask] = 0