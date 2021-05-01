import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from copy import deepcopy

import logging
log = logging.getLogger(__name__)

def gradient_interference(model, prev_grads, prev_nonzero_indices):
    grads, nonzero_indices = [], []
    for _, p in model.named_parameters():
        if p.grad is not None:
            # TODO: clone does not work for some reason
            grad = deepcopy(p.grad.detach().flatten())
            grads.append(grad)
            mask = (grad!=0.0).to(p.device)
            nonzero_indices.append(mask)
        # case for heads of a shared base network
        # where grad will be None
        else:
            shape = p.flatten().shape
            grads.append(torch.zeros(shape).to(p.device))
            nonzero_indices.append(torch.zeros(shape).bool().to(p.device))
    grads = torch.cat(grads)
    nonzero_indices = torch.cat(nonzero_indices)
    if prev_grads is None or prev_nonzero_indices is None:
        prev_grads = torch.zeros_like(grads).to(grads.device)
        prev_nonzero_indices = torch.zeros_like(nonzero_indices).to(grads.device)
    interference = 1 - F.cosine_similarity(grads, prev_grads, dim=0) 
    overlap = torch.sum(nonzero_indices * prev_nonzero_indices)
    return grads, nonzero_indices, interference, overlap

def gradient_similarity(model, names, dataloaders):
    grads, nonzero_mask = {}, {}
    for name, dataloader in zip(names, dataloaders):
        grads[name], nonzero_mask[name] = get_gradients(model, dataloader)
    grad_sim, grad_shared = {}, {}
    cos_sim = nn.CosineSimilarity(dim=0)
    for task_i, grad_i in grads.items():
        for task_j, grad_j in grads.items():
            shared_mask = nonzero_mask[task_i] * nonzero_mask[task_j]
            grad_sim[f"grad_sim/{task_i}/{task_j}"] = cos_sim(
                grad_i[shared_mask], 
                grad_j[shared_mask]
            ).detach().cpu().numpy().item()
            grad_shared[f"grad_shared/{task_i}/{task_j}"] = shared_mask.sum().detach().cpu().numpy().item()
    return grad_sim, grad_shared

def get_gradients(model, dataloader):
    # accumulate gradients
    for i, batch in enumerate(dataloader):
        loss, _ = model.step(batch)
        # scale loss to accumulate the average of gradients
        loss = loss/len(dataloader) 
        loss.backward()
    # extract gradients and indexes of nonzero gradients
    grads, nonzero_mask = [], []
    for _, p in model.named_parameters():
        if p.grad is not None:
            # TODO: clone does not work for some reason
            grad = deepcopy(p.grad.detach().flatten())
            grads.append(grad)
            mask = (grad!=0.0).to(p.device)
            nonzero_mask.append(mask)
        # case for heads of a shared base network
        # where grad will be None
        else:
            shape = p.flatten().shape
            grads.append(torch.zeros(shape).to(p.device))
            nonzero_mask.append(torch.zeros(shape).bool().to(p.device))
    model.zero_grad()
    return torch.cat(grads), torch.cat(nonzero_mask)