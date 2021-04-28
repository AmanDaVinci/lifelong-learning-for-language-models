import logging
import torch
import torch.nn as nn
from torch import optim
from copy import deepcopy

import logging
log = logging.getLogger(__name__)


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
    opt = optim.AdamW(model.parameters())
    # accumulate gradients
    for i, batch in enumerate(dataloader):
        # log.debug(f"batch idx: {i} batch shape: {batch['input_ids'].shape}")
        loss, _ = model.step(batch)
        # scale loss to accumulate the average of gradients
        loss = loss/len(dataloader) 
        loss.backward()
    # extract gradients and indexes of nonzero gradients
    grads, nonzero_mask = [], []
    # TODO: could be done without using optim
    for param_group in opt.param_groups:
        for p in param_group['params']:
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