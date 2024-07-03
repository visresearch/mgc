
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.loss import SoftTargetCrossEntropy
import copy
import numpy as np
import torch.distributed as dist

def contrastive_loss_patch(q, k, T, labels, offset, distributed, ious, batch_size, max_size):
    q = nn.functional.normalize(q, dim=1)
    k = nn.functional.normalize(k, dim=1)
        
    if distributed:
        k = concat_all_gather(k) # distribute
    logits = torch.einsum('nc,mc->nm', [q, k]).cuda()
    ious = ious.reshape(batch_size * max_size)
    loss = F.cross_entropy(logits / T, labels)*ious 
    similarity_pos = torch.mean(torch.diag(logits, diagonal=offset))
        
    return loss.sum()/ious.sum(), similarity_pos

@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output
