import torch.nn.functional as F

def contrastive_loss(out1, out2, label, margin=1.0):
    dist = F.pairwise_distance(out1, out2)
    loss = label * dist.pow(2) + (1 - label) * F.relu(margin - dist).pow(2)
    return loss.mean()