import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CosineSimilarity


class ContrastiveLoss(nn.Module):
    def __init__(self, tao):
        super(ContrastiveLoss, self).__init__()
        self.sim = CosineSimilarity(dim=-1)
        self.tao = tao

    def forward(self, features, debias_features, bias_features):
        features = F.normalize(features, dim=1)
        debias_features = F.normalize(debias_features, dim=1)
        bias_features = F.normalize(bias_features, dim=1)

        pos_sim = self.sim(features, debias_features)
        neg_sim = self.sim(features, bias_features)

        logits = torch.exp(pos_sim / self.tao) / (torch.exp(pos_sim / self.tao) + torch.exp(neg_sim / self.tao))
        loss = (-1.0 * torch.log(logits))

        return loss.mean()

