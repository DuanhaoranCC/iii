import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from torch_geometric.nn import BatchNorm


def sce_loss(x, y, alpha=3):
    """Code adapted from GraphMAE, Scaled Cosine Error"""
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    return loss.mean()


def mask(g, x, mask_rate=0.5, noise=0.05):
    """Random Mask Node and Noise Node from GraphMAE"""
    num_nodes = g.num_nodes()
    perm = torch.randperm(num_nodes, device=x.device)
    num_mask_nodes = int(mask_rate * num_nodes)

    mask_nodes = perm[: num_mask_nodes]

    num_noise_nodes = int(noise * num_mask_nodes)

    perm_mask = torch.randperm(num_mask_nodes, device=x.device)

    token_nodes = mask_nodes[perm_mask[: int((1 - noise) * num_mask_nodes)]]

    noise_nodes = mask_nodes[perm_mask[-int(noise * num_mask_nodes):]]

    noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

    return token_nodes, noise_nodes, noise_to_be_chosen, mask_nodes


class Encoder1(nn.Module):
    """Local Graph View Encoder, Only One layer GCN"""
    def __init__(self, in_hidden, out_hidden, p1):
        super(Encoder1, self).__init__()
        self.conv1 = GraphConv(in_hidden, out_hidden, norm='both',
                               bias=True, activation=nn.PReLU())
        self.dp1 = nn.Dropout(p1)
        self.act = nn.PReLU()
        self.bn = BatchNorm(out_hidden)

    def forward(self, graph, diff_graph, feat, edge_weight):
        x = self.dp1(feat)
        h = self.conv1(graph, x)
        h = self.bn(h)
        h = self.act(h)

        return h


class Encoder2(nn.Module):
    """Global Graph View Encoder, Only One layer GCN"""
    def __init__(self, in_hidden, out_hidden, p1):
        super(Encoder2, self).__init__()
        self.conv1 = GraphConv(in_hidden, out_hidden, norm='none',
                               bias=True, activation=nn.PReLU())
        self.dp1 = nn.Dropout(p1)
        self.act = nn.PReLU()
        self.bn = BatchNorm(out_hidden)

    def forward(self, graph, diff_graph, feat, edge_weight):
        x = self.dp1(feat)
        h = self.conv1(diff_graph, x, edge_weight=edge_weight)
        h = self.bn(h)
        h = self.act(h)

        return h


class MG(nn.Module):
    """Multi-View Mask Self-Supervised Representation Learning GraphMVM Main Model"""
    def __init__(self, in_hidden, out_hidden, p1, p2, beta, beta1, rate,
                 rate1, alpha):
        super(MG, self).__init__()
        self.enc = Encoder1(in_hidden, out_hidden, p1)
        self.dec = Encoder2(in_hidden, out_hidden, p2)
        self.rate = rate
        self.rate1 = rate1
        self.alpha = alpha
        self.noise = 0.05
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_hidden))
        self.enc_mask_token1 = nn.Parameter(torch.zeros(1, in_hidden))
        self.criterion = self.setup_loss_fn("sce", beta)
        self.criterion1 = self.setup_loss_fn("sce", beta1)

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        return criterion

    def forward(self, graph, diff_graph, feat, edge_weight):

        token_nodes, noise_nodes, \
        noise_to_be_chosen, mask_nodes = mask(graph, feat, mask_rate=self.rate,
                                              noise=self.noise)
        x = feat.clone()
        x[token_nodes] = 0.0
        if self.noise > 0:
            x[noise_nodes] = feat[noise_to_be_chosen]
        else:
            x[noise_nodes] = 0.0
        x[token_nodes] += self.enc_mask_token

        h1 = self.enc(graph, diff_graph, x, edge_weight)
        h2 = self.dec(graph, diff_graph, feat, edge_weight)
        loss1 = self.criterion(h1[mask_nodes], h2[mask_nodes])

        token_nodes1, noise_nodes1, \
        noise_to_be_chosen1, mask_nodes1 = mask(graph, feat, mask_rate=self.rate1,
                                                noise=self.noise)
        x = feat.clone()
        x[token_nodes1] = 0.0
        if self.noise > 0:
            x[noise_nodes1] = feat[noise_to_be_chosen1]
        else:
            x[noise_nodes1] = 0.0
        x[token_nodes1] += self.enc_mask_token1

        h1 = self.enc(graph, diff_graph, feat, edge_weight)
        h2 = self.dec(graph, diff_graph, x, edge_weight)
        loss2 = self.criterion1(h1[mask_nodes1], h2[mask_nodes1])

        return self.alpha * loss1 + loss2 * (1 - self.alpha)

    def get_embed(self, graph, diff_graph, feat, edge_weight):
        h1 = self.enc(graph, diff_graph, feat, edge_weight)
        h2 = self.dec(graph, diff_graph, feat, edge_weight)

        return h1, h2
