import torch
import argparse
import yaml
from model import MG
from torch_geometric import seed_everything
import numpy as np
import warnings
from eval import label_classification
from dataset import process_dataset

warnings.filterwarnings('ignore')


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "w" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = torch.optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = torch.optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = torch.optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = torch.optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return torch.optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
    return optimizer


def train(args):
    model = MG(feat.size(1), args.dim, args.p1, args.p2, args.beta, args.beta1,
               args.rate, args.rate1, args.alpha).to(device)
    optimizer = create_optimizer("adam", model, args.lr, args.w)

    for epoch in range(1, args.epoch + 1):
        model.train()
        loss = model(graph, diff_graph, feat, edge_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        z1, z2 = model.get_embed(graph, diff_graph, feat, edge_weight)
        acc = label_classification(z1 + z2, train_mask, val_mask, test_mask,
                                   label, args.label_type)['Acc']['mean']
        print(f"Epoch: {epoch}, Loss: {loss.item()}, Acc: {acc}")


parser = argparse.ArgumentParser(description="GraphMVM")
parser.add_argument("--dataset", type=str, default="Cora")
parser.add_argument("--cuda", type=int, default=0)
args = parser.parse_args()
args = load_best_configs(args, "config.yaml")
graph, diff_graph, feat, label, train_mask, val_mask, test_mask, \
edge_weight = process_dataset(args.dataset)
n_feat = feat.shape[1]
n_classes = np.unique(label).shape[0]
seed_everything(35536)
device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

graph = graph.to(device)
label = label.to(device)
diff_graph = diff_graph.to(device)
feat = feat.to(device)
edge_weight = torch.tensor(edge_weight).float().to(device)
# n_node = graph.number_of_nodes()

train(args)
