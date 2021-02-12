""" explainer_main.py

     Main user interface for the explainer module.
"""
import argparse
import os
import numpy as np

import sklearn.metrics as metrics

from tensorboardX import SummaryWriter

import pickle
import shutil
import torch

import configs_explainer

import src.GNNtrainer.models as models
import src.utils.io_utils as io_utils
from explainer import Explainer, train_explainer


def main():
    # Load a configuration
    prog_args = configs_explainer.explainer_arg_parse()

    if prog_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = prog_args.cuda
        print("CUDA", prog_args.cuda)
    else:
        print("Using CPU")

    # Configure the logging directory
    if prog_args.writer:
        path = os.path.join(prog_args.logdir, io_utils.gen_explainer_prefix(prog_args))
        if os.path.isdir(path) and prog_args.clean_log:
            print("Removing existing log dir: ", path)
            if (
                not input("Are you sure you want to remove this directory? (y/n): ")
                .lower()
                .strip()[:1]
                == "y"
            ):
                sys.exit(1)
            shutil.rmtree(path)
        writer = SummaryWriter(path)
    else:
        writer = None

    # Load a model checkpoint
    ckpt = io_utils.load_ckpt(prog_args)
    cg_dict = ckpt["cg"]  # get computation graph
    input_dim = cg_dict["feat"].shape[2]
    num_classes = cg_dict["pred"].shape[2]
    print("Loaded model from {}".format(prog_args.ckptdir))
    print("input dim: ", input_dim, "; num classes: ", num_classes)

    # build model
    print("Method: ", prog_args.method)

    # Explain Graph prediction
    model = models.GcnEncoderGraph(
        input_dim=input_dim,
        hidden_dim=prog_args.hidden_dim,
        embedding_dim=prog_args.output_dim,
        label_dim=num_classes,
        num_layers=prog_args.num_gc_layers,
        bn=prog_args.bn,
        args=prog_args,
    )
    if prog_args.gpu:
        model = model.cuda()

    # Load state_dict (obtained by model.state_dict() when saving checkpoint)
    model.load_state_dict(ckpt["model_state"])
    model = model.eval()

    # Extract the data relative to the chosen graph.
    adj = cg_dict["adj"]
    feat = cg_dict["feat"]
    label = cg_dict["label"]
    pred = cg_dict["pred"]

    graph_idx = prog_args.graph_idx
    sub_adj = adj[graph_idx]
    sub_feat = feat[graph_idx, :]
    sub_label = label[graph_idx]
    neighbors = np.asarray(range(adj.shape[0]))

    sub_adj = np.expand_dims(sub_adj, axis=0)
    sub_feat = np.expand_dims(sub_feat, axis=0)

    adj = torch.tensor(sub_adj, dtype=torch.float)
    x = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
    label = torch.tensor(sub_label, dtype=torch.long)

    # Create explainer
    explainer = Explainer(
        model=model,
        adj=adj,
        x=x,
        label=label,
        args=prog_args,
        writer=writer,
        graph_idx=prog_args.graph_idx,
    )

    # Run explainer.
    train_explainer(
        explainer=explainer, pred=pred, args=prog_args, graph_idx=prog_args.graph_idx
    )
    io_utils.plot_cmap_tb(writer, "tab20", 20, "tab20_cmap")


if __name__ == "__main__":
    main()
