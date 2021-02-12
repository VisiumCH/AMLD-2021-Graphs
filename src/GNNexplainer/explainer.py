import math
import time
import os

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import tensorboardX.utils

import torch
import torch.nn as nn
from torch.autograd import Variable

import sklearn.metrics as metrics
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.cluster import DBSCAN

import pdb

import src.utils.io_utils as io_utils
import src.utils.parser_utils as parser_utils
import src.utils.train_utils as train_utils
import src.utils.graph_utils as graph_utils


use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


def train_explainer(
    explainer,
    pred,
    args,
    print_training=True,
    graph_idx=0,
    method="exp",
):
    graph_mode = True

    pred_label = np.argmax(pred[0][graph_idx], axis=0)
    print("Graph predicted label: ", pred_label)

    if args.gpu:
        explainer = explainer.cuda()

    # gradient baseline
    if method == "grad":
        explainer.zero_grad()
        # pdb.set_trace()
        adj_grad = torch.abs(explainer.adj_feat_grad(pred_label)[0])[graph_idx]
        masked_adj = adj_grad + adj_grad.t()
        masked_adj = nn.functional.sigmoid(masked_adj)
        masked_adj = masked_adj.cpu().detach().numpy() * sub_adj.squeeze()
    else:
        explainer.train()
        # Initialize logger for images in tensorboard.
        logger = ExplainerLogger(explainer)
        begin_time = time.time()
        for epoch in range(args.num_epochs):
            explainer.zero_grad()
            explainer.optimizer.zero_grad()
            ypred, adj_atts = explainer()  # Equivalent to using forward.
            loss = explainer.loss(ypred, epoch)
            loss.backward()

            explainer.optimizer.step()
            if explainer.scheduler is not None:
                explainer.scheduler.step()

            mask_density = explainer.mask_density()
            if print_training:
                print(
                    "epoch: ",
                    epoch,
                    "; loss: ",
                    loss.item(),
                    "; mask density: ",
                    mask_density.item(),
                    "; pred: ",
                    ypred,
                )
            single_subgraph_label = explainer.label

            if explainer.writer is not None:
                explainer.writer.add_scalar("mask/density", mask_density, epoch)
                explainer.writer.add_scalar(
                    "optimization/lr",
                    explainer.optimizer.param_groups[0]["lr"],
                    epoch,
                )
                if epoch % 25 == 0:
                    logger.log_mask(epoch)
                    #### MODIFIED HERE (COMMENTED)
                    # logger.log_masked_adj(epoch, label=single_subgraph_label
                    # )

                    #### MODIFIED HERE (COMMENTED)
                    # logger.log_adj_grad(pred_label, epoch, label=single_subgraph_label
                    # )

                if epoch == 0:
                    if explainer.model.att:
                        # explain node
                        print("adj att size: ", adj_atts.size())
                        adj_att = torch.sum(adj_atts[0], dim=2)
                        # adj_att = adj_att[neighbors][:, neighbors]
                        node_adj_att = adj_att * adj.float().cuda()
                        io_utils.log_matrix(
                            explainer.writer, node_adj_att[0], "att/matrix", epoch
                        )
                        node_adj_att = node_adj_att[0].cpu().detach().numpy()
                        G = io_utils.denoise_graph(
                            node_adj_att,
                            threshold=3.8,  # threshold_num=20,
                            max_component=True,
                        )
                        io_utils.log_graph(
                            explainer.writer,
                            G,
                            name="att/graph",
                            identify_self=not graph_mode,
                            nodecolor="label",
                            edge_vmax=None,
                            args=args,
                        )
            if method != "exp":
                break

        print("finished training in ", time.time() - begin_time)
        if method == "exp":
            masked_adj = (
                explainer.masked_adj[0].cpu().detach().numpy()
                * explainer.adj.numpy().squeeze()
            )
        else:
            adj_atts = nn.functional.sigmoid(adj_atts).squeeze()
            masked_adj = (
                adj_atts.cpu().detach().numpy() * explainer.adj().numpy().squeeze()
            )

    # Save explanations to file.
    fname = (
        "masked_adj_"
        + io_utils.gen_explainer_prefix(args)
        + ("graph_idx_" + str(graph_idx) + ".npy")
    )

    log_directory_for_masks = os.path.join(
        args.logdir, io_utils.gen_explainer_prefix(args), "masks"
    )
    if not os.path.exists(log_directory_for_masks):
        os.makedirs(log_directory_for_masks)
    with open(
        os.path.join(
            args.logdir,
            io_utils.gen_explainer_prefix(args),
            "masks",
            fname,
        ),
        "wb",
    ) as outfile:
        np.save(outfile, np.asarray(masked_adj.copy()))
        print("Saved adjacency matrix to ", fname)
    return masked_adj


class Explainer(nn.Module):
    def __init__(
        self, adj, x, model, label, args, graph_idx=0, writer=None, use_sigmoid=True
    ):
        super(Explainer, self).__init__()
        self.adj = adj
        self.x = x
        self.model = model
        self.label = label
        self.graph_idx = graph_idx
        self.args = args
        self.writer = writer
        self.mask_act = args.mask_act
        self.use_sigmoid = use_sigmoid
        self.graph_mode = True
        # Relative weights for the terms in the loss function.
        self.coeffs = {
            "size": 0.005,
            "feat_size": 1.0,
            "ent": 1.0,
            "feat_ent": 0.1,
            "grad": 0,
            "lap": 1.0,
        }
        num_nodes = adj.size()[1]
        init_strategy = "normal"
        # Initialize the edge mask to be optimized.
        self.mask, self.mask_bias = self.construct_edge_mask(
            num_nodes, init_strategy=init_strategy
        )
        # Initialize the feature mask to be optimized.
        self.feat_mask = self.construct_feat_mask(x.size(-1), init_strategy="constant")
        params = [self.mask, self.feat_mask]
        if self.mask_bias is not None:
            params.append(self.mask_bias)
        # For masking diagonal entries.
        self.diag_mask = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
        if args.gpu:
            self.diag_mask = self.diag_mask.cuda()

        self.scheduler, self.optimizer = train_utils.build_optimizer(args, params)

    def construct_feat_mask(self, feat_dim, init_strategy="normal"):
        """Initialize the feature mask. init_strategy is a string specifying
        the chosen initialization strategy (can be 'costant' or 'normal'.)
        """
        mask = nn.Parameter(torch.FloatTensor(feat_dim))
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
        return mask

    def construct_edge_mask(self, num_nodes, init_strategy="normal", const_val=1.0):
        """Initialize the edge mask. init_strategy is a string specifying
        the chosen initialization strategy (eg 'costant' or 'normal'.)
        """
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        if init_strategy == "normal":
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (num_nodes + num_nodes)
            )
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "const":
            nn.init.constant_(mask, const_val)
        if self.args.mask_bias:
            mask_bias = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = None

        return mask, mask_bias

    def _masked_adj(self):
        """Computes the masked adjacency matrix of the graph. Since
        we work with undirected graphs, we make the mask symmetric.
        Self-loops are also excluded using a diagonal mask.
        """
        sym_mask = self.mask
        if self.mask_act == "sigmoid":
            sym_mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            sym_mask = nn.ReLU()(self.mask)
        sym_mask = (sym_mask + sym_mask.t()) / 2
        adj = self.adj.cuda() if self.args.gpu else self.adj
        masked_adj = adj * sym_mask
        if self.args.mask_bias:
            bias = (self.mask_bias + self.mask_bias.t()) / 2
            bias = nn.ReLU6()(bias * 6) / 6
            masked_adj += (bias + bias.t()) / 2
        return masked_adj * self.diag_mask

    def mask_density(self):
        mask_sum = torch.sum(self._masked_adj()).cpu()
        adj_sum = torch.sum(self.adj)
        return mask_sum / adj_sum

    def forward(self, mask_features=True, marginalize=False):
        """Computes the model prediction on the masked graph with masked features.
        Returns the model predictions and adjacency attention (if available).
        """
        x = self.x.cuda() if self.args.gpu else self.x
        self.masked_adj = self._masked_adj()
        if mask_features:
            feat_mask = (
                torch.sigmoid(self.feat_mask) if self.use_sigmoid else self.feat_mask
            )
            if marginalize:
                std_tensor = torch.ones_like(x, dtype=torch.float) / 2
                mean_tensor = torch.zeros_like(x, dtype=torch.float) - x
                z = torch.normal(mean=mean_tensor, std=std_tensor)
                x = x + z * (1 - feat_mask)
            else:
                x = x * feat_mask

        ypred, adj_att = self.model(x, self.masked_adj)
        res = nn.Softmax(dim=0)(ypred[0])

        return res, adj_att

    def loss(self, pred, epoch):
        """
        Args:
            pred: prediction made by current model (with current mask).
            epoch: training epoch.
        """
        # Prediction loss.
        gt_label = self.label
        logit = pred[gt_label]
        pred_loss = -torch.log(logit)
        # Adjacency mask size loss.
        mask = self.mask
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            mask = nn.ReLU()(self.mask)
        size_loss = self.coeffs["size"] * torch.sum(mask)
        # Feature mask size loss.
        feat_mask = (
            torch.sigmoid(self.feat_mask) if self.use_sigmoid else self.feat_mask
        )
        feat_size_loss = self.coeffs["feat_size"] * torch.mean(feat_mask)
        # Adjacency mask entropy loss.
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = self.coeffs["ent"] * torch.mean(mask_ent)
        # Feature mask entropy loss.
        feat_mask_ent = -feat_mask * torch.log(feat_mask) - (1 - feat_mask) * torch.log(
            1 - feat_mask
        )
        feat_mask_ent_loss = self.coeffs["feat_ent"] * torch.mean(feat_mask_ent)
        # Total loss.
        loss = pred_loss + size_loss + mask_ent_loss + feat_size_loss

        # Log data to tensorboard.
        if self.writer is not None:
            self.writer.add_scalar("optimization/size_loss", size_loss, epoch)
            self.writer.add_scalar("optimization/feat_size_loss", feat_size_loss, epoch)
            self.writer.add_scalar("optimization/mask_ent_loss", mask_ent_loss, epoch)
            self.writer.add_scalar(
                "optimization/feat_mask_ent_loss", mask_ent_loss, epoch
            )
            self.writer.add_scalar("optimization/pred_loss", pred_loss, epoch)
            # self.writer.add_scalar("optimization/lap_loss", lap_loss, epoch)
            self.writer.add_scalar("optimization/overall_loss", loss, epoch)

        return loss


class ExplainerLogger:
    def __init__(self, explainer):
        self.explainer = explainer

    def log_masked_adj(self, epoch, name="mask/graph", label=None):
        # use [0] to remove the batch dim
        masked_adj = self.explainer.masked_adj[0].cpu().detach().numpy()

        G = io_utils.denoise_graph(
            masked_adj,
            feat=self.x[0],
            threshold=0.2,  # threshold_num=20,
            max_component=True,
        )
        io_utils.log_graph(
            self.writer,
            G,
            name=name,
            identify_self=False,
            nodecolor="feat",
            epoch=epoch,
            label_node_feat=True,
            edge_vmax=None,
            args=self.args,
        )

    def adj_feat_grad(self, pred_label_node):
        self.explainer.model.zero_grad()
        self.explainer.adj.requires_grad = True
        self.explainer.x.requires_grad = True
        if self.explainer.adj.grad is not None:
            self.explainer.adj.grad.zero_()
            self.explainer.x.grad.zero_()
        if self.explainer.args.gpu:
            adj = self.explainer.adj.cuda()
            x = self.explainer.x.cuda()
            label = self.explainer.label.cuda()
        else:
            x, adj = self.explainer.x, self.adj
        ypred, _ = self.explainer.model(x, adj)

        logit = nn.Softmax(dim=0)(ypred[0])
        logit = logit[pred_label_node]
        loss = -torch.log(logit)
        loss.backward()
        return self.explainer.adj.grad, self.explainer.x.grad

    def log_adj_grad(self, pred_label, epoch, label=None):
        log_adj = False

        predicted_label = pred_label
        adj_grad, x_grad = self.explainer.adj_feat_grad(predicted_label)
        adj_grad = torch.abs(adj_grad)[0]
        x_grad = torch.sum(x_grad[0], 0, keepdim=True).t()
        adj_grad = (adj_grad + adj_grad.t()) / 2
        adj_grad = (adj_grad * self.explainer.adj).squeeze()
        if log_adj:
            io_utils.log_matrix(
                self.explainer.writer, adj_grad, "grad/adj_masked", epoch
            )
            self.explainer.adj.requires_grad = False
            io_utils.log_matrix(
                self.explainer.writer,
                self.explainer.adj.squeeze(),
                "grad/adj_orig",
                epoch,
            )

        masked_adj = self.expplainer.masked_adj[0].cpu().detach().numpy()

        # Denoise graph since many node neighborhoods for such tasks are relatively large for
        # visualization
        G = io_utils.denoise_graph(
            masked_adj,
            feat=self.x[0],
            threshold=None,
            max_component=False,
        )
        io_utils.log_graph(
            self.writer,
            G,
            name="grad/graph_orig",
            epoch=epoch,
            identify_self=False,
            label_node_feat=True,
            nodecolor="feat",
            edge_vmax=None,
            args=self.args,
        )
        io_utils.log_matrix(self.writer, x_grad, "grad/feat", epoch)

        adj_grad = adj_grad.detach().numpy()
        print("GRAPH model")
        G = io_utils.denoise_graph(
            adj_grad,
            feat=self.x[0],
            threshold=0.0003,  # threshold_num=20,
            max_component=True,
        )
        io_utils.log_graph(
            self.explainer.writer,
            G,
            name="grad/graph",
            epoch=epoch,
            identify_self=False,
            label_node_feat=True,
            nodecolor="feat",
            edge_vmax=None,
            args=self.args,
        )

    def log_mask(self, epoch):
        plt.switch_backend("agg")
        fig = plt.figure(figsize=(4, 3), dpi=400)
        plt.imshow(
            self.explainer.mask.cpu().detach().numpy(), cmap=plt.get_cmap("BuPu")
        )
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")

        plt.tight_layout()
        fig.canvas.draw()
        self.explainer.writer.add_image(
            "mask/mask", tensorboardX.utils.figure_to_image(fig), epoch
        )

        io_utils.log_matrix(
            self.explainer.writer,
            torch.sigmoid(self.explainer.feat_mask),
            "mask/feat_mask",
            epoch,
        )

        fig = plt.figure(figsize=(4, 3), dpi=400)
        # use [0] to remove the batch dim
        plt.imshow(
            self.explainer.masked_adj[0].cpu().detach().numpy(),
            cmap=plt.get_cmap("BuPu"),
        )
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")

        plt.tight_layout()
        fig.canvas.draw()
        self.explainer.writer.add_image(
            "mask/adj", tensorboardX.utils.figure_to_image(fig), epoch
        )

        if self.explainer.args.mask_bias:
            fig = plt.figure(figsize=(4, 3), dpi=400)
            # use [0] to remove the batch dim
            plt.imshow(self.mask_bias.cpu().detach().numpy(), cmap=plt.get_cmap("BuPu"))
            cbar = plt.colorbar()
            cbar.solids.set_edgecolor("face")

            plt.tight_layout()
            fig.canvas.draw()
            self.explainer.writer.add_image(
                "mask/bias", tensorboardX.utils.figure_to_image(fig), epoch
            )
