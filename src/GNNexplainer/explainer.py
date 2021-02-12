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


class Explainer:
    def __init__(
        self,
        model,
        adj,
        feat,
        label,
        pred,
        train_idx,
        args,
        writer=None,
        print_training=True,
        graph_idx=False,
    ):
        self.model = model
        self.model.eval()
        self.adj = adj
        self.feat = feat
        self.label = label
        self.pred = pred
        self.train_idx = train_idx
        self.graph_mode = True
        self.graph_idx = graph_idx
        self.args = args
        self.writer = writer
        self.print_training = print_training

    # Main method
    def explain(self, node_idx, graph_idx=0, unconstrained=False, model="exp"):
        """Explain a single node prediction"""
        # index of the query node in the new adj

        node_idx_new = node_idx
        sub_adj = self.adj[graph_idx]
        sub_feat = self.feat[graph_idx, :]
        sub_label = self.label[graph_idx]
        neighbors = np.asarray(range(self.adj.shape[0]))

        sub_adj = np.expand_dims(sub_adj, axis=0)
        sub_feat = np.expand_dims(sub_feat, axis=0)

        adj = torch.tensor(sub_adj, dtype=torch.float)
        x = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
        label = torch.tensor(sub_label, dtype=torch.long)

        pred_label = np.argmax(self.pred[0][graph_idx], axis=0)
        print("Graph predicted label: ", pred_label)

        explainer = ExplainModule(
            adj=adj,
            x=x,
            model=self.model,
            label=label,
            args=self.args,
            writer=self.writer,
            graph_idx=self.graph_idx,
        )

        if self.args.gpu:
            explainer = explainer.cuda()

        self.model.eval()

        # gradient baseline
        if model == "grad":
            explainer.zero_grad()
            # pdb.set_trace()
            adj_grad = torch.abs(
                explainer.adj_feat_grad(node_idx_new, pred_label[node_idx_new])[0]
            )[graph_idx]
            masked_adj = adj_grad + adj_grad.t()
            masked_adj = nn.functional.sigmoid(masked_adj)
            masked_adj = masked_adj.cpu().detach().numpy() * sub_adj.squeeze()
        else:
            explainer.train()
            # Initialize logger for images in tensorboard.
            logger = ExplainerLogger(explainer)
            begin_time = time.time()
            for epoch in range(self.args.num_epochs):
                explainer.zero_grad()
                explainer.optimizer.zero_grad()
                ypred, adj_atts = explainer(node_idx_new, unconstrained=unconstrained)
                loss = explainer.loss(ypred, pred_label, node_idx_new, epoch)
                loss.backward()

                explainer.optimizer.step()
                if explainer.scheduler is not None:
                    explainer.scheduler.step()

                mask_density = explainer.mask_density()
                if self.print_training:
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
                single_subgraph_label = sub_label.squeeze()

                if self.writer is not None:
                    self.writer.add_scalar("mask/density", mask_density, epoch)
                    self.writer.add_scalar(
                        "optimization/lr",
                        explainer.optimizer.param_groups[0]["lr"],
                        epoch,
                    )
                    if epoch % 25 == 0:
                        logger.log_mask(epoch)
                        #### MODIFIED HERE (COMMENTED)
                        # logger.log_masked_adj(
                        #    node_idx_new, epoch, label=single_subgraph_label
                        # )

                        #### MODIFIED HERE (COMMENTED)
                        # logger.log_adj_grad(
                        #    node_idx_new, pred_label, epoch, label=single_subgraph_label
                        # )

                    if epoch == 0:
                        if self.model.att:
                            # explain node
                            print("adj att size: ", adj_atts.size())
                            adj_att = torch.sum(adj_atts[0], dim=2)
                            # adj_att = adj_att[neighbors][:, neighbors]
                            node_adj_att = adj_att * adj.float().cuda()
                            io_utils.log_matrix(
                                self.writer, node_adj_att[0], "att/matrix", epoch
                            )
                            node_adj_att = node_adj_att[0].cpu().detach().numpy()
                            G = io_utils.denoise_graph(
                                node_adj_att,
                                node_idx_new,
                                threshold=3.8,  # threshold_num=20,
                                max_component=True,
                            )
                            io_utils.log_graph(
                                self.writer,
                                G,
                                name="att/graph",
                                identify_self=not self.graph_mode,
                                nodecolor="label",
                                edge_vmax=None,
                                args=self.args,
                            )
                if model != "exp":
                    break

            print("finished training in ", time.time() - begin_time)
            if model == "exp":
                masked_adj = (
                    explainer.masked_adj[0].cpu().detach().numpy() * sub_adj.squeeze()
                )
            else:
                adj_atts = nn.functional.sigmoid(adj_atts).squeeze()
                masked_adj = adj_atts.cpu().detach().numpy() * sub_adj.squeeze()

        fname = (
            "masked_adj_"
            + io_utils.gen_explainer_prefix(self.args)
            + (
                "node_idx_"
                + str(node_idx)
                + "graph_idx_"
                + str(self.graph_idx)
                + ".npy"
            )
        )

        log_directory_for_masks = os.path.join(
            self.args.logdir, io_utils.gen_explainer_prefix(self.args), "masks"
        )
        if not os.path.exists(log_directory_for_masks):
            os.makedirs(log_directory_for_masks)
        with open(
            os.path.join(
                self.args.logdir,
                io_utils.gen_explainer_prefix(self.args),
                "masks",
                fname,
            ),
            "wb",
        ) as outfile:
            np.save(outfile, np.asarray(masked_adj.copy()))
            print("Saved adjacency matrix to ", fname)
        return masked_adj

    # GRAPH EXPLAINER
    def explain_graphs(self, graph_indices):
        """
        Explain graphs.
        """
        masked_adjs = []

        for graph_idx in graph_indices:
            masked_adj = self.explain(node_idx=0, graph_idx=graph_idx, graph_mode=True)
            G_denoised = io_utils.denoise_graph(
                masked_adj,
                0,
                threshold_num=20,
                feat=self.feat[graph_idx],
                max_component=False,
            )
            label = self.label[graph_idx]
            io_utils.log_graph(
                self.writer,
                G_denoised,
                "graph/graphidx_{}_label={}".format(graph_idx, label),
                identify_self=False,
                nodecolor="feat",
                args=self.args,
            )
            masked_adjs.append(masked_adj)

            G_orig = io_utils.denoise_graph(
                self.adj[graph_idx],
                0,
                feat=self.feat[graph_idx],
                threshold=None,
                max_component=False,
            )

            io_utils.log_graph(
                self.writer,
                G_orig,
                "graph/graphidx_{}".format(graph_idx),
                identify_self=False,
                nodecolor="feat",
                args=self.args,
            )

        # plot cmap for graphs' node features
        io_utils.plot_cmap_tb(self.writer, "tab20", 20, "tab20_cmap")

        return masked_adjs


class ExplainModule(nn.Module):
    def __init__(
        self, adj, x, model, label, args, graph_idx=0, writer=None, use_sigmoid=True
    ):
        super(ExplainModule, self).__init__()
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

        init_strategy = "normal"
        num_nodes = adj.size()[1]
        self.mask, self.mask_bias = self.construct_edge_mask(
            num_nodes, init_strategy=init_strategy
        )

        self.feat_mask = self.construct_feat_mask(x.size(-1), init_strategy="constant")
        params = [self.mask, self.feat_mask]
        if self.mask_bias is not None:
            params.append(self.mask_bias)
        # For masking diagonal entries
        self.diag_mask = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
        if args.gpu:
            self.diag_mask = self.diag_mask.cuda()

        self.scheduler, self.optimizer = train_utils.build_optimizer(args, params)

        self.coeffs = {
            "size": 0.005,
            "feat_size": 1.0,
            "ent": 1.0,
            "feat_ent": 0.1,
            "grad": 0,
            "lap": 1.0,
        }

    def construct_feat_mask(self, feat_dim, init_strategy="normal"):
        mask = nn.Parameter(torch.FloatTensor(feat_dim))
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
                # mask[0] = 2
        return mask

    def construct_edge_mask(self, num_nodes, init_strategy="normal", const_val=1.0):
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        if init_strategy == "normal":
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (num_nodes + num_nodes)
            )
            with torch.no_grad():
                mask.normal_(1.0, std)
                # mask.clamp_(0.0, 1.0)
        elif init_strategy == "const":
            nn.init.constant_(mask, const_val)

        if self.args.mask_bias:
            mask_bias = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = None

        return mask, mask_bias

    def _masked_adj(self):
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

    def forward(
        self, node_idx, unconstrained=False, mask_features=True, marginalize=False
    ):
        x = self.x.cuda() if self.args.gpu else self.x

        if unconstrained:
            sym_mask = torch.sigmoid(self.mask) if self.use_sigmoid else self.mask
            self.masked_adj = (
                torch.unsqueeze((sym_mask + sym_mask.t()) / 2, 0) * self.diag_mask
            )
        else:
            self.masked_adj = self._masked_adj()
            if mask_features:
                feat_mask = (
                    torch.sigmoid(self.feat_mask)
                    if self.use_sigmoid
                    else self.feat_mask
                )
                if marginalize:
                    std_tensor = torch.ones_like(x, dtype=torch.float) / 2
                    mean_tensor = torch.zeros_like(x, dtype=torch.float) - x
                    z = torch.normal(mean=mean_tensor, std=std_tensor)
                    x = x + z * (1 - feat_mask)
                else:
                    x = x * feat_mask

        ypred, adj_att = self.model(x, self.masked_adj)
        if self.graph_mode:
            res = nn.Softmax(dim=0)(ypred[0])
        else:
            node_pred = ypred[self.graph_idx, node_idx, :]
            res = nn.Softmax(dim=0)(node_pred)
        return res, adj_att

    def loss(self, pred, pred_label, node_idx, epoch):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        mi_obj = False
        if mi_obj:
            pred_loss = -torch.sum(pred * torch.log(pred))
        else:
            pred_label_node = pred_label if self.graph_mode else pred_label[node_idx]
            gt_label_node = self.label if self.graph_mode else self.label[0][node_idx]
            logit = pred[gt_label_node]
            pred_loss = -torch.log(logit)
        # size
        mask = self.mask
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            mask = nn.ReLU()(self.mask)
        size_loss = self.coeffs["size"] * torch.sum(mask)

        # pre_mask_sum = torch.sum(self.feat_mask)
        feat_mask = (
            torch.sigmoid(self.feat_mask) if self.use_sigmoid else self.feat_mask
        )
        feat_size_loss = self.coeffs["feat_size"] * torch.mean(feat_mask)

        # entropy
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = self.coeffs["ent"] * torch.mean(mask_ent)

        feat_mask_ent = -feat_mask * torch.log(feat_mask) - (1 - feat_mask) * torch.log(
            1 - feat_mask
        )

        feat_mask_ent_loss = self.coeffs["feat_ent"] * torch.mean(feat_mask_ent)

        # laplacian
        D = torch.diag(torch.sum(self.masked_adj[0], 0))
        m_adj = self.masked_adj if self.graph_mode else self.masked_adj[self.graph_idx]
        L = D - m_adj
        pred_label_t = torch.tensor(pred_label, dtype=torch.float)
        if self.args.gpu:
            pred_label_t = pred_label_t.cuda()
            L = L.cuda()
        if self.graph_mode:
            lap_loss = 0
        else:
            lap_loss = (
                self.coeffs["lap"]
                * (pred_label_t @ L @ pred_label_t)
                / self.adj.numel()
            )

        loss = pred_loss + size_loss + lap_loss + mask_ent_loss + feat_size_loss
        if self.writer is not None:
            self.writer.add_scalar("optimization/size_loss", size_loss, epoch)
            self.writer.add_scalar("optimization/feat_size_loss", feat_size_loss, epoch)
            self.writer.add_scalar("optimization/mask_ent_loss", mask_ent_loss, epoch)
            self.writer.add_scalar(
                "optimization/feat_mask_ent_loss", mask_ent_loss, epoch
            )
            # self.writer.add_scalar('optimization/grad_loss', grad_loss, epoch)
            self.writer.add_scalar("optimization/pred_loss", pred_loss, epoch)
            self.writer.add_scalar("optimization/lap_loss", lap_loss, epoch)
            self.writer.add_scalar("optimization/overall_loss", loss, epoch)
        return loss


class ExplainerLogger:
    def __init__(self, explainer):
        self.explainer = explainer

    def log_masked_adj(self, node_idx, epoch, name="mask/graph", label=None):
        # use [0] to remove the batch dim
        masked_adj = self.explainer.masked_adj[0].cpu().detach().numpy()

        G = io_utils.denoise_graph(
            masked_adj,
            node_idx,
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

    def adj_feat_grad(self, node_idx, pred_label_node):
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

    def log_adj_grad(self, node_idx, pred_label, epoch, label=None):
        log_adj = False

        predicted_label = pred_label
        # adj_grad, x_grad = torch.abs(self.adj_feat_grad(node_idx, predicted_label)[0])[0]
        adj_grad, x_grad = self.explainer.adj_feat_grad(node_idx, predicted_label)
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

        # only for graph mode since many node neighborhoods for syn tasks are relatively large for
        # visualization
        G = io_utils.denoise_graph(
            masked_adj,
            node_idx,
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
            node_idx,
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
