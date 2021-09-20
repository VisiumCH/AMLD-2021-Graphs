"""Module implementing GNNExplainer class"""
from math import sqrt

import torch
from torch_geometric.nn import MessagePassing

EPS = 1e-15

################################################################################
# EXPLAINER


class GNNExplainer(torch.nn.Module):
    """The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and small subsets node features that play a crucial role in a
    GNN’s graph-prediction.
    """
    def __init__(
        self,
        model,
        epochs=100,
        lr=0.04,
        edge_size=0.01,
        edge_ent=1.0,
        node_feat_size=0.07,
        node_feat_ent=0.1,
    ):
        """ Initialize GNNExplainer Class.

        Args:
            model (torch.nn.Module): A (pretrained) GNN module to explain.
            epochs (int, optional): The number of epochs to train.
                (default: :obj:`100`)
            lr (float, optional): The learning rate to apply.
                (default: :obj:`0.04`)
        """
        super(GNNExplainer, self).__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.coeffs = dict(
            edge_size=edge_size,
            node_feat_size=node_feat_size,
            edge_ent=edge_ent,
            node_feat_ent=node_feat_ent,
        )

    def __set_masks__(self, x, edge_index):
        """ Initialize the masks for edges and node features.

        For each module contained in the GNN model, the attribute
        __edge_mask__ is set to the initialized edge mask, so that
        it is handled in message passing by pytorch geometric.

        Args:
            x (torch tensor): node features
            edge_index (torch tensor): pytorch geometric edge index
        """
        (N, F), E = x.size(), edge_index.size(1)

        # Node feature mask.
        self.node_feat_mask = torch.nn.Parameter(torch.randn(F) * 0.1)

        # Edge mask.
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        """Deletes the node and edge masks.
        """
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.node_feat_masks = None
        self.edge_mask = None

    def graph_loss(self, x: torch.Tensor, edge_index: torch.LongTensor,
                   batch_index: torch.LongTensor, expl_label: int,
                   **kwargs) -> torch.Tensor:
        """Computes the explainer loss function for explanation
        of graph classificaiton tasks.

        Args:
            x : torch.Tensor
                Feature matrix of datapoint to explain.
            edge_index : torch.LongTensor
                A Tensor that defines the underlying graph connectivity/message
                passing flow. `edge_index` holds the indices of a general (sparse)
                assignment matrix of shape `[N, M]`. Its shape must be defined as
                `[2, num_messages]`, where messages from nodes in `edge_index[0]`
                are sent to nodes in `edge_index[1]`.
            batch_index : torch.LongTensor
                Column vector which maps each node to its respective graph in the batch.
            expl_label : int
                Label with respect to which we want the explanation.
            **kwargs : optional
                Additional keyword arguments to be passed to the GNN model.

        Returns:
            torch.Tensor
                explainer loss function, which is a weighted sum of different terms.
        """
        # Mask node features
        h = x * self.node_feat_mask.view(1, -1).sigmoid()

        # Compute model output (we assume the model already gives log probabilities)
        model_pred_log_proba = self.model(h, edge_index, batch_index, **kwargs)

        # Prediction loss.
        loss = -model_pred_log_proba[:, expl_label]

        # Edge mask size loss.
        edge_mask = self.edge_mask.sigmoid()
        loss = loss + self.coeffs['edge_size'] * edge_mask.sum()

        # Edge mask entropy loss.
        ent = -edge_mask * torch.log(edge_mask + EPS) - (
            1 - edge_mask) * torch.log(1 - edge_mask + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()

        # Feature mask size loss.
        feat_mask = self.node_feat_mask.sigmoid()
        loss = loss + self.coeffs['node_feat_size'] * feat_mask.mean()

        return loss.sum()

    def explain_graph(self, x, edge_index, batch_index, expl_label: int,
                      **kwargs) -> (torch.nn.Parameter, torch.nn.Parameter):
        """Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for graph
        classification.

        Args:
            x : Tensor
                The node feature matrix.
            edge_index : torch.LongTensor
                A Tensor that defines the underlying graph connectivity/message passing flow.
            batch_index : torch.LongTensor
                Column vector which maps each node to its respective graph in the batch.
            expl_label : int
                Label with respect to which we want the explanation.
            **kwargs : optional
                Additional keyword arguments to be passed to the GNN model.

        Returns:
            torch.Tensor, torch.Tensor
                The node feature mask and edge mask
        """

        self.model.eval()
        self.__clear_masks__()

        self.__set_masks__(x, edge_index)
        self.to(x.device)

        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask],
                                     lr=self.lr)

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            loss = self.graph_loss(x, edge_index, batch_index, expl_label,
                                   **kwargs)
            loss.backward()

            optimizer.step()

        node_feat_mask = self.node_feat_mask.detach().sigmoid()
        edge_mask = self.edge_mask.detach().sigmoid()

        self.__clear_masks__()

        return node_feat_mask, edge_mask
