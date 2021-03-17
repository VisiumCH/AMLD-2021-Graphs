from math import sqrt

import torch
import networkx as nx
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
EPS = 1e-15


class GNNExplainer(torch.nn.Module):
    """The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and small subsets node features that play a crucial role in a
    GNN’s graph-prediction.
    """

    coeffs = {
        'edge_size': 0.001,
        'node_feat_size': 1.0,
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
    }

    def __init__(self, model, epochs=100, lr=0.01, log=True):
        """ Initialize GNNExplainer Class.

        Args:
            model (torch.nn.Module): The GNN module to explain.
            epochs (int, optional): The number of epochs to train.
                (default: :obj:`100`)
            lr (float, optional): The learning rate to apply.
                (default: :obj:`0.01`)
        """
        super(GNNExplainer, self).__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.log = log

    def __set_masks__(self, x, edge_index):
        """ Initialize the masks for edges and node features.

        For each module contained in the GNN model, the attribute
        __edge_mask__ is set to the initialized edge mask, so that
        this is automatically taken into account during message passing.

        Args:
            x (torch tensor): node features
            edge_index (torch tensor): pytorch geometric edge index
        """
        (N, F), E = x.size(), edge_index.size(1)

        # Node feature mask.
        self.node_feat_mask = torch.nn.Parameter(torch.randn(F) * 0.1)

        # Edge mask.
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        edge_mask = torch.randn(E) * std
        self.edge_mask = torch.nn.Parameter(edge_mask)
        # TODO: at some point we should enforce the mask to be symmetric?
        # Maybe do this iteratively in explain graph? --> but then should
        # maybe reset the module.__edge_mask__ as below

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

    def __graph_loss__(self, pred_proba, pred_label):
        """Computes the explainer loss function for explanation
        of graph classificaiton tasks.

        Args:
            pred_proba: predicted probabilities for the different
                classes from the model on the masked input
            pred_label: model prediction on the entire original
                graph (i.e. not masked in features or edges)
        Returns:
            loss (torch.tensor): explainer loss function, which
                is a weight sum of different terms.

        """
        # Prediction loss.
        # TODO: 0 removes the batch dimension? So this works only
        # for batchsize = 1 currently?
        loss = -torch.log(pred_proba[0, pred_label])

        # Edge mask size loss.
        m = self.edge_mask.sigmoid()
        loss = loss + self.coeffs['edge_size'] * m.sum()

        # Edge mask entropy loss.
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()

        # Feature mask size loss.
        m = self.node_feat_mask.sigmoid()
        loss = loss + self.coeffs['node_feat_size'] * m.mean()

        return loss

    def explain_graph(self, x, edge_index, batch_index, **kwargs):
        """Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for graph
        classification.

        Args:
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            batch_index (LongTensor): The batch index.

            **kwargs (optional): Additional arguments passed to the GNN module.

        Returns:
            (torch.tensor, torch.tensor): the node feature mask and edge mask
        """

        self.model.eval()
        self.__clear_masks__()

        # Get the initial prediction.
        with torch.no_grad():
            data = Data(x=x, edge_index=edge_index, batch=batch_index)
            model_pred = self.model(data, **kwargs)
            probs_Y = torch.softmax(model_pred, 1)
            pred_label = probs_Y.argmax(dim=-1)

        self.__set_masks__(x, edge_index)
        self.to(x.device)

        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask],
                                     lr=self.lr)

        epoch_losses = []

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0
            optimizer.zero_grad()

            # Mask node features
            h = x * self.node_feat_mask.view(1, -1).sigmoid()

            data = Data(x=h, edge_index=edge_index, batch=batch_index)
            model_pred = self.model(data, **kwargs)
            pred_proba = torch.softmax(model_pred, 1)
            loss = self.__graph_loss__(pred_proba, pred_label)
            loss.backward()
            # print("egde_grad:",self.edge_mask.grad)

            optimizer.step()
            epoch_loss += loss.detach().item()
            # print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
            epoch_losses.append(epoch_loss)

        node_feat_mask = self.node_feat_mask.detach().sigmoid()
        edge_mask = self.edge_mask.detach().sigmoid()

        self.__clear_masks__()

        return node_feat_mask, edge_mask

    def visualize_subgraph(self, edge_index, edge_mask, y=None,
                           threshold=0.5):
        """Visualizes the explanation subgraph.
        Args:
            edge_index (LongTensor): The edge indices.
            edge_mask (Tensor): The edge mask.
            threshold (float): Sets a threshold for visualizing
                important edges.

        Returns:
            (G_original, G_new): two networkx graphs, the original
                one and its subgraph explanation.
        """

        assert edge_mask.size(0) == edge_index.size(1)

        # Filter mask based on threshold
        print('Edge Threshold:', threshold)
        edge_mask = (edge_mask >= threshold).to(torch.float)

        subset = set()
        for index, mask in enumerate(edge_mask):
            node_a = edge_index[0, index]
            node_b = edge_index[1, index]
            if node_a not in subset:
                subset.add(node_a.cpu().item())
            if node_b not in subset:
                subset.add(node_b.cpu().item())

        edge_index_new = [[], []]
        for index, edge in enumerate(edge_mask):
            if edge:
                edge_index_new[0].append(edge_index[0, index].cpu())
                edge_index_new[1].append(edge_index[1, index].cpu())

        data = Data(edge_index=edge_index.cpu(), att=edge_mask, y=y,
                    num_nodes=y.size(0)).to('cpu')
        data_new = Data(edge_index=torch.tensor(edge_index_new).cpu().long(), att=edge_mask, y=y,
                        num_nodes=len(subset)).to('cpu')

        G_original = to_networkx(data, edge_attrs=['att'])
        G_new = to_networkx(data_new, edge_attrs=['att'])

        G_new.remove_nodes_from(list(nx.isolates(G_new)))
        nx.draw(G_new)

        print("Removed {} edges -- K = {} remain.".format(G_original.number_of_edges() -
                                                          G_new.number_of_edges(), G_new.number_of_edges()))
        print("Removed {} nodes -- K = {} remain.".format(G_original.number_of_nodes() -
                                                          G_new.number_of_nodes(), G_new.number_of_nodes()))

        return G_original, G_new
