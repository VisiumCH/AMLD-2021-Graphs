from collections import defaultdict
from math import sqrt

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.colorbar import ColorbarBase
from mpl_toolkits.axes_grid1 import make_axes_locatable

import networkx as nx  # Graph manipulation library

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx  # Conversion function

import torch
from torch.nn.functional import one_hot

EPS = 1e-15

################################################################################
# VISUALIZATION FUNCTIONS


def to_molecule(torch_graph: Data) -> nx.Graph:
    """Convert a Pytorch Geometric Data, with attribute _symbols_, into a networkx graph representing a molecule.

    Parameters
    ----------
    torch_graph : Data
        Input Pytoch graph

    Returns
    -------
    nx.Graph
        Converted graph
    """
    G = to_networkx(
        torch_graph,
        to_undirected=True,
        node_attrs=["symbols"]
    )
    return G


def plot_mol(
    G: nx.Graph,
    edge_mask=None,
    edge_type=None,
    threshold=None,
    drop_isolates=False,
    ax=None
):
    """Draw molecule.

    Parameters
    ----------
    G : nx.Graph
        Graph with _symbols_ node attribute.
    edge_mask : dict, optional
        Dictionary of edge/float items, by default None.
        If given the edges will be color coded. If `treshold` is given,
        `edge_mask` is used to filter edges with mask lower than value.
    edge_type : array of float, optional
        Type of bond encoded as a number, by default None.
        If given, bond width will represent the type of bond.
    threshold : float, optional
        Minumum value of `edge_mask` to include, by default None.
        Only used if `edge_mask` is given.
    drop_isolates : bool, optional
        Wether to remove isolated nodes, by default True if `treshold` is given else False.
    ax : matplotlib.axes.Axes, optional
        Axis on which to draw the molecule, by default None
    """
    if drop_isolates is None:
        drop_isolates = True if threshold else False
    if ax is None:
        fig, ax = plt.subplots(dpi=120)

    pos = nx.planar_layout(G)
    pos = nx.kamada_kawai_layout(G, pos=pos)

    if edge_type is None:
        widths = None
    else:
        widths = edge_type + 1

    edgelist = G.edges()

    if edge_mask is None:
        edge_color = 'black'
    else:
        if threshold is not None:
            edgelist = [
                (u,v) for u, v in G.edges() if edge_mask[(u,v)] > threshold
            ]

        edge_color = [edge_mask[(u, v)] for u, v in edgelist]

    nodelist = G.nodes()
    if drop_isolates:
        if not edgelist:  # Prevent errors
            print("No nodes left to show !")
            return

        nodelist = list(set.union(*map(set, edgelist)))

    node_labels = {
        node: data["symbols"] for node, data in G.nodes(data=True)
        if node in nodelist
    }

    nx.draw_networkx(
        G, pos=pos,
        nodelist=nodelist,
        node_size=200,
        labels=node_labels,
        width=widths,
        edgelist=edgelist,
        edge_color=edge_color, edge_cmap=plt.cm.Blues,
        edge_vmin=0., edge_vmax=1.,
        node_color='azure',
        ax=ax
    )

    if ax is None:
        fig.tight_layout()
        plt.show()


def compute_accuracy(
    model: torch.nn.Module,
    loader: torch.data.DataLoader,
    device=None
):
    """Compute accuracy of input model over all samples from the loader.

    Parameters
    ----------
    model : torch.nn.Module
        NN model
    loader : torch.data.DataLoader
        Data loader to evaluate on
    device : torch.device, optional
        Device to use, by default None.
        If None uses cuda if available else cpu.

    Returns
    -------
    float
        Accuracy in [0,1]
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()

    y_preds = []
    y_trues = []

    for data in loader:  # Iterate in batches over the training/test dataset.
        data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        y_preds.append(out.argmax(dim=1))  # Use the class with highest probability.
        y_trues.append(data.y)  # Check against ground-truth labels.

    y_pred = torch.cat(y_preds).flatten()
    y_true = torch.cat(y_trues).flatten()

    return torch.sum(y_pred == y_true).item() / len(y_true)  # Derive ratio of correct predictions.


def plot_cm(cm, display_labels=["Mutag", "Non Mutag"]):
    """Plot confusion matrix with heatmap.

    Parameters
    ----------
    cm : array
        Confusion matrix
    display_labels : list, optional
        Labels of classes in confusion matrix, by default ["Mutag", "Non Mutag"]
    """
    # Set fontsize for plots
    font = {"size": 20}
    rc("font", **font)

    # Plot confusion matrix
    f, axes = plt.subplots(1, 1, figsize=(7, 7), sharey="row")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=axes, xticks_rotation=45, cmap="Blues", values_format='d')
    disp.im_.colorbar.remove()
    disp.ax_.set_xlabel("Predicted label", fontsize=20)
    disp.ax_.set_ylabel("True label", fontsize=20)


def compute_cm(model, loader_test, device=None):
    """Compute confusion matrix of input model over all samples from the loader.

    Parameters
    ----------
    model : torch.nn.Module
        NN model
    loader_test : torch.data.DataLoader
        Data loader
    device : torch.device, optional
        Device to use, by default None.
        If None uses cuda if available else cpu.

    Returns
    -------
    array
        Confusion matrix
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        model.eval()
        test_batch = next(iter(loader_test)).to(device)

        y_pred = model(test_batch.x, test_batch.edge_index, test_batch.batch).argmax(dim=1)
        y_true = test_batch.y

    # Compute confusion matrix
    cm = confusion_matrix(y_true.flatten().cpu(), y_pred.cpu())

    return cm


class GNNExplainer(torch.nn.Module):
    """The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and small subsets node features that play a crucial role in a
    GNN’s graph-prediction.
    """

    coeffs = {
        'edge_size': 0.1,
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
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)

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

    def graph_loss(
        self,
        x: torch.tensor,
        edge_index: torch.tensor,
        batch_index: torch.tensor,
        expl_label: int,
        **kwargs
    ) -> torch.tensor:
        """Computes the explainer loss function for explanation
        of graph classificaiton tasks.

        Returns:
            loss (torch.tensor): explainer loss function, which
                is a weight sum of different terms.
        """
        # Mask node features
        h = x * self.node_feat_mask.view(1,-1).sigmoid()

        # Compute model output
        model_pred = self.model(h, edge_index, batch_index, **kwargs)
        pred_proba = torch.softmax(model_pred, 1)

        # Prediction loss.
        loss = -torch.log(pred_proba[:, expl_label])

        # Edge mask size loss.
        edge_mask = self.edge_mask.sigmoid()
        loss = loss + self.coeffs['edge_size'] * edge_mask.sum()

        # Edge mask entropy loss.
        ent = -edge_mask * torch.log(edge_mask + EPS) - (1 - edge_mask) * torch.log(1 - edge_mask + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()

        # Feature mask size loss.
        feat_mask = self.node_feat_mask.sigmoid()
        loss = loss + self.coeffs['node_feat_size'] * feat_mask.mean()

        return loss.sum()

    def explain_graph(self, x, edge_index, batch_index, expl_label: int, **kwargs):
        """Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for graph
        classification.

        Args:
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            batch_index (LongTensor): The batch index.
            expl_label (int): Label against which compute cross entropy

            **kwargs (optional): Additional arguments passed to the GNN module.

        Returns:
            (torch.tensor, torch.tensor): the node feature mask and edge mask
        """

        # if len(data.y) > 1:
        #     raise NotImplementedError(f"Can only explain one molecule, recieved {len(data.y)}")

        self.model.eval()
        self.__clear_masks__()

        self.__set_masks__(x, edge_index)
        self.to(x.device)

        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask],
                                     lr=self.lr)

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            loss = self.graph_loss(x, edge_index, batch_index, expl_label, **kwargs)
            loss.backward()

            optimizer.step()

        node_feat_mask = self.node_feat_mask.detach().sigmoid()
        edge_mask = self.edge_mask.detach().sigmoid()

        self.__clear_masks__()

        return node_feat_mask, edge_mask


def explain_graph_visualized(data):
    data.to(device)
    x = data.x
    edge_index = data.edge_index
    model.eval()

    # Initialize explainer
    explainer = GNNExplainer(model, epochs=200).to(device)
    # Train explainer
    model_args = (data.x, data.edge_index, data.batch)
    GNNExp_feat_mask, GNNExp_edge_mask = explainer.explain_graph(
        *model_args,
        model(*model_args).argmax(dim=1)
    )

    mol = to_molecule(data)

    GNNExp_edge_mask_dict = mask_to_dict(
       GNNExp_edge_mask,
       data
    )

    fig, ax = plt.subplots(
        1, 2,
        sharex=True,
        sharey=True,
        dpi=150, figsize=(10,4)
    )

    fig.tight_layout()

    plot_mol(
        mol,
        edge_type=(data.edge_attr.argmax(dim=1) + 1).to("cpu").numpy(),
        ax=ax[1]
    )
    
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("left", "4%", pad="1%")
    cbar = ColorbarBase(
        ax=cax, cmap=plt.cm.Blues,
        ticklocation="left",
    )

    def threshold_plot(threshold):
        xlim, ylim = ax[0].get_xlim(), ax[0].get_ylim()
        ax[0].clear()
        ax[0].set_xlim(xlim), ax[0].set_ylim(ylim)
        
        plot_mol(
            mol,
            GNNExp_edge_mask_dict,
            edge_type=(data.edge_attr.argmax(dim=1) + 1).to("cpu").numpy(),
            threshold=threshold,
            ax=ax[0]
        )

        cbar.add_lines([threshold], [[1, .5, 0, 1]], 3)

        display(fig)

    plt.close()

    return interact(
        threshold_plot,
        threshold=FloatSlider(value=0.1, min=0., max=.99, step=0.05)
    )


def big_interact(which):
    idces = []
    test_size = len(y_true)
    if which == "true_pos":
        idces += [
            i for i in range(test_size)
            if y_pred[i] == y_true[i] == 0
        ]
    if which == "true_neg":
        idces += [
            i for i in range(test_size)
            if y_pred[i] == y_true[i] == 1
        ]
    if which == "false_neg":
        idces += [
            i for i in range(test_size)
            if y_pred[i] != y_true[i] == 0
        ]
    if which == "false_pos":
        idces += [
            i for i in range(test_size)
            if y_pred[i] != y_true[i] == 1
        ]

    single_loader_te = DataLoader(
        dataset[idx_val:], batch_size=1, shuffle=False
    )

    
    return interact(
        lambda data: explain_graph_visualized(data),
        data=[graph for i, graph in enumerate(list(single_loader_te)) if i in idces],
    )


def mask_to_dict(edge_mask, data):
    """
    Conver an `edge_mask` in pytorch geometric format to a networkx compatible
    dictionary (_{(n1, n2) : mask_value}_).
    Multiple edge appearences are averaged.
    """
    edge_mask_dict = defaultdict(float)
    counts = defaultdict(int)

    for val, u, v in zip(edge_mask.to("cpu").numpy(), *data.edge_index):
        u, v = u.item(), v.item()
        if u > v:
            u, v = v, u
        edge_mask_dict[(u, v)] += val
        counts[(u,v)] += 1

    for edge, count in counts.items():
        edge_mask_dict[edge] /= count

    return edge_mask_dict

################################################################################
# UNUSED CODE


class OneHot(object):
    """
    Takes LongTensor with index values of shape (*) and returns a tensor of shape
    (*, num_classes) that have zeros everywhere except where the index of last
    dimension matches the corresponding value of the input tensor, in which case
    it will be 1.
    """
    def __init__(self, column=0, num_classes=-1):
        self.column = column
        self.num_classes = num_classes

    def __call__(self, graph):
        enc = one_hot(
            graph.x[:, self.column],
            num_classes=self.num_classes
        )

        x = torch.empty(
            (enc.shape[0], enc.shape[1] + graph.x.shape[1] - 1)
        )

        x[:, :self.column] = graph.x[:, :self.column]
        x[:, self.column:(self.column + enc.shape[1])] = enc
        x[:, (self.column + enc.shape[1]):] = graph.x[:, (self.column + 1):]
        graph.x = x

        return graph

    def __repr__(self):
        return f"OneHot(self.column={self.column}, num_classes={self.num_classes})"