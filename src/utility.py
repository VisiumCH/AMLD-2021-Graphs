from .explain import GNNExplainer
from .visualization import plot_mol

from IPython.display import display
from ipywidgets import interact, FloatSlider

import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch

from torch_geometric.data import Data, DataLoader
from torch.nn.functional import one_hot

################################################################################
# UTILITIES


def explain_graph_visualized(
    model: torch.nn.Module,
    data: Data,
    device: torch.device = None
):
    """Visualize the GNNExlainer explanation of a trained model for the input data.
    Returns an interactive widget with a threshold slider.

    Parameters
    ----------
    model : torch.nn.Module
        Trained GNN model
    data : Data
        Query datapoint
    device : torch.device, optional
        Torch device, by default None

    Returns
    -------
    interact
        Interactive widget
    """
    data.to(device)
    # x = data.x
    # edge_index = data.edge_index
    model.eval()

    # Initialize explainer
    explainer = GNNExplainer(model, epochs=200).to(device)
    # Train explainer
    model_args = (data.x, data.edge_index, data.batch)
    GNNExp_feat_mask, GNNExp_edge_mask = explainer.explain_graph(
        *model_args,
        model(*model_args).argmax(dim=1)
    )

    fig, ax = plt.subplots(
        1, 2,
        sharex=True,
        sharey=True,
        dpi=150, figsize=(10, 4)
    )

    fig.tight_layout()

    plot_mol(
        data,
        edge_mask=GNNExp_edge_mask,
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
            data,
            edge_mask=GNNExp_edge_mask,
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


def make_interactive_explainer(
    model: torch.nn.Module,
    dataset_te,
    device: torch.device = None
):
    """Create an interactive widget over a test dataset to query for single explanation.

    Parameters
    ----------
    model : torch.nn.Module
        Trained GNN Model
    dataset_te : pytorch_geometric.Dataset
        Test dataset
    device : torch.device, optional
        Torch device, by default None

    Returns
    -------
    interact
        Interactive widget
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model = model.to(device)

    test_batch = next(iter(
        DataLoader(dataset_te, batch_size=len(dataset_te))
    )).to(device)

    y_pred = model(test_batch.x, test_batch.edge_index, test_batch.batch).argmax(dim=1)
    y_true = test_batch.y

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
            dataset_te, batch_size=1, shuffle=False
        )

        return interact(
            lambda data: explain_graph_visualized(model, data, device),
            data=[graph for i, graph in enumerate(list(single_loader_te)) if i in idces],
        )

    return interact(
        big_interact,
        which=["true_pos", "true_neg", "false_pos", "false_neg"]
    )

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
