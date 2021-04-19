from collections import defaultdict

import matplotlib.pyplot as plt

import networkx as nx  # Graph manipulation library

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx  # Conversion function

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
                (u, v) for u, v in G.edges() if edge_mask[(u, v)] > threshold
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
        counts[(u, v)] += 1

    for edge, count in counts.items():
        edge_mask_dict[edge] /= count

    return edge_mask_dict


def plot_expl(data, GNNExp_edge_mask, **kwargs):
    mol = to_molecule(data)

    GNNExp_edge_mask_dict = mask_to_dict(
        GNNExp_edge_mask,
        data
    )
    plot_mol(mol, edge_mask=GNNExp_edge_mask_dict, **kwargs)
