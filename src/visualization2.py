import networkx as nx  # Graph manipulation library
from torch_geometric.utils import to_networkx  # Conversion function

import matplotlib.pyplot as plt
from torch import where

import pandas as pd

# Periodic table thanks to Chris Andrejewski <christopher.andrejewski@gmail.com>
link = "https://raw.githubusercontent.com/andrejewski/periodic-table/master/data.csv"
elements = pd.read_csv(link, index_col=0, sep=", ", engine="python")


def ogb_graph_to_mol(torch_graph):
    atoms = torch_graph.x[:, 0]
    symbols = {
        i: symbol for i, symbol in enumerate(elements.loc[atoms + 1, "symbol"])
    }

    G = to_networkx(
        torch_graph,
        to_undirected=True
    )

    return G, symbols


def plot_torch_as_mol(
    torch_graph,
    layout: str = None,
    edge_mask=None
):
    G, node_labels = ogb_graph_to_mol(torch_graph)

    fig, ax = plt.subplots(dpi=120)

    pos = nx.planar_layout(G)
    if layout in (None, "kamada_kawai"):
        pos = nx.kamada_kawai_layout(G, pos=pos)
    if layout == "spring":
        pos = nx.spring_layout(G, pos=pos)

    if edge_mask is None:
        edge_color = 'black'
    else:
        edge_color = [edge_mask[(u, v)] for u, v in G.edges()]

    not_aromatic = where(torch_graph.edge_type[:, 0] != 3)[0]
    aromatic = where(torch_graph.edge_type[:, 0] == 3)[0]

    nx.draw_networkx(
        G, pos,
        node_color="azure",
        labels=node_labels,
        edgelist=torch_graph.edge_index[:, not_aromatic].T.tolist(),
        width=torch_graph.edge_attr[:, 0] + 1,
        edge_color=edge_color, edge_cmap=plt.cm.Blues,
    )

    # AROMATIC bonds -> DASHED
    nx.draw_networkx_edges(
        G, pos,
        edgelist=torch_graph.edge_index[:, aromatic].T.tolist(),
        style="dashed",
        width=1.5,
        edge_color=edge_color, edge_cmap=plt.cm.Blues,
    )

    fig.tight_layout()
    plt.show()


def draw_molecule(
    G: nx.Graph,
    edge_mask=None,
    edge_type=None,
    draw_edge_labels=False
):
    fig, ax = plt.subplots(dpi=120)

    node_labels = {
        u: data['name'] for u, data in G.nodes(data=True)
    }

    pos = nx.planar_layout(G)
    pos = nx.kamada_kawai_layout(G, pos=pos)

    if edge_type is None:
        widths = None
    else:
        widths = edge_type

    if edge_mask is None:
        edge_color = 'black'
    else:
        edge_color = [edge_mask[(u, v)] for u, v in G.edges()]

    nx.draw_networkx(
        G, pos=pos,
        labels=node_labels,
        width=widths,
        edge_color=edge_color, edge_cmap=plt.cm.Blues,
        node_color='azure'
    )

    if draw_edge_labels and edge_mask is not None:
        edge_labels = {k: ('%.2f' % v) for k, v in edge_mask.items()}
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels,
            font_color='red'
        )

    fig.tight_layout()
    plt.show()
