import networkx as nx  # Graph manipulation library
from torch_geometric.utils import to_networkx  # Conversion function

import matplotlib.pyplot as plt
from torch import where

import pandas as pd
elements = pd.read_csv("/io/data/external/elementlist.csv", index_col=0)


def plot_mol(torch_graph, layout: str=None):
    fig, ax = plt.subplots(dpi=120)

    G = to_networkx(
        torch_graph,
        to_undirected=True
    )

    if layout in (None, "kamada_kawai"):
        pos = nx.kamada_kawai_layout(G)
    elif layout == "spring":
        pos = nx.spring_layout(G)

    atoms = torch_graph.x[:, 0]
    # single_bonds = where(torch_graph.edge_attr[:, 0] == 0)[0]
    # double_bonds = where(torch_graph.edge_attr[:, 0] == 1)[0]
    # triple_bonds = where(torch_graph.edge_attr[:, 0] == 2)[0]
    not_aromatic = where(torch_graph.edge_attr[:, 0] != 3)[0]
    aromatic = where(torch_graph.edge_attr[:, 0] == 3)[0]

    nx.draw_networkx(
        G, pos,
        node_color=atoms,  # color coded by atomic number
        cmap="prism",
        node_size=10 * atoms,    # size give by atomic number
        with_labels=False,
        edgelist=torch_graph.edge_index[:, not_aromatic].T.tolist(),
        width=torch_graph.edge_attr[:, 0] + 1,
    )

    # AROMATIC bonds -> DASHED
    nx.draw_networkx_edges(
        G, pos,
        edgelist=torch_graph.edge_index[:, aromatic].T.tolist(),
        style="dashed",
        width=1.5
    )

    # Add element symbols
    offset = 0.05
    nx.draw_networkx_labels(
        G,
        pos={
            node: (p[0] - offset, p[1] + 1.5 * offset)
            for node, p in pos.items()
        },
        labels={
            i: elements.iloc[torch_graph.x[i, 0].item()].symbol
            for i in range(torch_graph.num_nodes)
        },
    )

    fig.tight_layout()
    plt.show()
