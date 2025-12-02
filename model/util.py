import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def coords_to_residue_scores(S, n_coords=3):
    D = S.shape[0]
    assert D % n_coords == 0, "D must be divisible by n_coords"
    R = D // n_coords

    S_reshaped = S.reshape(R, n_coords, R, n_coords)
    S_res = S_reshaped.mean(axis=(1, 3))
    return S_res


def build_residue_adjacency(S_res, percentile=95):
    tau = np.percentile(S_res, percentile)
    G_res = (S_res > tau).astype(int)
    np.fill_diagonal(G_res, 0)
    return G_res, tau


def adjacency_to_digraph(G, node_names=None):
    G = np.asarray(G)
    N = G.shape[0]
    DG = nx.DiGraph()

    if node_names is None:
        DG.add_nodes_from(range(N))
    else:
        assert len(node_names) == N
        for idx, name in enumerate(node_names):
            DG.add_node(idx, label=name)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if G[i, j] != 0:
                DG.add_edge(j, i)

    return DG


def plot_digraph(DG, node_names=None, title=None, figsize=(8, 8)):
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(DG, k=0.3, iterations=200)

    if node_names is None:
        labels = {n: n for n in DG.nodes()}
    else:
        labels = {i: node_names[i] for i in DG.nodes()}

    nx.draw_networkx_nodes(DG, pos, node_size=300)
    nx.draw_networkx_edges(DG, pos, arrows=True, arrowstyle="->", arrowsize=12)
    nx.draw_networkx_labels(DG, pos, labels, font_size=8)

    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def normalize_scores(S):
    S = S.copy()
    max_val = S.max()
    if max_val <= 0:
        return S
    return S / max_val


def combine_position_angle_scores(
    S_pos,
    S_ang,
    alpha=0.5,
    beta=0.5,
    percentile=95,
):
    S_pos_norm = normalize_scores(S_pos)
    S_ang_norm = normalize_scores(S_ang)

    S_res_comb = alpha * S_pos_norm + beta * S_ang_norm

    tau_comb = np.percentile(S_res_comb, percentile)
    G_res_comb = (S_res_comb > tau_comb).astype(int)
    np.fill_diagonal(G_res_comb, 0)

    return G_res_comb, tau_comb, S_res_comb


def print_sccs(sccs, amino_acids):
    print("Strongly connected components (SCCs):")
    for cid, comp in enumerate(sccs):
        members_idx = sorted(list(comp))
        members_names = [amino_acids[i] for i in members_idx]
        print(f"  SCC {cid}: " + ", ".join(members_names))
