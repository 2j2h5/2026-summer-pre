import torch.nn.functional as F


def make_relation_graph(E, temperature=0.35, symmetrize=True, remove_self_loop=False, eps=1e-12):
    E_norm = F.normalize(E, p=2, dim=1)
    C = E_norm @ E_norm.T

    if remove_self_loop:
        C = C.clone()
        C.fill_diagonal_(-1e9)

    G = F.softmax(C / temperature, dim=1)

    if symmetrize:
        G = 0.5 * (G + G.T)

    G = G / (G.max() + eps)
    return G, C


def build_graphs_with_selected_T(embeddings, selected_T):
    graphs = {}
    for model_name, E in embeddings.items():
        T = selected_T[model_name]
        G, C = make_relation_graph(E, temperature=T, symmetrize=True)
        graphs[model_name] = {
            "E": E,
            "C": C,
            "G": G,
            "relation_temperature": T,
        }
        print("\nGraph built:", model_name)
        print("T:", T, "G:", tuple(G.shape), "G min/max:", float(G.min()), float(G.max()))
    return graphs
