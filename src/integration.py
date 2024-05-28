import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree


def get_diff_E_P(mode, P_dict, structure, cutoff_radius=np.inf):
    narrow_path = structure.positions[structure.select_index("O")]
    structure = structure.copy()[structure.get_chemical_symbols() != "O"]
    neigh = structure[structure.select_index("Fe")].get_neighborhood(
        narrow_path, num_neighbors=6
    )
    pca = PCA()
    rot = np.zeros((len(narrow_path), 3, 3))
    for ii, v in enumerate(neigh.vecs):
        if mode == "Fe":
            pca.fit(v)
        else:
            pca.fit(v[2:])
        rot[ii] = pca.components_

    vecs = structure.get_distances_array(
        narrow_path, structure[structure.select_index(mode)].positions, vectors=True
    )
    v_canonical = np.einsum("nij,nmj->nmi", rot, vecs)
    tree = cKDTree(P_dict["vecs_canonical"])
    dist, indices = tree.query(v_canonical)
    cond = np.isclose(dist, 0) * (np.linalg.norm(vecs, axis=-1) < cutoff_radius)
    P = np.einsum(
        "nki,nlj,nm,nmkl->nmij", rot, rot, cond, P_dict["dipole_canonical"][indices]
    )
    E = cond * P_dict["energy"][indices]
    return P, E


def get_E_P(structure, P_all_dict):
    structure = structure.copy()[structure.get_chemical_symbols() != "O"]

    P_mat = np.zeros((len(structure), len(structure), 3, 3))
    E_mat = np.zeros((len(structure), len(structure)))

    d = structure.get_distances_array(vectors=True)
    for elem1 in ["C", "v"]:
        for elem2 in ["C", "v"]:
            key = "_".join(sorted([elem1, elem2]))
            if elem1 == "v":
                elem1 = "Fe"
            if elem2 == "v":
                elem2 = "Fe"
            if key not in P_all_dict:
                continue
            tree = cKDTree(
                P_all_dict[key]["vecs"] * (1 - 2 * ([elem1, elem2] == ["C", "Fe"]))
            )
            dist, indices = tree.query(d[structure.select_index(elem1)])
            P_mat[structure.select_index(elem1)] += np.einsum(
                "j,ijkl,ij->ijkl",
                structure.get_chemical_symbols() == elem2,
                P_all_dict[key]["dipole"][indices],
                np.isclose(dist, 0),
            )
            E_mat[structure.select_index(elem1)] += np.einsum(
                "j,ij,ij->ij",
                structure.get_chemical_symbols() == elem2,
                P_all_dict[key]["energy"][indices],
                np.isclose(dist, 0),
            )
    return P_mat, E_mat


def get_fermi(mu, kBT, boltzmann=False):
    if boltzmann:
        return np.exp(-mu / kBT)
    return 1 / (1 + np.exp(mu / kBT))


def get_mu_0(mu, c_target, kBT, N=None, max_steps=30, dmu=-0.1):
    mu_0 = mu.mean()
    if N is None:
        N = np.ones_like(mu_0)
    n_sum = np.sum(N)
    if kBT == 0:
        return mu_0
    for _ in range(max_steps):
        mu_0 += dmu
        dc = c_target - np.sum(get_fermi(mu - mu_0, kBT) * N) / n_sum
        if dc * dmu < 0:
            dmu *= -0.5
    return mu_0
