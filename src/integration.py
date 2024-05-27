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
