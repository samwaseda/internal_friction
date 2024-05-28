import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
from pint import UnitRegistry
from functools import cached_property, cache
from scipy.sparse import coo_matrix


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


class Measurement:
    def __init__(
        self,
        structure,
        P_all_dict,
        c_C,
        c_v,
        v_diff=1.0,
        C_diff=1.2,
        std_concentration=0.001,
    ):
        self._phi = None
        self._structure = structure
        P_C_diff_v, E_C_diff_v = get_diff_E_P(
            "Fe", P_all_dict["C_diff_v"], structure=structure
        )
        P_v_diff_C, E_v_diff_C = get_diff_E_P(
            "C", P_all_dict["v_diff_C"], structure=structure
        )

        self.E_diff_dict = {"v": E_v_diff_C, "C": E_C_diff_v}
        self.P_diff_dict = {"v": P_v_diff_C, "C": P_C_diff_v}
        self._dict_element = {"Fe": "Fe", "v": "Fe", "C": "C"}
        self.concentration = {"C": c_C, "v": c_v}
        self.diffusion_barrier = {"v": v_diff, "C": C_diff}
        ureg = UnitRegistry()
        self.kB = (1 * ureg.boltzmann_constant * ureg.kelvin).to("eV").magnitude
        self.std_concentration = std_concentration

    @cached_property
    def matrix(self):
        return self._structure[self._structure.get_chemical_symbols() != "O"]

    @property
    def phi(self):
        if self._phi is None:
            phi_C = 1 + self.std_concentration * np.random.randn(
                len(self.get_index("C"))
            )
            phi_C = (
                self.concentration["C"]
                / (1 - self.concentration["C"])
                * phi_C
                / phi_C.mean()
            )
            phi_v = 1 + self.std_concentration * np.random.randn(
                len(self.get_index("v"))
            )
            phi_v = (
                self.concentration["v"]
                / (1 - self.concentration["v"])
                * phi_v
                / phi_v.mean()
            )
            self._phi = np.zeros(len(self.matrix))
            self._phi[self.get_index("v")] = phi_v
            self._phi[self.get_index("C")] = phi_C
        return self._phi

    @cache
    def get_index(self, element):
        return self.matrix.select_index(self._dict_element[element])

    def get_phi(self, element):
        return self.phi[self.get_index(element)]

    def set_phi(self, new_phi, element=None):
        if element is not None:
            self._phi[self.get_index(element)] = new_phi
        else:
            self._phi = new_phi

    @cached_property
    def _mat(self):
        return get_E_P(structure=self.matrix)

    @property
    def E_mat(self):
        return self._mat[1]

    @property
    def P_mat(self):
        return self._mat[0]

    def get_diff(self, elem_1, elem_2):
        return (
            self.E_diff_dict[elem_1] @ self.phi[self.get_index(elem_2)]
            + self.diffusion_barrier[elem_1]
        )

    @property
    def narrow_path(self):
        return self._structure.positions[self._structure.select_index("O")]

    @cached_property
    def pairs(self):
        return {
            key: self.matrix[self.get_index(key)]
            .get_neighborhood(self.narrow_path, num_neighbors=2)
            .indices
            for key in ["C", "v"]
        }

    def get_E_sub(self, epsilon=None):
        E = {
            key: (self.E_mat @ self.phi)[self.get_index(key)][self.pairs[key]]
            for key in ["C", "v"]
        }
        if epsilon is not None:
            P = np.einsum("nmij,m->nij", self.P_mat, self.phi)
            P_sub = {key: P[self.get_index(key)] for key in ["C", "v"]}
            E = {
                key: value
                + np.einsum("nij,ij->n", P_sub[key], epsilon)[self.pairs[key]]
                for key, value in E.items()
            }
        return E

    def get_E_diff(self, epsilon=None):
        E_diff = {"C": self.get_diff("C", "v"), "v": self.get_diff("v", "C")}
        if epsilon is not None:
            P = self.P_diff_dict
            E_diff = {
                key: value
                + np.einsum(
                    "nmij,m,ij->n", P[key], self.phi[self.get_index(key)], epsilon
                )
                for key, value in E_diff.items()
            }
        E_sub = self.get_E_sub(epsilon=epsilon)
        return {key: E_diff[key][:, None] - E_sub[key] for key in ["C", "v"]}

    @cached_property
    def inv_volume(self):
        volume = self._structure.get_volume()
        ureg = UnitRegistry()
        return (1 / volume * ureg.electron_volt / ureg.angstrom**3).to(
            "Pa"
        ).magnitude / 1e9

    def get_pressure(self):
        return (
            0.5
            * np.einsum("n,nmij,m->ij", self.phi, self.P_mat, self.phi)
            * self.inv_volume
        )

    def get_K(self, temperature, epsilon=None, kappa=1.0e13):
        kBT = self.kB * temperature
        K = {}
        E = self.get_E_diff(epsilon=epsilon)
        for key, pair in self.pairs.items():
            pairs = np.concatenate((pair, pair[:, ::-1]), axis=0)
            K[key] = coo_matrix(
                (np.exp(np.log(kappa) - E[key].T.flatten() / kBT), tuple(pairs.T))
            )
        return K

    def get_chemical_potential(self, temperature, element=None, normalize=True):
        kBT = self.kB * temperature
        mu = self.E_mat @ self.phi + kBT * np.log(self.phi / (1 - self.phi))
        if normalize:
            for key in ["C", "v"]:
                mu[self.get_index(key)] -= get_mu_0(
                    mu[self.get_index(key)], self.concentration[key], kBT
                )
        if element is None:
            return mu
        else:
            return mu[self.get_index(element)]

    def get_internal_energy(self):
        return 0.5 * self.phi @ self.E_mat @ self.phi

    def get_entropy(self):
        return (
            -self.kB
            * (self.phi * np.log(self.phi) + (1 - self.phi) * (1 - self.phi)).sum()
        )

    def get_free_energy(self, temperature):
        return self.get_internal_energy() + temperature * self.get_entropy()
