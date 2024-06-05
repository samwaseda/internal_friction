from pyiron_atomistics import Project as PyironProject
from functools import cache
import numpy as np
from scipy.spatial import cKDTree
from tqdm.auto import tqdm
from collections import defaultdict
from sklearn.decomposition import PCA


class Project(PyironProject):

    a_0 = 3.4812
    n_repeat = 3

    def load_(self, job_name):
        job = self.inspect(job_name)
        job.structure = job["input/structure"].to_object()
        return job

    def _get_job(self, job_name, latest=False):
        job_names = list(self.job_table(job=job_name.split("_run")[0] + "*").job)
        if latest:
            return self.load_(job_names[-1])
        else:
            return self.load_(job_names[0])

    @cache
    def get_energy(self, element, latest=True):
        if element == "Fe":
            job = self.load_("Fe")
            return job["output/generic/energy_pot"][0] / len(job.structure)
        elif element == "C":
            job = self._get_job("single_C", latest=latest)
            return (
                job["output/generic/energy_pot"][0]
                - len(job.structure.select_index("Fe")) * self.get_energy("Fe", latest=latest)
            )
        elif element == "v":
            job = self._get_job("single_v", latest=latest)
            return (
                job["output/generic/energy_pot"][0]
                - len(job.structure.select_index("Fe")) * self.get_energy("Fe", latest=latest)
            )
        elif element == "C_diff":
            job = self._get_job("C_diff", latest=latest)
            return (
                job["output/generic/energy_pot"][-1]
                - self.get_energy("Fe", latest=latest) * len(job.structure.select_index("Fe"))
                - self.get_energy("C", latest=latest)
            )
        elif element == "v_diff":
            job = self._get_job("v_diff", latest=latest)
            return (
                job["output/generic/energy_pot"][-1]
                - self.get_energy("Fe", latest=latest) * len(job.structure.select_index("Fe"))
            )

    def get_iron(self):
        return self.create.structure.bulk(
            'Fe', cubic=True, crystalstructure='fcc', a=self.a_0
        )

    def detect_vacancy(self, structure):
        struct_ref = self.get_structure()
        return struct_ref.positions[
            structure.get_neighborhood(
                struct_ref.positions, num_neighbors=1
            ).distances.flatten() > 1
        ].squeeze()

    def get_structure(self, x_C=None, n_repeat=None):
        if n_repeat is None:
            n_repeat = self.n_repeat
        structure = self.get_iron().repeat(n_repeat)
        if x_C is None:
            self.set_initial_magnetic_moments(structure)
            return structure
        return self.append_carbon(x_C, structure)

    def set_initial_magnetic_moments(self, structure, magnitude=3):
        m = magnitude * np.ones(len(structure))
        m[structure.select_index("C")] = 0.01
        structure.set_initial_magnetic_moments(m)

    def append_carbon(self, x_C, structure):
        x_C = np.reshape(x_C, (-1, 3))
        structure = structure + self.create.structure.atoms(
            positions=x_C, elements=len(x_C) * ["C"], cell=structure.cell
        )
        self.set_initial_magnetic_moments(structure)
        return structure

    @cache
    def get_kanzaki(self, element, cutoff=None):
        if element == "C":
            job = self.load_("single_C_run_0")
            x_origin = job.structure.positions[job.structure.select_index("C")[-1]]
        elif element == "v":
            job = self.load_("single_v_run_0")
            x_origin = self.detect_vacancy(job.structure)
        elif element == "C_diff":
            job = self.load_("C_diff_run_0")
            x_origin = job.structure.positions[job.structure.select_index("C")[-1]]
        elif element == "v_diff":
            job = self.load_("v_diff_run_0")
            x_origin = get_missing(self.get_structure(), get_diff(self.get_structure())[0])
        else:
            raise ValueError
        if cutoff is None:
            cutoff = 0.499 * job.structure.cell[0, 0]
        neigh = job.structure.get_neighborhood(
            x_origin,
            num_neighbors=None,
            cutoff_radius=cutoff
        )
        forces = job["output/generic/forces"][0]
        return neigh.vecs[neigh.distances > 0.1], forces[neigh.indices[neigh.distances > 0.1]]

    def project_kanzaki_forces(self, position, element, structure):
        forces = np.zeros_like(structure.positions)
        neigh = structure.get_neighborhood(position, num_neighbors=100)
        x, f = self.get_kanzaki(element)
        tree = cKDTree(x)
        distances, indices = tree.query(neigh.vecs)
        ind, dd, ii = neigh.indices, distances, indices
        forces[ind[dd < 0.1]] = f[ii[dd < 0.1]]
        return forces

    def _get_element_info(self, job):
        ref_structure = self.get_structure()
        octa_reference = self.get_octa(ref_structure, get_structure=True)
        struct_Fe = job.structure[job.structure.select_index("Fe")]
        struct_C = job.structure[job.structure.select_index("C")]
        elements = []
        positions = []
        C_diff = get_missing(octa_reference, struct_C, element="C")
        if len(C_diff) == 0:
            C_octa = struct_C.positions.squeeze()
            if len(C_octa) > 0:
                positions.extend(np.atleast_2d(C_octa))
                elements.extend(len(np.atleast_2d(C_octa)) * ["C"])
        else:
            positions = [C_diff]
            elements.append("C_diff")
            C_octa = struct_C.positions[np.linalg.norm(struct_C.positions - C_diff, axis=-1) > 0.01]
            if len(C_octa) > 0:
                positions.extend(np.atleast_2d(C_octa))
                elements.extend(len(np.atleast_2d(C_octa)) * ["C"])
        v_diff = get_missing(ref_structure, struct_Fe, element="Fe")
        if len(v_diff) > 0:
            elements.append("v_diff")
            positions.append(v_diff)
            v_sub = get_missing(struct_Fe, ref_structure, element="Fe")[2:].squeeze()
        else:
            v_sub = get_missing(struct_Fe, ref_structure, element="Fe")
        if len(v_sub) > 0:
            positions.extend(np.atleast_2d(v_sub))
            elements.extend(len(np.atleast_2d(v_sub)) * ["v"])
        return elements, positions

    def _get_binding_energy(self, job, elements, latest=False):
        E = job["output/generic/energy_pot"][-1]
        for element in ["Fe", "C"]:
            E -= len(job.structure.select_index(element)) * self.get_energy(element, latest=latest)
        for element in elements:
            if element == "C":
                continue
            E -= self.get_energy(element, latest=latest)
        return E

    def get_data(self, job):
        elements, positions = self._get_element_info(job)
        if np.linalg.norm(job.structure.find_mic(np.diff(positions, axis=0))) < 0.1:
            raise ValueError(f"Distance too short {job.job_name}")
        data = {
            "positions": np.array(positions),
            "elements": elements,
            "energy": self._get_binding_energy(job, elements),
            "min_energy": self._get_binding_energy(
                self._get_job(job.job_name, latest=True), elements, latest=True
            ),
            "dipole": [],
        }
        for ii, (cc, xx) in enumerate(zip(data["elements"], data["positions"])):
            force = job["output/generic/forces"][0].copy()
            if "diff" not in data["elements"][(ii + 1) % 2]:
                force -= self.project_kanzaki_forces(
                    data["positions"][(ii + 1) % 2], data["elements"][(ii + 1) % 2], job.structure
                )
            neigh = job.structure.get_neighborhood(
                xx, num_neighbors=None, cutoff_radius=job.structure.cell[0, 0] * 0.499
            )
            data["dipole"].append(np.einsum("ni,nj->ij", neigh.vecs, force[neigh.indices]))
        data["dipole"] = np.array(data["dipole"])
        data["vec"] = job.structure.find_mic(np.diff(data["positions"], axis=0)).squeeze()
        return data

    def get_octa(self, structure, get_structure=False):
        voro = structure.analyse.get_voronoi_vertices()
        neigh = structure.get_neighborhood(voro, num_neighbors=1)
        positions = voro[neigh.distances.flatten() > neigh.distances.flatten().mean()]
        if get_structure:
            return self.create.structure.atoms(
                elements=len(positions) * ["C"], positions=positions, cell=structure.cell
            )
        return positions

    @cached_property
    def _data_list(self):
        job_lst = [
            job for job in self.job_table(job="*_0").job
            if job.startswith("Cv") or job.startswith("vC") or job.startswith("CC") or job.startswith("vv")
        ]
        results = []
        for job_name in tqdm(job_lst):
            if "sub" in job_name:
                continue
            data = self.get_data(self.load_(job_name))
            results.append({k: v for k, v in data.items()})
        return results

    def get_P_solute(self):
        P_solute = {
            tag: np.einsum(
                "ni,nj->ij", *self.get_kanzaki(tag)
            ).diagonal().mean() * np.eye(3) for tag in ["C", "v"]
        }
        for tag in ["C_diff", "v_diff"]:
            P_solute[tag] = np.einsum("ni,nj->ij", *self.get_kanzaki(tag))
        return P_solute

    def get_data_dict(self, cutoff_radius=None):
        if cutoff_radius is None:
            cutoff_radius = 0.5 * self.get_structure().cell[0, 0] - 0.001
        P_dict = defaultdict(list)
        v_dict = defaultdict(list)
        E_dict = defaultdict(list)
        E_min_dict = defaultdict(list)
        P_solute = self.get_P_solute()
        for data in self._data_list():
            # if any(["diff" in ee for ee in data["elements"]]):
            #     continue
            if np.linalg.norm(data["vec"]) > cutoff_radius:
                continue
            elem = data["elements"]
            if "diff" in data["elements"][-1]:
                elem = data["elements"][::-1]
            tag = "_".join(elem)
            P_dict[tag].append(data["dipole"][0] - P_solute[elem[-1]])
            v_dict[tag].append(data["vec"].copy())
            E_dict[tag].append(data["energy"])
            E_min_dict[tag].append(data["min_energy"])
        return {
            k: {"dipole": v, "vecs": v_dict[k], "energy": E_dict[k], "min_energy": E_min_dict[k]}
            for k, v in P_dict.items()
        }

    def get_P_all_dict(self, cutoff_radius=None):
        octa = self.get_octa(self.get_structure())
        structure = self.get_structure(octa[0])
        data_dict = self.get_data_dict(cutoff_radius=cutoff_radius)
        P_all_dict = {"C_C": get_P_all(structure, octa - octa[0], octa, **data_dict["C_C"])}

        structure = self.get_structure()
        del structure[structure.get_neighborhood([0, 0, 0], num_neighbors=1).indices[0]]
        P_all_dict["C_v"] = get_P_all(structure, -octa, octa, **data_dict["C_v"])

        struct_v_diff, index = get_diff(self.get_structure())
        P_all_dict["v_diff_C"] = get_P_all(
            struct_v_diff, octa - struct_v_diff.positions[index], octa, **data_dict["v_diff_C"]
        )
        struct_C_diff = self.get_structure(
            0.5 * (octa[0] + octa[1:][np.linalg.norm(octa[1:] - octa[0], axis=-1).argmin()])
        )
        P_all_dict["C_diff_v"] = get_P_all(
            struct_C_diff,
            struct_C_diff.positions[:-1] - struct_C_diff.positions[-1],
            struct_C_diff.positions[:-1],
            **data_dict["C_diff_v"],
        )
        return self.append_canonical(P_all_dict)

    def append_canonical(self, P_all_dict):
        job = self.load_("C_diff_run_0")
        pca = PCA()
        neigh = job.structure.get_neighbors(num_neighbors=6)
        pca.fit(neigh.vecs[job.structure.select_index("C")].squeeze())
        P_all_dict["C_diff_v"]["vecs_canonical"] = np.einsum(
            "ij,nj->ni", pca.components_, P_all_dict["C_diff_v"]["vecs"]
        )
        P_all_dict["C_diff_v"]["dipole_canonical"] = np.einsum(
            "ik,jl,nkl->nij", pca.components_, pca.components_, P_all_dict["C_diff_v"]["dipole"]
        )

        job = self.load_("v_diff_run_0")
        neigh = job.structure.get_neighbors(num_neighbors=4)
        pca.fit(neigh.vecs[np.argmin(neigh.distances[:, -1])].squeeze())
        P_all_dict["v_diff_C"]["vecs_canonical"] = np.einsum(
            "ij,nj->ni", pca.components_, P_all_dict["v_diff_C"]["vecs"]
        )
        P_all_dict["v_diff_C"]["dipole_canonical"] = np.einsum(
            "ik,jl,nkl->nij", pca.components_, pca.components_, P_all_dict["v_diff_C"]["dipole"]
        )
        return P_all_dict

    def get_large_structure(self, n_repeat=10):
        structure = self.get_structure(n_repeat=n_repeat)
        voro = structure.analyse.get_voronoi_vertices()
        dist = structure.get_neighborhood(voro, num_neighbors=1).distances.squeeze()
        octa = voro[dist > np.mean(dist)]
        neigh = structure.get_neighbors(num_neighbors=12)
        narrow_paths = structure.positions[:, None, :] + neigh.vecs * 0.5
        narrow_paths = narrow_paths[neigh.indices > neigh.atom_numbers]
        octa_structure = self.create.structure.atoms(elements=len(octa) * ["C"], positions=octa, cell=structure.cell)
        diff_structure = self.create.structure.atoms(
            elements=len(narrow_paths) * ["O"], positions=narrow_paths, cell=structure.cell
        )
        return structure + octa_structure + diff_structure


def get_P_all(structure, v_all, points, vecs, dipole, energy, min_energy):
    P_all = np.zeros((len(points), 3, 3))
    E_all = np.zeros(len(points))
    E_min_all = np.zeros(len(points))
    indices, rot_all = get_equivalent_indices_and_rot(structure, points)
    v_all = structure.find_mic(v_all)
    for vv, PP, EE, min_E in zip(vecs, dipole, energy, min_energy):
        distances = structure.get_distances_array(v_all, vv)
        if distances.min() > 0.1:
            continue
        current_index = distances.argmin()
        cond = indices[current_index] == indices
        rot = np.einsum("ij,njk->nik", rot_all[current_index].T, rot_all[cond])
        P_all[cond] = np.einsum("nik,njl,kl->nij", rot, rot, PP)
        E_all[cond] = EE
        E_min_all[cond] = min_E
    return {"dipole": P_all, "energy": E_all, "vecs": v_all, "min_energy": E_min_all}


def get_equivalent_indices_and_rot(structure, points):
    sym = structure.get_symmetry()
    all_points = sym.generate_equivalent_points(points=points, return_unique=False)
    _, inverse = np.unique(
        np.round(all_points.reshape(-1, 3), decimals=4),
        axis=0,
        return_inverse=True,
    )
    inverse = inverse.reshape(all_points.shape[:-1])
    indices = np.min(inverse, axis=0)
    indices = np.unique(indices, return_inverse=True)[1]
    rot = sym.rotations[np.argmin(inverse, axis=0)]
    return indices, rot


def get_missing(structure, ref_structure, element=None, min_dist=0.1):
    if element is not None:
        structure = structure[structure.select_index(element)]
        ref_structure = ref_structure[ref_structure.select_index(element)]
    if len(structure) == 0 or len(ref_structure) == 0:
        return np.array([])
    neigh = structure.get_neighborhood(ref_structure.positions, num_neighbors=1)
    return ref_structure.positions[neigh.distances.flatten() > min_dist].squeeze()


def get_diff(structure):
    x_v = structure.positions[0]
    del structure[0]
    neigh = structure.get_neighborhood(x_v, num_neighbors=1)
    structure.positions[neigh.indices[0]] -= 0.5 * neigh.vecs[0]
    return structure, neigh.indices[0]
