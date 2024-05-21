from pyiron_atomistics import Project as PyironProject
from functools import cache
import numpy as np
from scipy.spatial import cKDTree
from tqdm.auto import tqdm


class Project(PyironProject):

    a_0 = 3.4812
    n_repeat = 3

    def load_(self, job_name):
        job = self.inspect(job_name)
        job.structure = job["input/structure"].to_object()
        return job

    def get_job(self, job_name, latest=False):
        job_names = list(self.job_table(job=job_name.split("_run")[0] + "*").job)
        if latest:
            job_name = job_names[-1]
        else:
            job_name = job_names[0]
        return self.load_(job_name)

    @cache
    def get_energy(self, element, latest=True):
        if element == "Fe":
            job = self.load_("Fe")
            return job["output/generic/energy_pot"][0] / len(job.structure)
        elif element == "C":
            job = self.get_job("single_C", latest=latest)
            return (
                job["output/generic/energy_pot"][0]
                - len(job.structure.select_index("Fe")) * self.get_energy("Fe", latest=latest)
            )
        elif element == "v":
            job = self.get_job("single_v", latest=latest)
            return (
                job["output/generic/energy_pot"][0]
                - len(job.structure.select_index("Fe")) * self.get_energy("Fe", latest=latest)
            )
        elif element == "C_diff":
            job = self.get_job("C_diff", latest=latest)
            return (
                job["output/generic/energy_pot"][-1]
                - self.get_energy("Fe", latest=latest) * len(job.structure.select_index("Fe"))
                - self.get_energy("C", latest=latest)
            )
        elif element == "v_diff":
            job = self.get_job("v_diff", latest=latest)
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

    def get_structure(self, x_C=None):
        structure = self.get_iron().repeat(self.n_repeat)
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
    def get_kanzaki(self, element):
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
        neigh = job.structure.get_neighborhood(
            x_origin,
            num_neighbors=None,
            cutoff_radius=0.499 * job.structure.cell[0, 0]
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
                self.get_job(job.job_name, latest=True), elements, latest=True
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

    def get_data_list(self, force_parsing=False):
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
