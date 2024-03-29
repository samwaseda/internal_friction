from pyiron_atomistics import Project as PyironProject
from sklearn import linear_model
import numpy as np
from itertools import chain
from collections import defaultdict
from hashlib import sha1


def set_input(job, cores=120, queue='cm'):
    job.set_encut(550)
    job.set_kpoints(k_mesh_spacing=0.1)
    job.set_convergence_precision(electronic_energy=1.0e-6)
    job.set_mixing_parameters(
        density_mixing_parameter=0.7,
        density_residual_scaling=0.1,
        spin_mixing_parameter=0.7,
        spin_residual_scaling=0.1
    )
    job.server.cores = cores
    if queue is not None:
        job.server.queue = queue


class Project(PyironProject):
    def __init__(
        self,
        path='',
        user=None,
        sql_query=None,
        default_working_directory=False,
        n_repeat=3,
        Mn_range=np.array([0.18, 0.24, 0.31]),
        a_0=3.5,
        eps_lst=np.linspace(-0.01, 0.04, 6)
    ):
        super().__init__(
            path=path,
            user=user,
            sql_query=sql_query,
            default_working_directory=default_working_directory,
        )
        self._n_repeat = n_repeat
        self._Mn_range = Mn_range
        self._a_0 = a_0
        self._eps_lst = eps_lst
        self._coeff_murn = None
        self._sqs_d_dict = None
        self._structure_dict = None

    @property
    def n_repeat(self):
        return self._n_repeat

    @property
    def Mn_range(self):
        return self._Mn_range

    def run_sqs(self):
        for c in self.c_range:
            job = self.create.job.SQSJob(('sqs', self.n_repeat, c))
            job.structure = self.create_structure('Fe', 'fcc', self._a_0).repeat(self.n_repeat)
            job.input['mole_fractions'] = {'Fe': 1 - c, 'Mn': c}
            job.run()

    @property
    def sqs_lst(self):
        structure_lst = [job.get_structure() for job in self.iter_jobs(job='sqs*')]
        if len(structure_lst) == 0:
            self.run_sqs()
            return self.sqs_lst
        return [self.create_structure('Fe', 'fcc', self._a_0)] + structure_lst

    def run_murn(self):
        for structure in self.sqs_lst:
            for epsilon in self._eps_lst:
                if epsilon == 0:
                    continue
                c = np.round(len(structure.select_index('Mn')) / len(structure), 2)
                job = self.create.job.Sphinx(('spx_murn', self.n_repeat, epsilon, c))
                if not job.status.initialized:
                    continue
                job.structure = structure.copy()
                job.structure.apply_strain(epsilon)
                set_input(job)
                m = 2 * np.ones(len(job.structure))
                m[job.structure.analyse.get_layers()[:, 0] % 2 == 0] *= -1
                job.structure.set_initial_magnetic_moments(m)
                job.calc_minimize()
                job.run()

    @staticmethod
    def _get_volume(job):
        return job['output/generic/volume'][-1] / len(job['input/structure/indices'])

    @staticmethod
    def _get_energy(job):
        return job['output/generic/energy_pot'][-1] / len(job['input/structure/indices'])

    @staticmethod
    def _get_concentration(job):
        indices = (np.array(job['input/structure/species'])[job['input/structure/indices']] == 'Mn')
        return np.sum(indices) / len(indices)

    @staticmethod
    def _db_filter_function(job):
        return (job.status in ['finished', 'not_converged']) & job.job_name.startswith('spx_murn')

    @property
    def murn_dataframe(self):
        table = self.create.table()
        table.filter_function = self._db_filter_function
        table.add['volume'] = self._get_volume
        table.add['energy'] = self._get_energy
        table.add['concentration'] = self._get_concentration
        table.run(delete_existing_job=True)
        return table.get_dataframe()

    @property
    def coeff_murn(self):
        if self._coeff_murn is None:
            df = self.murn_dataframe
            c = df.concentration
            E = df.energy
            a = (df.volume * 4)**(1 / 3)
            reg = linear_model.LinearRegression()

            def get_arguments(a, c, array=True):
                if array:
                    x = np.meshgrid(np.asarray([a]).reshape(-1, 1), np.asarray([c]).reshape(1, -1))
                    x = np.asarray(x).reshape(2, -1).T
                    c = x[:, 1]
                    a = x[:, 0]
                return np.vstack((c, c**2, a, a**2, c * a, c * a**2)).T
            x = get_arguments(a, c, array=False)
            reg.fit(x, E)
            self._coeff_murn = reg.coef_.copy()
        return self._coeff_murn

    def get_lattice_constant(self, c):
        return -(self.coeff_murn[2] + self.coeff_murn[4] * c) * 0.5 / (self.coeff_murn[3] + self.coeff_murn[5] * c)

    @property
    def _sqs_displacements(self):
        if self._sqs_d_dict is None:
            d_dict = defaultdict(list)
            for job in chain(*[
                self.iter_jobs(job='spx_murn*', chemicalformula='*Mn*', status=status)
                for status in ['finished', 'not_converged']
            ]):
                d_dict['formula'].append(job.structure.get_chemical_formula())
                d_dict['displacements'].append(
                    np.einsum(
                        'ji,nj->ni',
                        np.linalg.inv(job.structure.cell),
                        job.output.total_displacements[-1]
                    )
                )
            for k, v in d_dict.items():
                d_dict[k] = np.asarray(v)
            self._sqs_d_dict = {}
            for c in np.unique(d_dict['formula']):
                self._sqs_d_dict[c] = np.mean(
                    d_dict['displacements'][c == d_dict['formula']], axis=0
                )
        return self._sqs_d_dict

    @property
    def structure_lst(self):
        if self._structure_dict is None:
            self._structure_dict = {}
            for structure in self.sqs_lst:
                a_0 = self.get_lattice_constant(
                    np.sum(structure.get_chemical_symbols() == 'Mn') / len(structure)
                )
                strain = (a_0**3 / 4 / structure.get_volume(per_atom=True))**(1 / 3) - 1
                struct = structure.apply_strain(strain, return_box=True)
                cf = struct.get_chemical_formula()
                if cf in self._sqs_displacements.keys():
                    struct.positions += np.einsum('ji,nj->ni', struct.cell, self._sqs_displacements[cf])
                self._structure_dict[cf] = struct.copy()
        return self._structure_dict

    def run_nonmag(self):
        for k, structure in self.structure_lst.items():
            cf = structure.get_chemical_formula()
            if 'Mn' not in cf:
                continue
            dz = np.array([0, 0, 0.5 * (structure.get_volume(per_atom=True) * 4)**(1 / 3)])
            for ii, pos in enumerate(structure.positions + dz):
                job = self.create.job.Sphinx(('spx_carbon_nonmag', cf, ii))
                if not job.status.initialized:
                    continue
                job.structure = structure.copy()
                set_input(job)
                job.structure += self.create.structur.atoms(
                    elements=['C'], positions=[pos], cell=structure.cell
                )
                job.calc_minimize()
                job.run()

    def run_afm(self):
        n_cores = 80
        for job_nonmag in self.iter_jobs(job='spx_carbon_nonmag*'):
            if job_nonmag.status.running or job_nonmag.status.submitted:
                print(job_nonmag.job_name, 'not ready')
                continue
            structure = job_nonmag.get_structure()
            job_name = job_nonmag.job_name.replace('nonmag', 'afm')
            mixer = self.create_job('Intermixer', job_name.replace('spx', 'mixer'))
            if not mixer.status.initialized:
                continue
            total_cores = 0
            for ii in range(3):
                spx = self.create.job.Sphinx((job_name, ii))
                magmoms = 2 * np.ones(len(structure))
                spx.structure = structure.copy()
                magmoms[structure.analyse.get_layers(distance_threshold=0.5)[:, ii] % 2 == 0] *= -1
                magmoms[spx.structure.select_index('C')] = 0.01
                spx.structure.set_initial_magnetic_moments(magmoms)
                spx.server.run_mode.interactive_non_modal = True
                set_input(spx, cores=n_cores, queue=None)
                mixer.ref_job = spx
                total_cores += n_cores
            mixer.server.cores = total_cores
            minimizer = self.create.job.SxExtOptInteractive(job_name.replace('spx', 'sxextopt'))
            minimizer.ref_job = mixer
            minimizer.server.cores = total_cores
            minimizer.server.queue = 'cm'
            minimizer.run()

    def run_elast(self):
        for k, structure in self.structure_dict.items():
            job = self.create.job.Sphinx(('spx_elast', k))
            if not job.status.initialized:
                continue
            job.structure = structure.copy()
            set_input(job)
            m = 2 * np.ones(len(job.structure))
            m[job.structure.analyse.get_layers(distance_threshold=0.1)[:, 0] % 2 == 0] *= -1
            job.structure.set_initial_magnetic_moments(m)
            job.calc_minimize()
            elast = job.create_job('ElasticTensor', job.job_name.replace('spx_', ''))
            elast.input['use_elements'] = False
            elast.run()


def setup_lmp_input(lmp, n_atoms=None, direction=None, fix_id=-1):
    """
    Change input for LAMMPS to run a drag calculation.

    Args:
        lmp (pyiron_atomistics.lammps.lammps.Lammps): LAMMPS job
        n_atoms (int): number of free atoms (default: None, i.e. it is
            determined from the job structure)
        direction (None/numpy.ndarray): direction along which the force is
            cancelled. None if all forces are to be cancelled (default: None)
        fix_id (None/int): id of the atom to be fixed (default: -1, i.e. last
            atom)

    Returns:
        None (input of lmp is changed in-place)

    In the context of this function, a drag calculation is a constraint energy
    minimization, in which one atom is either not allowed to move at all, or
    not allowed to move along a given direction. In order for the system to not
    fall to the energy minimum, the sum of the remaining forces is set to 0.

    Exp: Hydrogen diffusion

    >>> from pyiron_atomistics import Project
    >>> pr = Project("DRAG")
    >>> bulk = pr.create.structure.bulk('Ni', cubic=True)
    >>> a_0 = bulk.cell[0, 0]
    >>> x_octa = np.array([0, 0, 0.5 * a_0])
    >>> x_tetra = np.array(3 * [0.25 * a_0])
    >>> dx = x_tetra - x_octa
    >>> transition = np.linspace(0, 1, 101)
    >>> x_lst = transition[:, None] * dx + x_octa
    >>> structure = bulk.repeat(4) + pr.create.structure.atoms(
    ...     positions=[x_octa],
    ...     elements=['H'],
    ...     cell=structure.cell
    ... )
    >>> lmp = pr.create.job.Lammps('lmp')
    >>> lmp.structure = structure
    >>> lmp.calc_minimize()
    >>> lmp.potential = potential_of_your_choice
    >>> setup_lmp_input(lmp, direction=dx)
    >>> lmp.interactive_open()
    >>> for xx in x_lst:
    >>>     lmp.structure.positions[-1] = xx
    >>>     lmp.run()
    >>> lmp.interactive_close()
    """
    if lmp.input.control["minimize"] is None:
        raise ValueError("set calc_minimize first")
    if n_atoms is None:
        try:
            n_atoms = len(lmp.structure) - 1
        except TypeError:
            raise AssertionError("either `n_atoms` or the structure must be set")
    fix_id = np.arange(n_atoms)[fix_id] + 2
    lmp.input.control['atom_modify'] = 'map array'
    lmp.input.control["group___fixed"] = f"id {fix_id}"
    lmp.input.control["group___free"] = "subtract all fixed"
    if direction is None:
        for ii, xx in enumerate(['x', 'y', 'z']):
            lmp.input.control[f'variable___f{xx}_free'] = f'equal f{xx}[{fix_id}]/{n_atoms}'
            lmp.input.control[f'variable___f{xx}_fixed'] = f'equal -f{xx}[{fix_id}]'
    else:
        direction = np.array(direction) / np.linalg.norm(direction)
        direction = np.outer(direction, direction)
        direction = np.around(direction, decimals=8)
        for grp, ss in zip(["free", "fixed"], [f"1/{n_atoms}*", "-"]):
            for ii, xx in enumerate(['x', 'y', 'z']):
                txt = "+".join([f"({ss}f{xxx}[{fix_id}]*({direction[ii][iii]}))" for iii, xxx in enumerate(['x', 'y', 'z'])])
                lmp.input.control[f'variable___f{xx}_{grp}'] = f" equal {txt}"
    lmp.input.control['variable___energy'] = "atom 0"
    for key in ["free", "fixed"]:
        txt = " ".join([f"v_f{x}_{key}" for x in ["x", "y", "z"]])
        lmp.input.control[f"fix___f_{key}"] = f"{key} addforce {txt} energy v_energy"
    lmp.input.control['min_style'] = 'quickmin'
