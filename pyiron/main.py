from pyiron_atomistics import Project as PyironProject
from sklearn import linear_model
import numpy as np
from itertools import chain
from collections import defaultdict


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
        return -(self.coeff[2] + self.coeff[4] * c) * 0.5 / (self.coeff[3] + self.coeff[5] * c)

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
        structure_dict = {}
        for structure in self.sqs_lst:
            a_0 = self.get_lattice_constant(
                np.sum(structure.get_chemical_symbols() == 'Mn') / len(structure)
            )
            strain = (a_0**3 / 4 / structure.get_volume(per_atom=True))**(1 / 3) - 1
            struct = structure.apply_strain(strain, return_box=True)
            cf = struct.get_chemical_formula()
            if cf in self._sqs_displacements.keys():
                struct.positions += np.einsum('ji,nj->ni', struct.cell, self._sqs_displacements[cf])
            structure_dict[cf] = struct.copy()
        return structure_dict

    def run_nonmag(self):
        for k, structure in self.structure_dict.items():
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
