from pyiron_atomistics import Project
from sklearn import linear_model
import numpy as np


def set_input(job):
    job.set_encut(550)
    job.set_kpoints(k_mesh_spacing=0.1)
    job.set_convergence_precision(electronic_energy=1.0e-6)
    job.set_mixing_parameters(
        density_mixing_parameter=0.7,
        density_residual_scaling=0.1,
        spin_mixing_parameter=0.7,
        spin_residual_scaling=0.1
    )


class InternalFriction(Project):
    def __init__(
        self,
        project,
        job_name,
        n_repeat=3,
        Mn_range=np.array([0.18, 0.24, 0.31]),
        a_0=3.5,
        eps_lst=np.linspace(-0.01, 0.04, 6)
    ):
        super().__init__(project, job_name)
        self._n_repeat = n_repeat
        self._Mn_range = Mn_range
        self._a_0 = a_0
        self._eps_lst = eps_lst
        self._coeff_murn = None

    @property
    def n_repeat(self):
        return self._n_repeat

    @property
    def Mn_range(self):
        return self._Mn_range

    def run_sqs(self):
        for c in self.c_range:
            job = self.create_job('SQSJob', 'sqs_{}_{}'.format(self.n_repeat, c).replace('.', 'c'))
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
                job.server.cores = 120
                job.server.queue = 'cm'
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
