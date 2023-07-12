import numpy as np
import hashlib
from pyiron_contrib.atomistics.atomistics.master.qha import QuasiHarmonicApproximation
from scipy.optimize import minimize

class Hessian:
    def __init__(
        self,
        pr,
        structure, 
        potential,
        symprec=0.001,
        minimize_pressure=True
    ):
        self.project = pr
        self.potential = potential
        self.structure = structure.copy()
        self.symprec = symprec
        self.minimize_pressure = minimize_pressure

    def get_lmp_minim(self, structure):
        lmp = self.project.create.job.Lammps(
            (
                'lmp',
                hashlib.sha1(
                    (structure.__repr__() + self.potential).encode()
                ).hexdigest()
            )
        )
        if lmp.status.initialized:
            lmp.structure = structure.copy()
            lmp.potential = self.potential
            lmp.interactive_open()
            qn = lmp.create_job('QuasiNewton', lmp.job_name.replace('lmp', 'qn'))
            qn.input['ionic_force_tolerance'] = 1.0e-4
            qn.run()
        return lmp

    def get_energy(self, strain):
        lmp = self.get_lmp_minim(self.structure.apply_strain(strain / 100, return_box=True))
        return lmp.output.energy_pot[-1]

    def get_stress(self, strain):
        lmp = self.get_lmp_minim(self.structure.apply_strain(strain / 100, return_box=True))
        return -lmp.output.pressures[-1].flatten()[[0, 4, 8]]

    def get_relaxed_structure(self):
        if not self.minimize_pressure:
            return self.get_lmp_minim(self.structure).get_structure()
        minim = minimize(
            self.get_energy, [0, 0, 0], method='Newton-CG', jac=self.get_stress
        )
        if not minim.success:
            raise AssertionError("Box minimization did not converge")
        return self.get_lmp_minim(
            self.structure.apply_strain(minim.x / 100, return_box=True)
        ).get_structure()

    @property
    def force_constants(self):
        structure = self.get_relaxed_structure()
        lmp = self.project.create.job.Lammps('lmp')
        lmp.structure = structure
        lmp.potential = self.potential
        lmp.interactive_open()
        qn = lmp.create_job(
            QuasiHarmonicApproximation, 
            'h_' + hashlib.sha1(
                (structure.__repr__() + self.potential).encode()
            ).hexdigest()
        )
        qn.input['num_points'] = 1
        qn.input['symprec'] = self.symprec
        qn.input['displacement'] = 0.001
        qn.run()
        return qn["output/force_constants"].squeeze()
