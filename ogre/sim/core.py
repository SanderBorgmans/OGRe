#!/usr/bin/env python

import os,h5py,numpy as np
from molmod.units import *
from pathlib import Path
from types import SimpleNamespace
from molmod.log import TimerGroup, ScreenLog

from yaff import *

from ogre.sim.utils_analytic import SimpleHDF5Writer, AbstractPotential, SimpleVerletIntegrator, SimpleCSVRThermostat

__all__ = ['OGRe_Simulation']


def Log():
    timer = TimerGroup()
    log = ScreenLog('OGRe', 1.0, '', '', timer)
    log.set_level(0)
    return log,timer

def load_potential_file(file_loc):
    import importlib.util
    spec = importlib.util.spec_from_file_location("module",file_loc)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class OGRe_Simulation(object):
    def __init__(self,grid,nr,cvs,kappas,input=None,potential=None,custom_cv=None,
                      temp=300*kelvin,press=1e5*pascal,mdsteps=10000,timestep=0.5*femtosecond,h5steps=5,timecon_thermo=100*femtosecond,timecon_baro=1000*femtosecond):
        '''
            **Arguments**

            grid
                the grid number

            nr
                the number of the grid point in this grid

            cvs
                array/list of collective variables corresponding to this point
                expressed in units as defined in input

            kappas
                array/list of kappas corresponding to each collective variable
                expressed in units of kJ/mol/units**2 for each collective variable

            **Optional Arguments**

            input
                yaml dict or OGRe_Input object

            potential
                file in which a potential function is defined
                or an AbstractPotential object for analytic potentials

            custom_cv
                module in which a get_cv(ff) and adapt_structure(ff,cv) function
                are defined

        '''
        # If input is yaml dict, convert it to attribute object using SimpleNamespace
        if isinstance(input,dict):
            input = SimpleNamespace(**input)

        # Initialize a custom log and timer object to allow multiprocessing with dask
        log, timer = Log()

        self.log = log
        self.timer = timer

        # Input parameters
        self.input = input

        # Grid point parameters
        self.grid = grid
        self.nr = nr
        self.cvs = cvs
        self.kappas = kappas

        # Simulation parameters
        self.potential = potential
        self.custom_cv = custom_cv
        self.temp = input.temp if hasattr(input, 'temp') else temp
        self.press = input.press if hasattr(input, 'press') else press
        self.timestep = input.timestep if hasattr(input, 'timestep') else timestep
        self.mdsteps = input.mdsteps if hasattr(input, 'mdsteps') else mdsteps
        self.h5steps = input.h5steps if hasattr(input, 'h5steps') else h5steps
        self.timecon_thermo = input.timecon_thermo if hasattr(input, 'timecon_thermo') else timecon_thermo
        self.timecon_baro = input.timecon_baro if hasattr(input, 'timecon_baro') else timecon_baro

        # Evaluate units
        if hasattr(input, 'cv_units'):
            if not isinstance(input.cv_units, list):
                input.cv_units = [input.cv_units]
            self.cv_units = np.array([eval(unit) if unit is not None else 1.0 for unit in input.cv_units])
        else:
            self.cv_units = np.ones(len(cvs))

        if hasattr(input, 'fes_unit'):
            self.fes_unit = eval(input.fes_unit)
        else:
            self.fes_unit = 1

        # Convert all quantities to atomic units
        self.cvs    = np.asarray(self.cvs)*self.cv_units
        self.kappas = np.asarray(self.kappas)*self.fes_unit/self.cv_units**2

        # Evalute potential
        if self.input.mode in ['analytic']:
            if isinstance(self.potential,str):
                # this should be a file name
                if os.path.exists(self.potential):
                    self.potential = load_potential_file(self.potential).Potential
                else:
                    raise IOError('Could not locate the potential module!')
            else:
                try:
                    assert issubclass(self.potential,AbstractPotential)
                except AssertionError:
                    raise ValueError('Your potential was not a module location or a class that is based on an AbstractPotential, try again.')

        # Evalute custom_cv
        elif self.input.mode == 'application':
            if isinstance(self.custom_cv,str):
                # this should be a file name
                if os.path.exists(self.custom_cv):
                    self.custom_cv = load_potential_file(self.custom_cv)
                else:
                    raise IOError('Could not locate the custom_cv module!')
            else:
                raise ValueError('The custom_cv should point to a python module with the cv functions!')
        else:
            raise NotImplementedError('Your combination of simulation parameters has not been implemented or is non sensical!')


    def simulate(self):
        # Select simulation mode
        if self.input.mode in ['analytic']:
            self.sim_analytic()
        elif self.input.mode == 'application':
            self.sim_application()
        else:
            raise NotImplementedError('Unknown mode')


    def sim_analytic(self):
        # Define umbrella
        potential = self.potential(self.cvs, self.kappas)

        # Initialize the input/output files:
        Path('trajs/').mkdir(parents=True, exist_ok=True)
        f=h5py.File('trajs/traj_{}_{}.h5'.format(int(self.grid),int(self.nr)),mode='w')
        hdf=SimpleHDF5Writer(f,step=self.h5steps)

        # Initialize the thermostat and the screen logger
        thermo = SimpleCSVRThermostat(self.temp, timecon=self.timecon_thermo)
        vsl = VerletScreenLog(step=100)

        # Initialize the US simulation
        verlet = SimpleVerletIntegrator(potential, self.timestep, hooks=[vsl, thermo, hdf], state=[potential.state],timer=self.timer,log=self.log)

        # Run the simulation
        verlet.run(self.mdsteps)

    def sim_application(self):
        self.log.set_level(self.log.medium)
        Path('logs/').mkdir(parents=True, exist_ok=True)

        with open('logs/log_{}_{}.txt'.format(self.grid,self.nr), 'w') as g:
            self.log.set_file(g)

            system = System.from_file('init.chk', log=self.log)

            # Create a force field object
            pars=[]
            for fn in os.listdir(os.getcwd()):
                if fn.startswith('pars') and fn.endswith('.txt'):
                    pars.append(fn)

            ff = ForceField.generate(system, pars, log=self.log, timer=self.timer, rcut=12*angstrom, alpha_scale=3.2, gcut_scale=1.5, smooth_ei=True, tailcorrections=True)

            # Create CV list
            cv = self.custom_cv.get_cv(ff)

            # Adapt structure, and save it for later use
            Path('init_structures/').mkdir(parents=True, exist_ok=True)
            self.custom_cv.adapt_structure(self,ff)

            # Define umbrella
            umbrella = ForcePartBias(system, log=self.log, timer=self.timer)
            for n,c in enumerate(cv):
                bias = HarmonicBias(self.kappas[n], self.cvs[n], c)
                umbrella.add_term(bias)

            # Add the umbrella to the force field
            ff.add_part(umbrella)

            # Initialize the input/output files:
            Path('trajs/').mkdir(parents=True, exist_ok=True)
            f=h5py.File('trajs/traj_{}_{}.h5'.format(int(self.grid),int(self.nr)),mode='w')
            hdf=HDF5Writer(f,step=self.h5steps)

            # Initialize the thermostat, barostat and the screen logger
            thermo = NHCThermostat(self.temp, timecon=self.timecon_thermo)
            baro = MTKBarostat(ff,self.temp, self.press, timecon=self.timecon_baro, vol_constraint=False)
            TBC = TBCombination(thermo, baro)
            vsl = VerletScreenLog(step=100)

            # Initialize the US simulation
            verlet = VerletIntegrator(ff, self.timestep, hooks=[vsl, TBC, hdf], state=[CVStateItem(cv)])

            # Run the simulation
            verlet.run(self.mdsteps)