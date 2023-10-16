#!/usr/bin/env python

import os,h5py,numpy as np
from molmod.units import *
from pathlib import Path
from types import SimpleNamespace

__all__ = ['OGRe_Simulation']


def load_potential_file(file_loc):
    import importlib.util
    spec = importlib.util.spec_from_file_location("module",file_loc)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
    

class OGRe_Simulation(object):
    def __init__(self,layer,nr,input=None,md_params={}):
        '''
            **Arguments**

            layer
                the layer number

            nr
                the number of the grid point in this layer

            **Optional Arguments**

            input
                yaml dict or OGRe_Input object
            
            md_params
                dictionary with the parameters for the MD simulation

        '''
        # If input is yaml dict, convert it to attribute object using SimpleNamespace
        if isinstance(input,dict):
            input = SimpleNamespace(**input)

        # Input parameters
        self.input = input

        # Grid point parameters
        self.layer = layer
        self.nr = nr

        # Load the rest from the grid file
        fname = 'layer{0:0=2d}.txt'.format(self.layer)
        
        grid = np.genfromtxt(fname, delimiter=',',dtype=None,skip_header=1)
        if grid.size==1: # problem with single line files
            grid = np.array([grid])

        point = grid[self.nr]
        assert point[0]==self.layer
        assert point[1]==self.nr

        if isinstance(point[2],float):
            cvs = np.array([point[2]])
        else:
            cvs = np.array([float(p) for p in point[2].decode().split('*')])

        # Define kappas
        if isinstance(point[3],float):
            kappas = np.array([point[3]])
        else:
            kappas = np.array([float(k) for k in point[3].decode().split('*')])

        dtype = point[4].decode()

        self.cvs = cvs
        self.kappas = kappas
        self.type = dtype

        # Simulation parameters
        self.md_params = SimpleNamespace(**md_params) # convert dict to attributes

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



    def simulate(self):        
        # Set up simulation

        # Create trajectories folder if it does not exist
        Path('trajs/').mkdir(parents=True, exist_ok=True)

        # h5py file with correct name
        f=h5py.File('trajs/traj_{}_{}.h5'.format(int(self.layer),int(self.nr)),mode='w')

        # Run simulation

        # Fix me
        raise NotImplementedError
