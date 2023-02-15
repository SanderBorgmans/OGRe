#! /usr/bin/python
import os,sys,copy,warnings,glob,h5py
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as pt

import ogre.post.grid_utils as grid_utils
import ogre.post.sampling as sampling
from ogre.input.utils import get_cv_units

from molmod.units import *


#import pymmbar
__all__ = ['generate_fes_thermolib','investigate_overlap']

################################
# Free energy evaluation

# An implementation for the evaluation of the free energy is provided below using ThermoLIB
# Similar functions can be created by the user based on this example

import thermolib 

def generate_fes_thermolib(data,index=None,step_factor=0.1,error_estimate='mle_f',suffix=None):
    # Generate the required colvars and metadata file
    #print(data['edges'])
    locations, trajs, kappas, _, _ = grid_utils.load_grid(data,index,verbose=False)
    cv_units = get_cv_units(data)
    fes_unit = eval(data['fes_unit'])
    edges = { k:copy.copy(np.array(v)*cv_units) for k,v in data['edges'].items() }
    spacings = data['spacings'] * cv_units
    steps = [spacing*step_factor for spacing in spacings]
    #print(steps)
    temp = data['temp'] if 'temp' in data else 300.*kelvin

    if not 'temp' in data:
        warnings.warn('No temperature (temp) attribute was found in the data.yml file. Taking 300 K as a default value.')

    # Ravel the trajectories and grids
    rtrajs = np.zeros((0,*trajs[0][:,data['runup']:].shape[1:]))
    rlocations = np.zeros((0,*locations[0].shape[1:]))
    rkappas = np.zeros((0,*kappas[0].shape[1:]))
    for key in locations.keys():
        rtrajs = np.vstack((rtrajs,trajs[key][:,data['runup']:]))
        rlocations = np.vstack((rlocations,locations[key]))
        rkappas = np.vstack((rkappas,kappas[key]))

    print("The full trajectories shape taken into account is: ", rtrajs.shape)

    # Convert rlocations and rkappas to atomic units
    rkappas *= 1/cv_units**2 # energy unit remains fes_unit (since thermolib works with kjmol wham units for kappa, no conversion is performed)
    rlocations *= cv_units

    bins = [int(np.round((edges['max'][i]-edges['min'][i])/(steps[i]))) for i,_ in enumerate(edges['min'])]

    #WHAM analysis
    filename = 'metadata'
    if not index is None:
        filename += '_{}'.format(index)

    if not suffix is None:
        filename += '_{}'.format(suffix)

    grid_utils.write_colvars(filename,rtrajs,rlocations,rkappas,verbose=False)

    # Launch thermolib
    fes_err = np.array([np.nan,np.nan])
    bin_edges = [np.linspace(edges['min'][n],edges['max'][n],b+1) for n,b in enumerate(bins)]

    if len(steps) == 1:
        # 1D CASE
        # here the bias potential functional form can be adapted by providing the appropriate value to 'bias_potential'
        _, biasses, trajectories = thermolib.read_wham_input(filename, path_template_colvar_fns='%s', stride=1, verbose=False)
        
        hist = thermolib.Histogram1D.from_wham_c(bin_edges[0], trajectories, biasses, temp, error_estimate=error_estimate,
                                       verbosity='low', convergence=1e-7, Nscf=10000)
                                       
        fes = thermolib.BaseFreeEnergyProfile.from_histogram(hist, temp)
        fes.set_ref(ref='min')
        grid = fes.cvs.copy().reshape(*bins,len(steps))
        fes_array = fes.fs.copy().reshape(*bins)

        if error_estimate is not None:
            if fes.flower is not None and fes.fupper is not None: 
                fes_err = np.array([fes.flower.copy().reshape(*bins),fes.fupper.copy().reshape(*bins)])

    elif len(steps) == 2:
        # 2D CASE
        # here the bias potential functional form can be adapted by providing the appropriate value to 'bias_potential'
        _, biasses, trajectories = thermolib.read_wham_input_2D(filename, path_template_colvar_fns='%s', stride=1, verbose=False)

        hist = thermolib.Histogram2D.from_wham_c(bin_edges, trajectories, biasses, temp, error_estimate=error_estimate,
                                       verbosity='low', convergence=1e-7, Nscf=10000, overflow_threshold=1e-150)

        fes = thermolib.FreeEnergySurface2D.from_histogram(hist, temp)
        fes.set_ref(ref='min')

        grid = np.array(np.meshgrid(fes.cv1s.copy(),fes.cv2s.copy())).T.reshape(*bins,len(steps))

        fes_array = fes.fs.copy().reshape(*bins)
        if error_estimate is not None:
            if fes.flower is not None and fes.fupper is not None: 
                fes_err = np.array([fes.flower.copy().reshape(*bins),fes.fupper.copy().reshape(*bins)])
    else:
        raise NotImplementedError('Thermolib does not support N-dim free energy evaluation at this point.')

    grid = grid/cv_units
    fes_array = fes_array/fes_unit
    fes_err = fes_err/fes_unit

    grid_utils.write_fes(data,grid,fes_array,index,suffix=suffix,fes_err=fes_err)
    grid_utils.plot_fes(data,grid,fes_array,index,suffix=suffix,fes_err=fes_err)
    
################################
# CORE CODE

def investigate_overlap(data,debug=False):
    """
        Check the overlap heuristic, and create new refinement
    """
    print('Loading Grid')
    grid = sampling.Grid(data,debug=debug)

    print('Refining Grid')
    grid.refine()

    print('Outputting Grid')
    grid.output()
