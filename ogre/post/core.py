#! /usr/bin/python
import os,sys,copy,warnings,glob
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
    grids, trajs, kappas, _, _ = grid_utils.load_grid(data,index,verbose=False)
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
    rgrids = np.zeros((0,*grids[0].shape[1:]))
    rkappas = np.zeros((0,*kappas[0].shape[1:]))
    for key in grids.keys():
        rtrajs = np.vstack((rtrajs,trajs[key][:,data['runup']:]))
        rgrids = np.vstack((rgrids,grids[key]))
        rkappas = np.vstack((rkappas,kappas[key]))

    print("The full trajectories shape taken into account is: ", rtrajs.shape)

    # Convert rgrids and rkappas to atomic units
    rkappas *= 1/cv_units**2 # energy unit remains fes_unit (since thermolib works with kjmol wham units for kappa, no conversion is performed)
    rgrids *= cv_units

    bins = [int(np.round((edges['max'][i]-edges['min'][i])/(steps[i]))) for i,_ in enumerate(edges['min'])]

    #WHAM analysis
    filename = 'metadata'
    if not index is None:
        filename += '_{}'.format(index)

    if not suffix is None:
        filename += '_{}'.format(suffix)

    grid_utils.write_colvars(filename,rtrajs,rgrids,rkappas,verbose=False)

    # Launch thermolib
    fes_err = np.array([np.nan,np.nan])
    if len(steps) == 1:
        # 1D CASE
        temp_none, biasses, trajectories = thermolib.read_wham_input(filename, path_template_colvar_fns='%s', stride=1, verbose=False),
        bin_edges = [np.linspace(edges['min'][n],edges['max'][n],b+1) for n,b in enumerate(bins)]

        #1D case
        hist = thermolib.Histogram1D.from_wham_c(bin_edges[0], trajectories, biasses, temp, error_estimate=error_estimate,
                                       verbosity='low', convergence=1e-7, Nscf=10000)

        fes = thermolib.BaseFreeEnergyProfile.from_histogram(hist, temp)
        fes.set_ref(ref='min')
        grid = fes.cvs.copy().reshape(*bins,len(steps))
        fes_array = fes.fs.copy().reshape(*bins)
        if error_estimate is not None:
            fes_err = np.array([fes.flower.copy().reshape(*bins),fes.fupper.copy().reshape(*bins)])

    elif len(steps) == 2:
        # 2D CASE
        hist = thermolib.Histogram2D.from_wham_c(bin_edges, trajectories, biasses, temp, error_estimate=error_estimate,
                                       verbosity='low', convergence=1e-7, Nscf=10000, overflow_threshold=1e-150)

        fes = thermolib.FreeEnergySurface2D.from_histogram(hist, temp)
        fes.set_ref(ref='min')

        grid = np.meshgrid(fes.cv1.copy(),fes.cv2s.copy()).reshape(*bins,len(steps))
        fes_array = fes.fs.copy().reshape(*bins)
        if error_estimate is not None:
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

def format_grid_file(data,test=False,verbose=True):
    if test: # if test the user is only interested in the possible outcome
        return
    # Get all compressed trajectory identities
    identities_compressed = []
    ctnames = glob.glob('trajs/compressed_*.h5')
    for ct in ctnames:
        with h5py.File(ct,'r') as hct:
            identities_ct = [tuple(val) for val in hct['identities']]
            for ict in identities_ct:
                identities_compressed.append(ict)

    # Get all trajectory identities
    tnames = glob.glob('trajs/traj_*_*.h5')
    ids = [tn.split('/')[1].split('.')[0].split('_') for tn in tnames]
    identities_traj = [(int(i[1]),int(i[2])) for i in ids]

    # Get all gridpoint identities
    gnames = glob.glob('grid*.txt')
    identities_gp = []
    gp_lines = {}
    for gn in gnames:
        if gn=='grid_restart.txt':
            continue # skip restart grid
        else:
            gnr = int(gn.split('.')[0][4:])
            if 'MAX_LAYERS' in data and not gnr < data['MAX_LAYERS']:
                continue
            with open(gn,'r') as f:
                lines = f.readlines()
            lines = lines[1:] # skipheader
            for l in lines:
                id_line = l.split(',')[:2]
                id = (int(id_line[0]),int(id_line[1]))
                data = l.split(',')[2:]
                identities_gp.append(id)
                gp_lines[id] = data

    # Find those elements in identities_gp that are not in identities_traj and sort them for clean grid file
    identities = list(set(identities_gp)-set(identities_traj)-set(identities_compressed))
    identities = sorted(identities, key=lambda tup: (tup[0],tup[1]))

    with open('run.txt','w') as f:
        f.write('grid,nr,cvs,kappas\n')
        for id in identities:
            f.write('{},{},{}'.format(id[0],id[1],",".join(gp_lines[id])))

    if len(identities)==0 and verbose:
        print('It might be a good idea to compress your data using ogre_compress_iteration.py, if you do not need the trajectory data for other purposes.')

def investigate_overlap(data,test=False):
    """
        Check the overlap heuristic, and create new refinement
    """
    grids, trajs, kappas, identities = grid_utils.load_grid(data)

    major_grid = None
    grid_idx = sorted(grids.keys()) # grids are identified by a grid number
    for idx in grid_idx:
        # Create grid object based on larger grids
        grid = sampling.create_grid(idx,grids[idx],trajs[idx],kappas[idx],identities[idx],data,major_grid=major_grid)

        # Check overlap and create finer grid with corresponding reference points
        grid.check_overlap(idx,data)
        grid.refine_grid() # create additional grid points (and refine confinement for those with deviating trajectories)

        # Throw away points generated outside fringe and output this grid
        grid_utils.cut_edges(data,grid)
        grid_utils.write_grid(idx,data,grid,test=test)
        grid_utils.plot_grid(idx,data,grid)

        # Pass this grid object to the next iteration
        major_grid = grid

    # Write a single data file with all the points which have to be simulated from all grids
    format_grid_file(data,test=test)
