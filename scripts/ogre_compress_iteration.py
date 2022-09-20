#! /usr/bin/python
import os,time,h5py,glob,warnings,yaml,sys
import numpy as np
from molmod.units import *

def compress_grid(data,index):
    """
        Create h5 file to compress trajectories to contain only required information
    """
    # Load all grids and divide in sub grids based on gridnr
    init = time.time()
    grid_names = glob.glob('grid[0-9][0-9].txt')
    numbers = [int(gn[4:6]) for gn in grid_names]

    grids = {}
    trajs = {}
    dkappas = {}
    identities = {}
    energies = {}

    gn = grid_names[numbers.index(index)]
    grid = np.genfromtxt(gn, delimiter=',',dtype=None,skip_header=1)
    for point in grid:
        # Define identity
        gnr = int(point[0])
        nr = int(point[1])
        identity = np.array([gnr,nr])

        # Define point
        if isinstance(point[2],float):
            cvs = np.array([point[2]])
        else:
            cvs = np.array([float(p) for p in point[2].decode().split('*')])

        # Define kappas
        if isinstance(point[3],float):
            kappas = np.array([point[3]])
        else:
            kappas = np.array([float(k) for k in point[3].decode().split('*')])
        try:
            with h5py.File('trajs/traj_{}_{}.h5'.format(gnr,nr),'r') as f:
                tr = f['trajectory/cv_values'][:].reshape((1,-1,len(kappas)))
                energy = f['trajectory/epot'][:].reshape((1,-1))

                if not gnr in grids.keys() or not gnr in trajs.keys():
                    grids[gnr]   = cvs
                    trajs[gnr]   = tr
                    dkappas[gnr] = kappas
                    energies[gnr] = energy
                    identities[gnr] = identity
                else:
                    grids[gnr]   = np.vstack((grids[gnr], cvs))
                    trajs[gnr]   = np.vstack((trajs[gnr], tr))
                    dkappas[gnr] = np.vstack((dkappas[gnr], kappas))
                    energies[gnr] = np.vstack((energies[gnr], energy))
                    identities[gnr] = np.vstack((identities[gnr], identity))

        except OSError:
            raise ValueError('Could not find one of the required trajectories! Exiting ...')

    assert len(grids.keys())==1

    h5 = h5py.File("trajs/compressed_{}.h5".format(index),mode='w')
    h5['grids'] = grids[index]
    h5['trajs'] = trajs[index]
    h5['dkappas'] = dkappas[index]
    h5['energies'] = energies[index]
    h5['identities'] = identities[index]

    print("Compressing trajectories took {} seconds.".format(time.time()-init))

if __name__ == '__main__':
    index = int(sys.argv[1])

    if os.path.exists('data.yml'):
        with open('data.yml','r') as f:
            data = yaml.full_load(f)

        compress_grid(data,index)
    else:
        raise IOError('No data file found!')