#!/usr/bin/env python

# The following two lines make sure that we dont need a serialized h5py module
import mpi4py
mpi4py.rc.initialize = False

import sys,os,yaml
import numpy as np
from optparse import OptionParser
from ogre.sim.core import *


if __name__ == "__main__":
    parser = OptionParser(usage="Usage: %prog [<options>]")
    parser.add_option("--grid",
              action="store", type="int", dest="grid")
    parser.add_option("--nr",
              action="store", type="int", dest="nr")
    (options, args) = parser.parse_args(sys.argv[1:])

    # Load data file
    if os.path.exists('data.yml'):
        with open('data.yml','r') as f:
            data = yaml.full_load(f)
    else:
        raise AssertionError('There was no data file!')

    # Load cvs and kappas from relevant grid file
    fname = 'grid{0:0=2d}.txt'.format(options.grid)
    grid = np.genfromtxt(fname, delimiter=',',dtype=None,skip_header=1)
    if grid.size==1: # problem with single line files
        grid = np.array([grid])
    
    point = grid[options.nr]
    assert point[0]==options.grid
    assert point[1]==options.nr

    if isinstance(point[2],float):
        cvs = np.array([point[2]])
    else:
        cvs = np.array([float(p) for p in point[2].decode().split('*')])

    # Define kappas
    if isinstance(point[3],float):
        kappas = np.array([point[3]])
    else:
        kappas = np.array([float(k) for k in point[3].decode().split('*')])


    if data['mode'] == 'analytic':
        sim = OGRe_Simulation(options.grid,options.nr,cvs,kappas,input=data,potential='./potential.py')
    elif data['mode'] == 'application':
        sim = OGRe_Simulation(options.grid,options.nr,cvs,kappas,input=data,custom_cv='./custom_cv.py')
    else:
        raise ValueError('An invalid mode was selected for terminal based use of OGRe.')
    sim.simulate()
