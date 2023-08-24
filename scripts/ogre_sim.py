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
    parser.add_option("--layer",
              action="store", type="int", dest="layer")
    parser.add_option("--nr",
              action="store", type="int", dest="nr")
    parser.add_option("--potential",
              action="store", type="str", dest="potential")
    parser.add_option("--custom_cv",
              action="store", type="str", dest="custom_cv")
    (options, args) = parser.parse_args(sys.argv[1:])

    # Load data file
    if os.path.exists('data.yml'):
        with open('data.yml','r') as f:
            data = yaml.full_load(f)
    else:
        raise AssertionError('There was no data file!')
    
    if not hasattr(options, 'potential'): options.potential = './potential.py'
    if not hasattr(options, 'custom_cv'): options.custom_cv = './custom_cv.py'

    if data['mode'] == 'analytic':
        sim = OGRe_Simulation(options.layer,options.nr,input=data,potential=options.potential)
    elif data['mode'] == 'application':
        sim = OGRe_Simulation(options.layer,options.nr,input=data,custom_cv=options.custom_cv)
    else:
        raise ValueError('An invalid mode was selected for terminal based use of OGRe.')
    sim.simulate()
