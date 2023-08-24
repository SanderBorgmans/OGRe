#! /usr/bin/python
import os,yaml,sys
from optparse import OptionParser
from ogre.post.core import *
from ogre.post.fes import *

from molmod.units import *


if (len(sys.argv) < 2):
    print("Usage: ogre_post.py [<options>]")
    print("This postprocessing script expects you to have your trajectory data stored in a 'trajs' folder using the h5 file format,\n"
           + "with each trajectory representing a US simulation idenitified by its grid number and simulation index; e.g. trajs/traj_0_0.h5.\n" 
           + "The only required attribute is 'trajectory/cv_values' representing the CVs during the simulations with shape (N_sim, N_CV)." 
           )
    sys.exit(1)

parser = OptionParser(usage="Usage: %prog [<options>]")
parser.add_option("--refresh",
          action="store_true", dest="refresh", help="perform a clean run, removes all pickle files [default: %default]", default=False)
parser.add_option("--overlap",
          action="store_true", dest="overlap", help="perform the overlap analysis, this should not be repeated before performing new simulations! [default: %default]", default=False)
parser.add_option("--fes",
          action="store_true", dest="fes", help="calculate the FES for the current data [default: %default]", default=False)
parser.add_option("--test",
          action="store_true", dest="test", help="perform the overlap analysis, without creating/changing/removing any files [default: %default]", default=False)
parser.add_option("--fes_index",
                  action="store", type="int", dest="fes_index", help="number of grid iterations to take into account [default: %default]", default=0)
parser.add_option("--plot_deviating",
          action="store_true", dest="plot_deviating", help="plot all deviating trajectories [default: %default]", default=False)
parser.add_option("--plot_overlap",
          action="store_true", dest="plot_overlap", help="plot all non overlapping trajectories [default: %default]", default=False)
(options, args) = parser.parse_args(sys.argv[1:])

# Load data file
if os.path.exists('data.yml'):
    with open('data.yml','r') as f:
        data = yaml.full_load(f)
else:
    raise AssertionError('There was no data file!')

# Store the plot
data['plot_deviating'] = options.plot_deviating
data['plot_overlap'] = options.plot_overlap

if options.refresh or options.overlap:
    # if the overlap is checked, these files should be regenerated
    # this also prevents to continuous adaptation of the grid file without new simulations
    if os.path.exists('trajs.pkl'):
        os.remove('trajs.pkl')
    if os.path.exists('grids.pkl'):
        os.remove('grids.pkl')
    if os.path.exists('kappas.pkl'):
        os.remove('kappas.pkl')
    if os.path.exists('identities.pkl'):
        os.remove('identities.pkl')    

if options.overlap:
    # Check overlap and generate fine grid if necessary
    ogre_refinement(data,debug=options.debug)

if options.fes:
    if options.fes_index==0:
        # Calculate FES with all data
        generate_fes(data)
    else:
        generate_fes(data,index=options.fes_index-1) # index corresponds to grid number, which starts counting at 0
