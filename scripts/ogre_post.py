#! /usr/bin/python
import os,yaml,sys
from optparse import OptionParser
from ogre.post.core import *

from molmod.units import *

parser = OptionParser(usage="Usage: %prog [<options>]")
parser.add_option("--refresh",
          action="store_true", dest="refresh", help="perform a clean run [default: %default]", default=False)
parser.add_option("--overlap",
          action="store_true", dest="overlap", help="only perform the overlap analysis, debugging purposes [default: %default]", default=False)
parser.add_option("--fes",
          action="store_true", dest="fes", help="construct the FES for the subsequent refinements [default: %default]", default=False)
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
    investigate_overlap(data,test=options.test)

if options.fes:
    if options.fes_index==0:
        # Calculate FES with all data
        generate_fes_thermolib(data)
    else:
        generate_fes_thermolib(data,index=options.fes_index-1) # index corresponds to grid number, which starts counting at 0