#! /usr/bin/python
import sys, yaml, os, shutil
from optparse import OptionParser, OptionGroup
from ogre.input.utils import *

# taken from https://stackoverflow.com/questions/392041/python-optparse-list
def callback(option, opt, value, parser):
    """
        This function makes a list from a comma separated string
    """
    setattr(parser.values, option.dest, [float(v) for v in value.split(',')])

def callback_string(option, opt, value, parser):
    """
        This function makes a list from a comma separated string
    """
    setattr(parser.values, option.dest, [str(v) for v in value.split(',')])


if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("Usage: ogre_input.py [<options>]")
        sys.exit(1)

    
    parser = OptionParser(usage="Usage: %prog [<options>]")
    parser.add_option("--kappas",
                action="callback", type="string", dest="kappas", callback=callback, help="set umbrella constants, comma separated list")
    parser.add_option("--spacings",
                action="callback", type="string", dest="spacings", callback=callback, help="CV spacings, comma separated list")
    parser.add_option("--edges",
                action="callback", type="string", dest="edges", callback=callback, help="minimum and maximum value for each CV, comma separated list")
    parser.add_option("--runup",
                action="store", type="int", dest="runup", help="number of steps to remove for equilibration [default: %default]", default=0)
    parser.add_option("--cv_units",
                  action="callback", type="string", dest="cv_units", callback=callback_string, help="CV units, comma separated list")
    parser.add_option("--fes_unit",
                  action="store", type="str", dest="fes_unit", help="fes unit [default: %default]", default="kjmol")
    parser.add_option("--CONFINEMENT_THR",
                action="store", type="float", dest="CONFINEMENT_THR", help="minimal percentage of the simulation that should be contained in the hypervolume defined by all the surrounding grid points to be considered as non-deviating [default: %default]", default=0.3)
    parser.add_option("--OVERLAP_THR",
                action="store", type="float", dest="OVERLAP_THR", help="minimal percentage for the overlap of the histograms of two neighbouring trajectories [default: %default]", default=0.3)
    parser.add_option("--KAPPA_GROWTH_FACTOR",
                action="store", type="float", dest="KAPPA_GROWTH_FACTOR", help="factor by which the kappa value is multiplied if the trajectory is deviating [default: %default]", default=2.0)
    parser.add_option("--MAX_LAYERS",
                action="store", type="int", dest="MAX_LAYERS", help="maximum number of grid layers that will be generated by the program, this in turn defines the minimal step size for each CV [default: %default]", default=1)
    parser.add_option("--MAX_KAPPA",
                action="store", type="float", dest="MAX_KAPPA", help="maximum value for kappa, if the protocol would attempt to increase the kappa value for a deviating simulation above this value, the free energy for this region is simply considered too high, and further refinement of this region is halted")

    (options, args) = parser.parse_args(sys.argv[1:])

    inp = OGRe_Input(vars(options))

    # Based on the input settings, make the grid
    inp.make_grid()

    # Copy the grid00.txt file to run.txt
    from shutil import copyfile
    copyfile('layer00.txt', 'run.txt') # create initial run file
