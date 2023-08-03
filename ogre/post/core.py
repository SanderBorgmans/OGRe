#! /usr/bin/python
import ogre.post.sampling as sampling

__all__ = ['ogre_refinement']

################################
# CORE CODE

def ogre_refinement(data,debug=False):
    """
        Load the Grid object through all layer files, and refine it based on the trajectories and the ogre metrics
    """
    print('Loading Grid')
    grid = sampling.Grid(data,debug=debug)

    print('Refining Grid')
    grid.refine()

    print('Outputting Grid')
    grid.output()
