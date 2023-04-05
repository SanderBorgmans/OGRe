#! /usr/bin/python
import ogre.post.sampling as sampling

__all__ = ['investigate_overlap']

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
