#! /usr/bin/python
import numpy as np
import matplotlib.pyplot as pt

from scipy.spatial import cKDTree as KDTree # identical to KDTree but a lot faster before v1.6.0 
from scipy.spatial.distance import jensenshannon
from molmod.units import *
from molmod.constants import *
from ogre.post.sampling_utils import *
from ogre.post.nodes import *
from ogre.input.utils import get_cv_units
import ogre.post.grid_utils as grid_utils


# NOTE #
# The grid is defined in terms of 'cv_units' to avoid conversion errors
# Some quantities/constants are defined in sampling_utils.py

class Grid:
    def __init__(self,data,debug=False):
        self.data = data
        self._load_data()
        self.layer_idx = sorted(self.locations.keys()) # grid layers are identified by a number
        self.layer_dictionary = {-1: None} # this holds references to the grid layer objects
        self.debug = debug # if this is True no files are altered, ideal for debugging purposes

    def _load_data(self):
        # Load data variables or assign default values - this is just for convenience to avoid self.data[..]
        self.RUN_UP_TIME          = self.data['runup']                if 'runup'                in self.data else 0
        self.MAX_LAYERS           = self.data['MAX_LAYERS']           if 'MAX_LAYERS'           in self.data else 1
        self.CONFINEMENT_THR      = self.data['CONFINEMENT_THR']      if 'CONFINEMENT_THR'      in self.data else 0.30
        self.CONVERGENCE_THR      = self.data['CONVERGENCE_THR']      if 'CONVERGENCE_THR'      in self.data else None
        self.OVERLAP_THR          = self.data['OVERLAP_THR']          if 'OVERLAP_THR'          in self.data else 0.30
        self.KAPPA_GROWTH_FACTOR  = self.data['KAPPA_GROWTH_FACTOR']  if 'KAPPA_GROWTH_FACTOR'  in self.data else 2.
        self.MAX_KAPPA            = self.data['MAX_KAPPA']            if 'MAX_KAPPA'            in self.data else np.inf
        self.HISTOGRAM_BIN_WIDTHS = self.data['HISTOGRAM_BIN_WIDTHS'] if 'HISTOGRAM_BIN_WIDTHS' in self.data else [spacing/self.MAX_LAYERS/2. for spacing in self.data['spacings']]
        self.PLOT_CON             = self.data['plot_con']             if 'plot_con'             in self.data else False
        self.PLOT_OVERLAP         = self.data['plot_overlap']         if 'plot_overlap'         in self.data else False
        self.PLOT_CONSISTENT      = self.data['plot_consistent']      if 'plot_consistent'      in self.data else False
        self.CONSISTENCY_THR      = self.data['CONSISTENCY_THR']      if 'CONSISTENCY_THR'      in self.data else None
        self.report_consistency   = self.data['report_consistency']   if 'report_consistency'   in self.data else False

        # Load up other data variables which are required
        self.edges = self.data['edges']
        self.spacings = self.data['spacings']
        self.cv_units = get_cv_units(self.data)
        self.fes_unit = eval(self.data['fes_unit'])

        # Load simulation data
        locations, trajs, kappas, identities, types = grid_utils.load_grid(self.data)
        self.locations = locations
        self.trajs = {k: t/self.cv_units for k,t in trajs.items()} # work in natural units for the grid
        self.kappas = kappas
        self.identities = identities
        self.types = types

        # Init FES property which can be used as consistency check
        self.fes = None
        if self.CONSISTENCY_THR is not None:
            from ogre.post.fes import generate_fes
            self.fes = generate_fes(self.data,interactive=True,error_estimate=None) # we do not need error estimate for this

    def refine(self):
        for idx in self.layer_idx:
            if self.layer_dictionary[idx-1] is not None and len(self.layer_dictionary[idx-1].sublayer_nodes)==0:
                break

            # Create layer object
            layer = self.create_layer(idx)
            layer.check_extreme_kappa()

            # Check overlap and create finer grid with corresponding reference points
            layer.refine_layer()

            # Check whether the layer contains inconsistent nodes, if so, refrain from refining any lower lying layers
            # until this is resolved
            if not layer.check_consistent():
                print('Halting refinement as some nodes in layer {} are not consistent, which will impact further refinement of their environment.'.format(idx))
                self.layer_dictionary[idx] = layer
                return # stop the grid refinement immediately

            layer.generate_sublayer() # create additional grid points (and refine confinement for those with deviating trajectories)
            layer.cut_edges() # throw away points generated outside fringe
            
            # Save this layer object
            self.layer_dictionary[idx] = layer        
    
    def output(self):
        # Write the layer information and create nice plots as a visualization of this information
        for idx in self.layer_idx:
            if not idx in self.layer_dictionary.keys(): continue
            if self.debug:
                grid_utils.dump_layer(self.layer_dictionary[idx])
            grid_utils.plot_layer(self.layer_dictionary[idx])
            grid_utils.write_layer(self.layer_dictionary[idx],self.MAX_LAYERS,debug=self.debug)
            
        # Write a single data file with all the points which have to be simulated from all layers
        grid_utils.format_layer_files(self.data,debug=self.debug)

        # If there are extreme kappa nodes, let the user know
        for idx in self.layer_idx:
            if not idx in self.layer_dictionary.keys(): continue
            if any([node.extreme_kappa for node in self.layer_dictionary[idx].nodes if isinstance(node,Node)]):
                with open('extreme_kappas','w') as f:
                    pass # the file does not need any input


    @staticmethod
    def get_node_from_nodes(tree,point):
        found,index = find_node_index(tree,point)
        if found: return index


    def create_layer(self,index):
        """
        This function returns a Layer based on the grid points and the trajectories from the Grid object with index 'index'
        The grid points should be defined on a single distance scale (for each refinement a different grid layer should be defined)
        * Arguments
            locations       NxD array with all grid point locations
            trajs           NxMxD array with trajectory of M steps for every grid point
            kappas          NxD array with all kappa values
            identities      list with N entries, with the identity of each grid point
            types           list with N entries, the type of that grid point
        """
        # Perform sanity check
        steps = np.array(self.spacings)/(2**index)
        if any(steps/2.<precision): 
            raise ValueError('Making another refinement would require increasing DECIMALS. Adapt this in the sampling_utils.py file.')

        # Define variables
        nodes = []
        superlayer = self.layer_dictionary[index-1]

        # Create a KDTree with the realized sublayer nodes of the superlayer (these are both the Nodes and the BenchmarkNodes that span the current layer)
        # We create a KDTree to be able to identify the individual simulations with the nodes
        tree = None
        if superlayer is not None:
            tree = KDTree(np.asarray([node.loc for node in superlayer.sublayer_nodes]))

        # Iterate over all points of the current layer
        for n,point in enumerate(self.locations[index]):
            # CASE 1: the node is real, i.e. not a benchmark node
            if 'benchmark' not in self.types[index][n]: # ['node','new_node','deviant_node']
                # the convergence and sanity of deviant nodes will be checked later
                nodes.append(Node(point,self.trajs[index][n],self.kappas[index][n],self.identities[index][n]))

                # also alter these in the sublayer of the superlayer
                if superlayer is not None:
                    #print(point, np.asarray([node.loc for node in superlayer.sublayer_nodes]))
                    rn_idx = self.get_node_from_nodes(tree,point)
                    if rn_idx is None:
                        continue
                    superlayer.sublayer_nodes[rn_idx] = Node(point,self.trajs[index][n],self.kappas[index][n],self.identities[index][n])

            # CASE 2: the node is a benchmark node, and needs to be updated, i.e. a (deviant) realized benchmark node
            elif 'realized_benchmark_node' in self.types[index][n]: #['deviant_real_benchmark_node']: 
                # this requires a superlayer, and it should exist by definition if sublayer nodes are present
                assert superlayer is not None

                # check the sanity of the new node
                tmp_node = Node(point,self.trajs[index][n],self.kappas[index][n],self.identities[index][n])
                confined = superlayer.sanity_check(tmp_node,factor=2.,plot=False) # choosing a factor 2 effectively checks confinement for current layer
                converged = superlayer.convergence_check(tmp_node) 
                tmp_node.confined = confined

                if tmp_node.confined:
                    tmp_node.converged = converged
                    
                    if not tmp_node.converged:
                        print(tmp_node.identity, ': (benchmark) was not converged, try a longer run time, or a higher kappa to increase convergence.')


                # although unlikely, we should check whether a deviant benchmark node would cross the max_kappa
                # if so set the sanity to True, but throw a warning
                if not tmp_node.sane:
                    tmp_node.set_extreme_kappa(self.MAX_KAPPA,self.KAPPA_GROWTH_FACTOR)
                    if tmp_node.extreme_kappa:
                        tmp_node.confined = True
                        tmp_node.converged = True
                        print('A benchmark node at {} tried to increase its kappa value above the allowed value. Setting sanity to True ...'.format(",".join(['{:.8f}'.format(p) for p in tmp_node.loc])))

                # identify the simulation with a node from the superlayer.sublayer_nodes and replace it before it gets added below
                rn_idx = self.get_node_from_nodes(tree,point)
                if rn_idx is None:
                    # this happens when virtual nodes are suddenly recognized as deviant, and all dependent nodes temporarily disappear
                    # since this can only happen for already existing grid files, they will never be deleted, and later recognized when required
                    # even when a realized benchmark node would be replaced by a new node, it will recognize the location, and take its place
                    continue
                superlayer.sublayer_nodes[rn_idx] = RealizedBenchmarkNode(tmp_node)

            # CASE 3: the node is a benchmark node, but does not need updating
            else:
                # other node types should either not be replaced or not occur
                raise AssertionError('This type {} should not occur!'.format(self.types[index][n]))
                #assert self.types[n] not in ['virtual_benchmark_node','deviant_virtual_benchmark_node','superlayer_benchmark_node','deviant_superlayer_benchmark_node']
        

        # Add the benchmark nodes to the grid
        if not superlayer is None:
            for node in superlayer.sublayer_nodes:
                if isinstance(node,BenchmarkNode):
                    nodes.append(node)

        return Layer(self,index,nodes,steps) 


class Layer(object):
    def __init__(self,grid,index,nodes,steps):
        self.grid = grid
        self.index = index
        self.nodes = nodes # these are the nodes/reference nodes that contain trajectories
        self.steps = steps
        
        self.superlayer = self.grid.layer_dictionary[index-1]
        self.sublayer_nodes = []  # this will be calculated

        self.overlap = None

    def check_extreme_kappa(self):
        if not self.grid.MAX_KAPPA is None:
            for node in self.nodes:
                if isinstance(node, Node):
                    node.set_extreme_kappa(self.grid.MAX_KAPPA, self.grid.KAPPA_GROWTH_FACTOR)

    def check_consistent(self):
        return all([node.consistent for node in self.nodes if isinstance(node,Node)])

    def cut_edges(self):
        edges = self.grid.edges
        spacings = self.grid.spacings
        for i,_ in enumerate(spacings):
            self.sublayer_nodes = [node for node in self.sublayer_nodes if node.loc[i] <= edges['max'][i]+precision and node.loc[i] >= edges['min'][i]-precision]

    def convergence_check(self,node):
        """
        This tests whether the PDFs of the first half and the second half of the trajectory are consistent enough as a convergence check
        """
        if self.grid.CONVERGENCE_THR is None:
            return True

        trajs=node.trajs
        t = trajs[0][self.grid.RUN_UP_TIME:]

        binwidths = self.grid.HISTOGRAM_BIN_WIDTHS
        bins = tuple([int(((self.grid.edges['max'][i]+self.grid.spacings[i])-(self.grid.edges['min'][i]-self.grid.spacings[i]))//binwidths[i]) for i in range(t.shape[1])])


        h1, _ = np.histogramdd(t[:len(t)//2], bins=bins, range=[(self.grid.edges['min'][i]-self.grid.spacings[i],
                                                                 self.grid.edges['max'][i]+self.grid.spacings[i]) for i in range(t.shape[1])], density=True)
        h2, _ = np.histogramdd(t[len(t)//2:], bins=bins, range=[(self.grid.edges['min'][i]-self.grid.spacings[i],
                                                                 self.grid.edges['max'][i]+self.grid.spacings[i]) for i in range(t.shape[1])], density=True)
        
        convergence = (1-jensenshannon(h1.ravel(),h2.ravel(),base=2)**2)
        converged = convergence >= self.grid.CONVERGENCE_THR
        if not converged:
            print(node.identity, ': was not converged ({}<{}), try a longer run time, or a higher kappa to increase convergence.'.format(np.round(convergence,2),self.grid.CONVERGENCE_THR))

        return converged

    
    def consistency_check(self,node):
        """
        This tests whether the trajectory data is consistent with the underlying FES
        """
        if self.grid.CONSISTENCY_THR is None or self.grid.fes is None:
            return True
                
        beta = 1/(boltzmann*self.grid.data['temp']*kelvin/self.grid.fes_unit)

        trajs=node.trajs
        t = trajs[0][self.grid.RUN_UP_TIME:]

        # Get the (biased) FES
        grid = self.grid.fes[0].copy() # in cv_units
        biased_fes = self.grid.fes[1].copy() # in fes_units

        bias = np.sum(0.5*node.kappas*(grid-node.loc)**2,axis=-1)
        biased_fes += bias

        # Calc expected biased probability
        biased_prob = np.exp(-beta*biased_fes)
        biased_prob[np.isnan(biased_prob)] = 0.0 # if fes is nan, this region was not sampled

        # Calculate bin_edges of the FES
        bin_edges, bin_widths = get_grid_edges(grid)

        # Calculate histogram of trajectory data with bin centers defined by the FES data
        h1, edges = np.histogramdd(t, bins=bin_edges, density=True)

        h1 = h1/np.sum(h1)
        density_biased_prob = np.round(biased_prob/np.sum(biased_prob),20)

        # Calculate consistency
        consistency = (1 - jensenshannon(density_biased_prob.ravel(), h1.ravel(), base=2)**2)
        consistent =  consistency >= self.grid.CONSISTENCY_THR

        if self.grid.report_consistency:
            print(node.identity, consistency)

        if self.grid.PLOT_CONSISTENT: # and not consistent:
            fig = pt.figure()
            ax = fig.gca()
            grid_utils.plot_ref_grid(ax,self,size=50.0,show_deviants=False) 
            grid_utils.plot_consistency(ax, node, edges, h1, density_biased_prob, bin_widths, self.steps)
            title = "Consistent = {}, {}".format(consistent, consistency)
            fig.suptitle(title)
            #pt.show()
            fig.savefig('consistency_{}.pdf'.format("-".join([str(i) for i in node.identity])),bbox_inches='tight')
            
        return consistent

    
    def sanity_check(self,node,factor=1.,plot=True):
        trajs = node.trajs

        t = trajs[0][self.grid.RUN_UP_TIME:]
        fullt = trajs[0]
        if len(trajs)>1:
            for traj in trajs[1:]:
                t = np.vstack((t,traj[self.grid.RUN_UP_TIME:]))
                fullt = np.vstack((t,traj))

        # check confinement ~ 1st moment
        mask = np.ones(t.shape[0],dtype=bool)
        for n,step in enumerate(self.steps):
            mask = mask & (t[:,n]<node.loc[n]+step/factor) & (t[:,n]>node.loc[n]-step/factor)
        confinement = sum(mask)/float(len(mask))
        sane = confinement >= self.grid.CONFINEMENT_THR

        if self.grid.PLOT_CON and plot: #and not sane:
            fig = pt.figure()
            ax = fig.gca()
            grid_utils.plot_ref_grid(ax,self,size=50.0,show_deviants=False) 
            grid_utils.plot_dev(ax, node, t, self.steps, self)
            title = "Sanity = {:4.3f}".format(confinement)
            fig.suptitle(title)
            #pt.show()
            fig.savefig('confinement_{}.pdf'.format("-".join([str(i) for i in node.identity])),bbox_inches='tight')
            
        return sane


    def overlap_check(self,node1,node2):
        trajs1 = node1.trajs
        trajs2 = node2.trajs

        overlap = 0
        assert len(trajs1)==1 # this has to be for non reference node
        t1 = trajs1[0]

        binwidths = self.grid.HISTOGRAM_BIN_WIDTHS
        bins = tuple([int(((self.grid.edges['max'][i]+self.grid.spacings[i])-(self.grid.edges['min'][i]-self.grid.spacings[i]))//binwidths[i]) for i in range(t1.shape[1])])
        
        #try:
        h1, edges = np.histogramdd(t1[self.grid.RUN_UP_TIME:], bins=bins, range=[(self.grid.edges['min'][i]-self.grid.spacings[i],
                                                                                self.grid.edges['max'][i]+self.grid.spacings[i]) for i in range(t1.shape[1])], density=True)
        #except FloatingPointError:
            # This happens if the trajectory falls outside of the edges


        bin_volume = np.prod([np.abs(edge[1] - edge[0]) for edge in edges])
        for t2 in trajs2:
            # Convert traj to histogram
            h2, _ = np.histogramdd(t2[self.grid.RUN_UP_TIME:], bins=bins, range=[(self.grid.edges['min'][i]-self.grid.spacings[i],
                                                                            self.grid.edges['max'][i]+self.grid.spacings[i]) for i in range(t1.shape[1])], density=True)
            # Check overlap of histograms
            overlap += np.sum(np.minimum(h1,h2)*bin_volume)
        overlap /= len(trajs2)

        if self.grid.PLOT_OVERLAP: # and overlap < self.grid.OVERLAP_THR:
            fig = pt.figure()
            ax = fig.gca()
            grid_utils.plot_ref_grid(ax,self,size=10.0,show_deviants=False) 
            grid_utils.plot_overlap(ax, node1, node2, t1, trajs2, self.steps, self)
            #fig.legend()
            fig.suptitle("Overlap = {:4.3f}".format(overlap))
            #pt.show()
            fig.savefig('overlap_{}_{}.pdf'.format("-".join([str(i) for i in node1.identity]),"-".join([str(i) for i in node2.identity])),bbox_inches='tight')
        return overlap

    def characterize_nodes(self):
        """
            Find and assign the neighbours for each point
        """
        neighbours = find_neighbours(self.steps,self.nodes)
        for n,node in enumerate(self.nodes):
            node.neighbours = neighbours[n]

    def sanity_analysis(self,ax):
        # Check Node confinement, convergence, and consistency
        for n,node in enumerate(self.nodes):
            if isinstance(node, Node) and not node.extreme_kappa:
                confined = self.sanity_check(node)
                # If the simulation went out of bounds, the overlap loses all meaning at this location
                if not confined:
                    node.confined = False
                    if len(node.loc)==1:
                        ax.scatter(node.loc[0],0., s=5., color='b', marker='x')
                    elif len(node.loc)==2:
                        ax.scatter(node.loc[0],node.loc[1], s=5., color='b', marker='x')
                else: # if the simulation went out of bounds we don't even need to check the convergence
                    # Check traj convergence
                    converged = self.convergence_check(node)
                    if not converged:
                        node.converged = False
                        if len(node.loc)==1:
                            ax.scatter(node.loc[0],0., s=5., color='purple', marker='x')
                        elif len(node.loc)==2:
                            ax.scatter(node.loc[0],node.loc[1], s=5., color='purple', marker='x')

    def consistency_analysis(self,ax):
        all_consistent = True
        for n,node in enumerate(self.nodes):
            if isinstance(node, Node) and not node.extreme_kappa:
                if node.sane:
                    # Check traj consistency by considering the expected PDF versus the sampled PDF
                    # When the biased FES contains multiple minima, separated by unsurmountable barriers,
                    # and the trajectory only samples a single minimum, the kappa value should be increased
                    # until only a single minimum remains
                    consistent = self.consistency_check(node)
                    if not consistent:
                        all_consistent = False
                        print(node.identity, ': this node trajectory was not consistent')
                        node.consistent = False
                        if len(node.loc)==1:
                            ax.scatter(node.loc[0],0., s=5., color='pink', marker='x')
                        elif len(node.loc)==2:
                            ax.scatter(node.loc[0],node.loc[1], s=5., color='pink', marker='x')

        return all_consistent

    def overlap_analysis(self,ax):
        self.overlap = np.ones((len(self.nodes),2*len(self.steps)), dtype=bool) # default everything overlaps

        # Check overlap between point and neighbours in idx to find points for which finer grid is required
        for n,node in enumerate(self.nodes):
            if isinstance(node, Node) and not node.extreme_kappa:
                # Check overlap for all Nodes (not reference nodes)
                # Plot overlap grid to see whether nearest neighbour assignment works
                for m,neighbour in enumerate(node.neighbours):
                     if len(node.loc)==1:
                         ax.plot((node.loc[0],self.nodes[neighbour].loc[0]), (0,0) , linestyle=':' ,color='gray', lw=0.3)
                     elif len(node.loc)==2:
                         ax.plot((node.loc[0],self.nodes[neighbour].loc[0]),
                                 (node.loc[1],self.nodes[neighbour].loc[1]), linestyle=':' ,color='gray', lw=0.3)
                 # Only consider overlap if this node is not deviant (else overlap loses meaning)
                if node.sane:
                    self.overlap[n,:len(node.neighbours)] = False # initialize all neighbouring overlap values to be faulty
                    for m,neighbour in enumerate(node.neighbours):
                        # Make a distinction between Nodes and BenchmarkNodes
                        # - If the node is another node (or a realized benchmark node)
                        #       * check if sane
                        # - If the node is a superlayer benchmark node (by definition sane):
                        #       * no checks required
                        # - If the node is a virtual benchmark node (by definition sane):
                        #       * if faulty overlap replace by RealizedBenchmarkNode (perform a simulation at that location instead)

                        if isinstance(self.nodes[neighbour],Node) and self.nodes[neighbour].sane:
                            # first check if the overlap between these two nodes has already been checked
                            if n > neighbour:
                                self.overlap[n,m] = self.overlap[neighbour,self.nodes[neighbour].neighbours.index(n)]
                            else:
                                self.overlap[n,m] = self.overlap_check(node, self.nodes[neighbour])>=self.grid.OVERLAP_THR

                        elif isinstance(self.nodes[neighbour],RealizedBenchmarkNode) and self.nodes[neighbour].sane:
                            self.overlap[n,m] = self.overlap_check(node, self.nodes[neighbour])>=self.grid.OVERLAP_THR
                        
                        elif isinstance(self.nodes[neighbour],SuperlayerBenchmarkNode):
                            self.overlap[n,m] = self.overlap_check(node, self.nodes[neighbour])>=self.grid.OVERLAP_THR

                        elif isinstance(self.nodes[neighbour],VirtualBenchmarkNode):
                            if not self.nodes[neighbour].sane:
                                # it might already be visited by another node
                                raise AssertionError('This cannot happen. It should have been replaced')
                                continue 
                            overlap = self.overlap_check(node, self.nodes[neighbour])>=self.grid.OVERLAP_THR
                            if not overlap:
                                self.nodes[neighbour] = self.nodes[neighbour].realize() # this will result in the generation of a RealizedBenchmarkNode at this location
                                if len(node.loc)==1:
                                    ax.plot((node.loc[0],self.nodes[neighbour].loc[0]), (0,0) , color='orange', lw=0.3)
                                elif len(node.loc)==2:
                                    ax.plot((node.loc[0],self.nodes[neighbour].loc[0]),
                                            (node.loc[1],self.nodes[neighbour].loc[1]) , color='orange', lw=0.3)
                            self.overlap[n,m] = True

                        else:
                            # the reference node is deviant, so we should not be refining here
                            assert not self.nodes[neighbour].sane
                            self.overlap[n,m] = True

                        if not self.overlap[n,m]:
                            if len(node.loc)==1:
                                ax.plot((node.loc[0],self.nodes[neighbour].loc[0]), (0,0) , color='r', lw=0.3)
                            elif len(node.loc)==2:
                                ax.plot((node.loc[0],self.nodes[neighbour].loc[0]),
                                        (node.loc[1],self.nodes[neighbour].loc[1]), color='r', lw=0.3)

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),frameon=False)
        ax.get_figure().savefig('overlap_{}.pdf'.format(self.index),bbox_inches='tight')


    def refine_layer(self):
        """
            Check overlap for layer.
            The plot functionality only works for ND grids with N < 3
        """
        # Assign al nearest neighbours
        self.characterize_nodes()

        # Set up figures
        if len(self.steps) == 1:
            fig = pt.figure('check_overlap',figsize=((self.index+1)*4,2))
            pt.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,        # ticks along the left edge are off
            right=False,       # ticks along the right edge are off
            labelleft=False)  # labels along the left edge are off
        elif len(self.steps) == 2:
            fig = pt.figure('check_overlap',figsize=((self.index+1)*2,(self.index+1)*2))
        else:
            fig = pt.figure('check_overlap')
            print("Plotting the overlap data is not implemented for more than 2 dimension!")
        ax = fig.gca()

        # Plot all previous grids as a reference for this grid
        grid_utils.plot_ref_grid(ax,self,size=1.)
        pt.close('check_overlap') # make sure this does not get plotted somewhere

        # Sanity analysis
        self.sanity_analysis(ax)

        if self.grid.CONSISTENCY_THR is not None:
            # Consistency analysis
            consistent = self.consistency_analysis(ax)

            # If there are any inconsistencies the refinement should stop here
            if not consistent: return

        # Overlap analysis
        self.overlap_analysis(ax)
        
        return

    def generate_sublayer(self):
        """
            Make sublayer based on overlap heuristic
            This also identifies a list of benchmark points required for this grid
        """
        assert self.overlap is not None
        if np.sum(1-self.overlap)==0:
            # if there are no overlap issues we can immediately stop
            return 

        # Create a grid of new nodes and benchmark nodes for overlap refinement
        # Create set of unique node pairs where overlap is faulty
        overlap_pairs = np.array(np.where(self.overlap==0)).T  # with transpose the shape is (N,2)

        # The first index of the overlap pair wil always be a Node, never a BenchmarkNode
        # Convert node and neighbour number to node numbers
        overlap_pairs = list(set([tuple(sorted([pair[0],self.nodes[pair[0]].neighbours[pair[1]]])) for pair in overlap_pairs]))

        dims = (2*(len(self.steps)-1),                       # half step in each other direction
                2*(len(self.steps)-1) * 2*len(self.steps))   # half step in each direction for every point in fine grid except halfway point

        m_steps = np.diag(self.steps)

        # Construct sublayer using half the previous step size in the direction of the faulty overlap
        kappas            = np.zeros((len(overlap_pairs), 1+dims[0], len(self.steps)))
        sublayer          = np.zeros((len(overlap_pairs), 1+dims[0], len(self.steps)))
        if dims[0]>0:
            virtual_benchmark_nodes  = np.zeros((len(overlap_pairs), dims[1], len(self.steps))) # benchmark nodes that do not correspond to superlayer nodes
        superlayer_benchmark_nodes   = np.zeros((len(overlap_pairs),       2, len(self.steps))) # the superlayer nodes between which the overlap was faulty

        for n,(i,j) in enumerate(overlap_pairs):
            # If the second node is a benchmark node, take the kappa value of the node, else take the maximum
            if isinstance(self.nodes[j],BenchmarkNode):
                kappas[n,:] = self.nodes[i].kappas
            else:
                kappas[n,:] = np.max(np.array([self.nodes[i].kappas, self.nodes[j].kappas]),axis=0)

            # Determine direction of faulty overlap
            diff = np.abs(self.nodes[i].loc - self.nodes[j].loc)
            od = np.argmax(diff)
            reduced_m_steps = np.delete(m_steps,od,0)
            try:
                assert np.sum(np.hstack((diff[:od],diff[od+1:])))<precision
            except AssertionError:
                raise AssertionError('A problem occured when trying to define the overlap direction.')

            # Construct sublayer
            sublayer[n,0] = 0.5*(self.nodes[i].loc + self.nodes[j].loc) # halfway point
            if dims[0]>0:
                sublayer[n,1:] = sublayer[n,0] + np.array([k*vec for k in [-0.5,0.5] for vec in reduced_m_steps]) # halfway step in every other direction

            # Construct superlayer_benchmark_nodes
            superlayer_benchmark_nodes[n][0] = self.nodes[i].loc
            superlayer_benchmark_nodes[n][1] = self.nodes[j].loc

            # Construct virtual_benchmark_nodes
            if dims[1]>0:
                virtual_benchmark_nodes[n]  = np.array([fn + k*vec for fn in sublayer[n,1:] for k in (-0.5,0.5) for vec in m_steps])

        kappas = kappas.reshape((-1,len(self.steps)))
        sublayer = sublayer.reshape((-1,len(self.steps)))
        if dims[0]>0:
            virtual_benchmark_nodes = virtual_benchmark_nodes.reshape((-1,len(self.steps)))
        superlayer_benchmark_nodes = superlayer_benchmark_nodes.reshape((-1,len(self.steps)))


        # Remove duplicate sublayer points, and store the corresponding kappa values
        mask = make_unique(self.steps/2.,sublayer.reshape((-1,len(self.steps))),get_mask=True)
        sublayer = np.round(sublayer[mask],DECIMALS)
        kappas = kappas[mask]

        # Remove duplicate benchmark points
        if dims[1]>0:
            virtual_benchmark_nodes  = np.round(make_unique(self.steps/2.,virtual_benchmark_nodes),DECIMALS)
        superlayer_benchmark_nodes = np.round(make_unique(self.steps/2.,superlayer_benchmark_nodes),DECIMALS)

        # Remove benchmark points that correspond to sublayer points constructed above
        if dims[1]>0:
            virtual_benchmark_nodes  = make_unique_between(self.steps/2.,sublayer,virtual_benchmark_nodes)
        superlayer_benchmark_nodes = make_unique_between(self.steps/2.,sublayer,superlayer_benchmark_nodes) 


        # Sort all the grids according to cvs
        ind = np.lexsort(sublayer[:,::-1].T) # Sort by first column, then second column, ...
        sublayer = sublayer[ind]

        if dims[1]>0:
            ind = np.lexsort(virtual_benchmark_nodes[:,::-1].T) # Sort by first column, then second column, ...
            virtual_benchmark_nodes = virtual_benchmark_nodes[ind]

        ind = np.lexsort(superlayer_benchmark_nodes[:,::-1].T) # Sort by first column, then second column, ...
        superlayer_benchmark_nodes = superlayer_benchmark_nodes[ind]

        # Convert to Nodes
        # Convert sublayer points into NewNodes and add them to the sublayer nodes
        self.sublayer_nodes += [NewNode(point,kappas[n]) for n,point in enumerate(sublayer)]

        # Convert benchmark points to benchmark nodes
        tree = KDTree(np.array([node.loc for node in self.nodes])) # self.nodes should be self contained, if a benchmarknode was realized it should have been replaced in this array

        # Add benchmark nodes to the sublayer nodes
        for node in superlayer_benchmark_nodes:
            ds,idx = tree.query(node,1) # find this point in grid
            if ds<10*precision:
                # The node has to be sane, if it is not sane, we should wait to add it as a benchmark node
                # since it does not make sense to test the overlap with
                # It will be refined by the next iteration and added accordingly
                if self.nodes[idx].sane:
                    # Create a SuperlayerBenchmarkNode
                    sln = SuperlayerBenchmarkNode(self, self.nodes[idx])

                    # If the SuperlayerBenchmarkNode is sane in the sublayer add it, else create a RealizedBenchmarkNode at that location with increased kappa
                    if sln.sane:
                        self.sublayer_nodes.append(sln)
                    else:
                        increased_kappas = np.clip(sln.kappas*self.grid.KAPPA_GROWTH_FACTOR,0,self.grid.MAX_KAPPA)
                        self.sublayer_nodes.append(RealizedBenchmarkNode(NewNode(sln.loc,increased_kappas)))

        if dims[1]>0:
            for node in virtual_benchmark_nodes:
                ds,idx = tree.query(node,2) # find two closest neighbours
                neighbour_nodes = idx[(ds-ds[0])<precision] # check if they both fall inside the grid precision

                # Check whether the neighbouring nodes are sane, if not, it does not make sense to create a reference yet
                # It will be generated at a later instance
                if all([self.nodes[nnode].sane for nnode in neighbour_nodes]) and len(neighbour_nodes)>0::
                    # Create a VirtualBenchmarkNode
                    vn = VirtualBenchmarkNode(node, [self.nodes[nnode] for nnode in neighbour_nodes])
                    self.sublayer_nodes.append(vn)
