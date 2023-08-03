#! /usr/bin/python
import numpy as np

######### AUXILARY FUNCTIONS #########
def parse_identity(identity):
    return tuple([int(i.decode()) if isinstance(i,bytes) else int(i) for i in identity])

########## NODE DEFINITIONS ##########

class Node(object):
    def __init__(self, loc, traj, kappas, identity):
        self.loc = loc
        self.trajs = [traj]
        self.kappas = kappas
        self.identity = identity
        self.neighbours = []
        self.extreme_kappa = False
        self.converged = True
        self.confined = True
        self.consistent = True
        self._type = None

    def set_extreme_kappa(self,MAX_KAPPA,KAPPA_GROWTH_FACTOR):
        if np.any(MAX_KAPPA<self.kappas*KAPPA_GROWTH_FACTOR):
            self.extreme_kappa = True

    @property
    def sane(self):
        return self.confined and self.converged and self.consistent

    @property
    def type(self):
        """if not self.converged:
            self._type = 'non_converged_node'
        else:"""
        self._type = 'node' if self.sane else 'deviant_node'
        return self._type

    def node_info(self,identity=None,grid=None):
        if self.confined and self.converged and self.consistent:
            kappas = self.kappas
        else:
            kappas = np.clip(self.kappas*grid.KAPPA_GROWTH_FACTOR,0,grid.MAX_KAPPA)

        return np.array([self.identity[0], self.identity[1],
                    "*".join(['{:.8f}'.format(p) for p in self.loc]).encode(),
                    "*".join(['{:.8e}'.format(kappa) for kappa in kappas]).encode(),
                    self.type.encode()
                    ],dtype=object)


class NewNode(Node):
    def __init__(self, loc, kappas):
        super(NewNode,self).__init__(loc, None, kappas, None) # identity is None
        self.trajs = [] # overwrite trajs to be empty
        
    @property
    def type(self):
        self._type = 'new_node'
        return self._type

    def node_info(self,identity=None,grid=None):
        identity = parse_identity(identity)
        return np.array([identity[0], identity[1],
                        "*".join(['{:.8f}'.format(p) for p in self.loc]).encode(),
                        "*".join(['{:.8e}'.format(kappa) for kappa in self.kappas]).encode(),
                        self.type.encode()
                        ],dtype=object)



class BenchmarkNode(object):
    def __init__(self, loc, nodes):
        self.loc = loc
        self.nodes = nodes
        self.trajs = [traj for node in self.nodes for traj in node.trajs]
        self.kappas = np.max([node.kappas for node in self.nodes],axis=0)
        self._sane = None 
        self._type = None

    @property
    def type(self):
        # this will be implemented in subclasses
        raise NotImplementedError

    @property
    def sane(self):
        # this will be implemented in subclasses
        raise NotImplementedError

    def node_info(self,identity=None,grid=None):
        # this will be implemented in subclasses
        raise NotImplementedError
    
class SuperlayerBenchmarkNode(BenchmarkNode):
    def __init__(self,layer,node):
        # This will initialize the benchmark node based on a node from the upper lying layer
        assert node.sane
        super(SuperlayerBenchmarkNode,self).__init__(node.loc,[node])
        self._sane = layer.sanity_check(self,factor=2.,plot=False)

    @property
    def type(self):
        if self.sane:
            self._type = 'superlayer_benchmark_node'
        else:
            self._type = 'deviant_superlayer_benchmark_node' # this should never occur, it will be replaced by a Realized Benchmark Node instance
        return self._type

    @property
    def sane(self):
        return self._sane

    @property
    def identity(self):
        return self.nodes[0].identity

    def node_info(self,identity=None,grid=None):
        raise ValueError('This function should never be called on this object')


class VirtualBenchmarkNode(BenchmarkNode):
    def __init__(self,loc,nodes):
        # This will initialize the benchmark node based on a node from the upper lying layer
        assert all([node.sane for node in nodes])
        super(VirtualBenchmarkNode,self).__init__(loc,nodes)
    
    @property
    def type(self):
        if self.sane:
            self._type = 'virtual_benchmark_node'
        else:
            self._type = 'deviant_virtual_benchmark_node' # this should never occur, it will be replaced by a Realized Benchmark Node instance
        return self._type

    @property
    def sane(self):
        return True

    def realize(self):
        return RealizedBenchmarkNode(NewNode(self.loc,self.kappas))

    def node_info(self,identity=None,grid=None):
        raise ValueError('This function should never be called on this object')
        
class RealizedBenchmarkNode(BenchmarkNode):
    def __init__(self, node):
        # This will act as a mask on top of a newly generated node and parse the attributes as a benchmark node
        assert isinstance(node,NewNode) or isinstance(node,Node)
        self.node = node # for later reference
        super(RealizedBenchmarkNode,self).__init__(self.node.loc,[self.node])
        self.identity = node.identity
        self._type = None

    def set_extreme_kappa(self,MAX_KAPPA,KAPPA_GROWTH_FACTOR):
        if np.any(MAX_KAPPA<self.kappas*KAPPA_GROWTH_FACTOR):
            self.extreme_kappa = True
            self.node.set_extreme_kappa(MAX_KAPPA,KAPPA_GROWTH_FACTOR)

    @property
    def type(self):
        self._type = 'realized_benchmark_node' if self.sane else 'deviant_realized_benchmark_node'
        # Overwrite if the underlying node is not converged
        #if isinstance(self.node,Node):
        #    if not self.node.converged:
        #        self._type = 'non_converged_realized_benchmark_node'
        return self._type

    @property
    def sane(self):
        if isinstance(self.node,NewNode):
            self._sane = False # by default a NewNode requires a new simulation
        else:
            self._sane = self.node.sane
        return self._sane

    
    def node_info(self,identity=None,grid=None):
        reference_node_info = self.node.node_info(identity=identity,grid=grid)

        # Update type to benchmark type
        reference_node_info[-1] = self.type.encode()
        return reference_node_info