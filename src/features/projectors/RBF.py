from Projector import Projector
import numpy as np

class RBF(Projector):
    
    def __init__(self,num_centers=np.array([5]),
                 stdev=0.1,limits=np.array([[0,1],[0,1]]),
                 randomize=False,normalize = True,bias=True):
                     
        super(RBF,self).__init__(self,normalize,limits,bias) 
        if not (type(num_centers) is np.ndarray):
            num_centers = np.array(num_centers)
        if num_centers.size == 1:
            num_centers = np.ones(limits.shape[0])*num_centers[0]
        self.stdev = stdev
        dim = []
        if randomize:
            #randomly spaced centers
            for d in range(limits.shape[0]):
                dim.append(np.sort(np.random.rand(num_centers[d])))
        else: 
            #equally spaced centers
            for d in range(limits.shape[0]):
                dim.append(np.linspace(0,1,num_centers[d]))
        if len(dim) == 1:
            self.centers=dim[0].flatten()
        else:
            grid = np.meshgrid(*dim)
            self.centers=grid[0].flatten()
            for d in range(1,len(grid)):
                self.centers = np.c_[self.centers,grid[d].flatten()]

        
    def num_features(self):
        return (self.centers.shape[0]+int(self.bias))
        
    def phi(self,state):
        if len(self.centers.shape)==1:
            dists = self.centers-self.normalize_state(state)
        else:
            dists = np.linalg.norm(self.centers-
                self.normalize_state(state),axis=1)
        res = np.exp(-0.5 * dists**2 / self.stdev**2)
        
        return res