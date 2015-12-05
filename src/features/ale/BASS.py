from Basic import Basic
import numpy as np

# BASS features, see Najaf,2010     
class BASS(Basic):
    def __init__(self,im_size=np.array([210,160]),
                 num_tiles=np.array([21,16]),
                 background_file='background.pkl'):
        #use secam colors to avoid explosion of pairwise features
        super(BASS,self).__init__(im_size=im_size,
                                            num_tiles= num_tiles,
                                            background_file=background_file, 
                                            secam = True)
    
    def num_features(self):
        #add # pairwise features
        return self._num_feat+(self._num_feat**2+self._num_feat)//2
        
    def phi_idx(self, state):
        #these are basic features
        idx= super(BASS,self).ph_idx(state)
        #now add paiwise feature combinations
        p_idx = self.pairwise(idx,self._num_feat)
        return np.r_[idx,(self._num_feat+p_idx)]