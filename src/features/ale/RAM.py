from ALE import ALE
import numpy as np

#RAM features, see Najaf,2010      
class RAM(ALE):
    def __init__(self):
        self.num_bits = 1024 #bits in ramvector
        self._num_feat = self.num_bits + (self.num_bits**2 +self.num_bits)//2

    def phi(self, state):
        return super(RAM, self).phi(state)
        
    def phi_idx(self, state):
        state = self.ram_data(state)
        #represent 128 byte ram vector as 1024 bit vector
        s_bits = np.unpackbits(state)
        r_idx = np.nonzero(s_bits)[0]
        p_idx = self.pairwise(r_idx,s_bits.size)
        return np.r_[r_idx,(s_bits.size+p_idx)].astype('uint')