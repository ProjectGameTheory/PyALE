from features.Feature import Feature
import numpy as np

class ALE(Feature):
    _num_feat = 0
    
    #number of features
    def num_features(self):
        return self._num_feat
        
    #normalize state values if need
    def normalize_state(self,state):
        return state
        
    #sparse representation possible (e.g. tilecoding)
    def supports_sparse(self):
        return True
    
    #return full feature vector
    def phi(self,s):
        res = np.zeros(self.num_features())
        res[self.phi_idx(s)] = 1.
        return res
        
    #return non-zero feature vector
    def phi_idx(self,s):
        return NotImplementedError()

    def frame_data(self, state):
        return state[128:33728]

    def ram_data(self, state):
        return np.array(state[:128], dtype=np.ubyte)
        
    '''
    pairwise features for binary vectors. Calculates and of all
    combinations of bits in binary vector.
    Inputs: 
        r_idx: indices of nonzero bits (i.e. sparse representation)
                assumed to be sorted in ascending order
        num_bits: length of binary vector
    Returns:
        pairwise combination of features, i.e. indices of all nonzero bits
        when we take the AND of all combinations of bits in the vector
        (only consides unique pairs i.e. it only includes  
        AND(bit0,bit1)) and not AND(bit1,bit0) as well)
    '''
    
    def pairwise(self,r_idx,num_bits):
            #assumes r_idx is sorted in ascending order        
            #get pairwise nonzero indices 
            #nr of  combos for n 1bits: (n**2+n)/2) 
            # assuming we include and of bit with itself
            p_idx = np.zeros((r_idx.size**2+r_idx.size)//2,dtype='uint')
            # pairwise vector is indexed as follows:
            #first n bits are AND of bit 0 with all bits
            #next n-1 bits are AND of bit 1 with bits 1:n
            #next n-2 bits are AND of bit 2 with bits 2:n ...
            #so the offset for
            #AND of bit k with bits k:n is sum i:0 to k-1 (n-i)
            # = k*n - k(k+1)/2
            offs = r_idx*num_bits - 0.5*r_idx*(r_idx+1)
            idx = 0
            for i in range(r_idx.size):
                p_idx[idx:(idx+r_idx.size-i)] = offs[i] +r_idx[i:]-r_idx[i]
                idx = idx+r_idx.size-i
            return p_idx

    '''!use vectorization, iteration over pairs is very slow (order of magn):
        for i,ind1 in enumerate(r_idx):
            for ind2 in r_idx[i:]:
                offset = ind1*num_bits - 0.5*ind1*(ind1+1)
                p_idx[j] = (offset+ind2-ind1)
                j+=1   
        return p_idx
    '''