from Projector import Projector
import numpy as np

class StateToStateAction(Projector):
    num_actions = 1
    state_projector = None
    
    def __init__(self,projector,num_actions=1):
        self.state_projector = projector
        self.num_actions = num_actions

        
    def num_features(self):
        return self.num_actions*self.state_projector.num_features()
        
    def to_state_action(self,phi,a,idx=False):
        if idx:
            phi_a = phi + (a*phi.size)
        else:
            phi_a = np.zeros(phi.size*self.num_actions)
            phi_a[(a*phi.size):((a+1)*phi.size)]=phi
        return phi_a
        
    def phi_idx(self,state):
        return self.state_projector.phi_idx(state)
        
    def phi(self,state):
        return self.state_projector.phi(state)
        
    def project(self,state):
        return self.state_projector.project(state)
        
    def supports_sparse(self):
        return self.state_projector.supports_sparse()
    
        
    def phi_sa(self,state,action):
        phi = self.state_projector.phi(state)
        return self.to_state_action(phi,action)
        
    def project_sa(self,state,a):
        phi = self.phi(state,a)
        if self.normalized:
            return phi / np.sum(phi)
        else:
            return phi