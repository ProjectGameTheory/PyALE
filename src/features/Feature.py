
class Feature(object):
    def __init__(self, state_length=0):
        self.state_length = state_length

    def num_features(self):
        return self.state_length

    def normalize_state(self,state):
        return state
        
    def phi(self, state):
        return state

    def phi_idx(self, state):
        pass

    def supports_sparse(self):
        return False