from Policy import Policy

class TimePolicy(Policy):
    def __init__(self, formula=None):
        self.formula = formula
        if self.formula is None:
            self.formula = lambda step: 0

    def select_action(self, actions=[], values=[], step=0):
        raise NotImplementedError()