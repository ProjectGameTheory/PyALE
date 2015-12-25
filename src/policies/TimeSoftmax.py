from Softmax import Softmax
from TimePolicy import TimePolicy

class TimeSoftmax(TimePolicy, Softmax):
    def __init__(self, formula=None):
        TimePolicy.__init__(self, formula=formula)

    def select_action(self, actions=[0,1], values=[2,3], time=0):
        self.temperature = self.formula(time)
        return Softmax.select_action(self, actions=actions, values=values)