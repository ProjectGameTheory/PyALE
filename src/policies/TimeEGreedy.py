from EGreedy import EGreedy
from TimePolicy import TimePolicy

class TimeEGreedy(TimePolicy, EGreedy):
    def __init__(self, formula=None):
        TimePolicy.__init__(self, formula=formula)

    def select_action(self, actions=[0,1], values=[2,3], time=0):
        self.epsilon = self.formula(time)
        return EGreedy.select_action(self, actions=actions, values=values)