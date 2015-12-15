from STRIPS import STRIPS
from FlagShaper import FlagShaper

class CombinedShaper(STRIPS, FlagShaper):

    # For the argument passing in Multiple Inheritance to work
    # all args need to be in **kwargs
    # CombinedShaper takes the following args:
    #  strips_plan, convert, num_flags, omega
    # super(init) goes from left to right, so STRIPS, FlagShaper, Shaper
    def __init__(self, **kwargs):
        super(CombinedShaper,self).__init__(**kwargs)

    def start(self):
        self.flags_collected = 0
        self.current_step = 0

    def potential(self, state):
        self.count_flags(state)
        return (self.flags_collected + self.match_state(state)) * self.omega