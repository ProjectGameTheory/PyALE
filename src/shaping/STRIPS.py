from Shaper import Shaper


class STRIPS(Shaper):

    def __init__(self, strips_plan=None, convert=None, **kwargs):
        super(STRIPS, self).__init__(**kwargs)
        self.strips_plan = strips_plan
        # function that takes a state
        # and convert it to a list of strips GroundedPositions
        self.convert = convert
        self.current_step = 0

    def start(self):
        self.current_step = 0

    def to_strips(self, state):
        return self.convert(state)

    def match_state(self, state):
        strips_state = self.to_strips(state)
        try:
            i = self.strips_plan.index(strips_state)
            self.current_step = i
        except:
            pass
        return self.current_step

    def match_post(self, po1, po2):
        # po1 and po2 sets of strings
        return po1 == po2

    def potential(self, state):
        return self.omega*self.match_state(state)