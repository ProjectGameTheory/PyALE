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

    def to_strips(self, id, state):
        return self.convert(id, state)

    def match_state(self, id, state):
        strips_state = self.to_strips(id, state)
        if self.match_post(strips_state, self.strips_plan[self.current_step]):
            self.current_step += 1
        return self.current_step

    def match_post(self, po1, po2):
        # po1 and po2 sets of strings
        return po1 == po2

    def potential(self, id, state):
        return self.omega*self.match_state(id, state)