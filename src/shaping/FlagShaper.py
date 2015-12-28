from Shaper import Shaper
import numpy as np

class FlagShaper(Shaper):

    def __init__(self, num_flags, grid_length=234, **kwargs):
        super(FlagShaper,self).__init__(**kwargs)
        self.num_flags = num_flags
        self.grid_length = grid_length
        self.flags_collected = 0

    def start(self):
        self.flags_collected = 0

    def count_flags(self, state):
        agent_idx = np.where(state == 1)[0][0]
        flag_idx = agent_idx / self.grid_length
        # convert flag_idx in flags taken
        taken = bin(flag_idx)[2:]
        total = 0
        for i in range(len(taken)):
            if taken[i] == '1':
                total += 1
        self.flags_collected = total

    def potential(self, state):
        self.count_flags(state)
        return 100 * self.flags_collected
