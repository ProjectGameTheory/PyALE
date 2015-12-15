from Shaper import Shaper

class FlagShaper(Shaper):

	def __init__(self, num_flags, **kwargs):
		super(FlagShaper,self).__init__(**kwargs)
		self.num_flags = num_flags
		self.flags_collected = 0

	def start(self):
		self.flags_collected = 0

	def count_flags(self, state):
		flags = state[-self.num_flags:]
		self.flags_collected = sum(flags == 1)

	def potential(self, state):
		self.count_flags(state)
		return 100 * self.flags_collected
