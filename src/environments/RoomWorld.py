from MultiEnvironment import MultiEnvironment
import numpy as np
import random

class RoomWorld(MultiEnvironment):
	def __init__(self, grid, population_size=0, ids=None, begins=None, goal=[0,0], *args, **kwargs):
		self.grid = np.array(grid)
		self.size = np.shape(self.grid)
		super(RoomWorld, self).__init__(*args, **kwargs)
		# Number of agents in this environment
		self.population_size = population_size
		# Begin positions of agents
		self.begins = dict(zip(ids,begins)) if len(ids) == len(begins) else {}
		# Goal position of all agents
		self.goal = goal
		if goal is None:
			self.goal = self.random_position()
		# 4 possible moves
		self.directions = np.array([[1,0], [0,1], [-1,0], [0, -1]])
		# Flags to collect
		self.flags_taken = dict()
		for agent_id in ids:
			self.flags_taken[agent_id] = []
		return

	def random_position(self):
		return np.array([random.randint(0, self.size[0]), random.randint(0, self.size[1])])

	def get_cell(self,position):
		return self.grid[position[0]][position[1]]

	def vertical_wall(self, position):
		walls = self.get_cell(position)[0]
    		return int(walls) & 1

	def horizontal_wall(self, position):
		walls = self.get_cell(position)[0]
    		return int(walls) & 2

	def move(self, position, direction):
		#Direction 0=up, 1=right, 2=down, 3=left
		new_position = position + self.directions[direction]
		if direction == 0 and (new_position[0] >= self.size[0] or self.horizontal_wall(new_position)):
			return position
		elif direction == 2 and (new_position[0] < 0 or self.horizontal_wall(position)):
			return position
		elif direction == 1 and (new_position[1] >= self.size[1] or self.vertical_wall(new_position)):
			return position
		elif direction == 3 and (new_position[1] < 0 or self.vertical_wall(new_position)):
			return position
		else:
			return new_position

	def terminal(self, position):
		return np.array_equal(position, self.goal)

	def at_flag(self, position):
		return self.get_cell(position)[1]

	def flag_taken(self, position):
		flag = self.get_cell(position)[1]
		return (flag and any(flag in agent_flags for agent_flags in self.flags_taken.values()))

	def take_flag(self, id, position):
		flag = self.get_cell(position)[1]
		if flag:
			self.flags_taken[id].append(flag)

	def reward(self, id, position):
		reward = 0
		if self.terminal(position):
			flags_collected = len(self.flags_taken[id])
			reward = 100 * flags_collected
		return reward

	def to_binary(self, position):
		state = np.zeros(self.size[0]*self.size[1])
		state[position[0]+position[1]*self.size[0]] = 1
		return state

	def start(self, id):
		self.current_states[id] = self.begins[id]
		return self.to_binary(self.begins[id])

	def step(self, id, action):
		position = self.move(self.current_states[id], action)
		if self.at_flag(position):
			if not self.flag_taken(position):
				self.take_flag(id, position) #take_flag modifies self.flags_taken
		self.current_states[id] = position
		reward = self.reward(id, position)
		terminal = self.terminal(position)
		return self.to_binary(position), reward, terminal

