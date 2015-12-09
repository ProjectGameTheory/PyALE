from GridWorld import GridWorld
import numpy as np
import random

class RoomWorld(MultiEnvironment):
	def __init__(self, size=[10,10], population_size=0, begins=None, goal=[0,0], flags=None, *args, **kwargs):
		super(RoomWorld, self).__init__(*args, **kwargs)
		# Number of agents in this environment
		self.population_size = population_size
		# Begin positions of agents
		self.temp_begins = begins if len(begins) == population_size else None
		if begins is None:
			self.temp_begins = [self.random_position() for x in range(self.population_size)]
		self.begins = {}
		# Goal position of all agents
		self.goal = goal
		if goal is None:
			self.goal = self.random_position()
		# 4 possible moves
        self.directions = np.array([[0,1], [1,0], [0,-1], [-1, 0]])
        # Flags to collect
		self.flags = np.array(flags) # flags must be a list of coordinates: [[X1,Y1], [X2,Y2], ...]
		self.flags_taken = np.zeros(len(self.flags)) # 0=flag not taken, ID=flag taken by agent with ID
		# Walls are hard-coded
		self.horizontal_walls = np.array([[0,5,0,6], [2,5,2,6],   [3,5,3,6],   [4,5,4,6], [5,5,5,6],
			                              [0,8,0,9], [1,8,1,9],   [2,8,2,9],   [3,8,3,9], [5,8,5,9],
			                              [6,8,6,9], [7,8,7,9],   [8,8,8,9],   [6,3,6,4], [8,3,8,4], 
			                              [9,3,9,4], [10,3,10,4], [11,3,11,4], [12,3,12,4]])
		self.vertical_walls = np.array([[5,0,6,0],   [5,1,6,1],     [5,2,6,2],     [5,3,6,3],
			                            [5,4,6,4],   [5,5,6,5],     [5,6,6,6],     [5,8,6,8],
			                            [5,9,6,9],   [5,10,6,10],   [5,11,6,11],   [5,12,6,12],
			                            [8,4,9,4],   [8,5,9,5],     [8,6,9,6],     [8,8,9,8],
			                            [12,0,13,0], [12,2,13,2],   [12,3,13,3],   [12,4,13,4],
			                            [12,5,13,5], [12,6,13,6],   [12,7,13,7],   [12,8,13,8],
			                            [12,9,13,9], [12,10,13,10], [12,11,13,11], [12,12,13,12]])

	def random_position(self):
        return [random.randint(0, self.size[0]), random.randint(0, self.size[1])]

	def move(self, position, direction):
		#Direction 0=up, 1=right, 2=down, 3=left
		coordinates = {0: [[x,y] for x, y in zip(self.horizontal_walls[:,0], self.horizontal_walls[:,1])],
		               1: [[x,y] for x, y in zip(self.vertical_walls[:,0],   self.vertical_walls[:,1])],
		               2: [[x,y] for x, y in zip(self.horizontal_walls[:,2], self.horizontal_walls[:,3])],
		               3: [[x,y] for x, y in zip(self.vertical_walls[:,2],   self.vertical_walls[:,3])]}.get(direction)
		if position in coordinates:
			return np.array(position)
		else:
			moved = np.array(position) + self.directions[direction]
        	return self.in_grid(moved)

     def in_grid(self, position):
        if position[0] < 0:
            position[0] = 0
        if position[0] >= self.size[0]:
            position[0] = self.size[0]-1
        if position[1] < 0:
            position[1] = 0
        if position[1] >= self.size[1]:
            position[1] = self.size[1]-1
        return position

    def terminal(self, position):
        return np.array_equal(position, self.goal)

    def at_flag(self, position):
    	return position in self.flags

    def flag_taken(self, position):
    	if position in self.flags:
    		pos = [i for i,x in enumerate(self.flags) if np.array_equal(x, position)]
    		return self.flags_taken[pos[0]]

	def take_flag(self, id, position):
		if position in self.flags:
			pos = [i for i,x in enumerate(self.flags) if np.array_equal(x, position)]
			self.flags_taken[pos[0]] = id

	def reward(self, id, position):
        reward = 0
        if self.terminal(position):
        	flags_collected = self.flags_taken.count(id)
            reward = 100 * flags_collected
        return reward

      def to_binary(self, position):
        state = np.zeros(self.size[0]*self.size[1])
        state[position[0]+position[1]*self.size[0]] = 1
        return state

   	def start(self, id):
   		self.begins[id] = self.temp_begins[0]
   		self.temp_begins = self.temp_begins[1:]
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

