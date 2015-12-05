from Environment import Environment
import numpy as np

class GridWorld(Environment):
    def __init__(self, size=[10,10], begin=None, goal=[0,0]):
        super(GridWorld, self).__init__()
        self.size = size
        # start position of agent
        self.begin = begin
        if begin is None:
            self.begin = self.random_position()
        #goal position of agent
        self.goal = goal
        if goal is None:
            self.goal = self.random_position()
        # 4 possible moves
        self.directions = np.array([[0,1], [1,0], [0,-1], [-1, 0]])

    def random_position(self):
        return [np.randint(0, self.size[0]), np.randint(0, self.size[1])]

    def move(self, position, direction):
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

    def reward(self, position):
        reward = -1
        if self.terminal(position):
            reward = 10
        return reward

    def terminal(self, position):
        return np.array_equal(position, self.goal)

    def to_binary(self, position):
        state = np.zeros(self.size[0]*self.size[1])
        state[position[0]+position[1]*self.size[0]] = 1
        return state
        #sparse state
        #return np.array([position[0]+position[1]*self.size[0]])

    def start(self):
        self.current_state = self.begin
        return self.to_binary(self.current_state)

    def step(self, action):
        position = self.move(self.current_state, action)
        self.current_state = position
        reward = self.reward(position)
        terminal = self.terminal(position)
        return self.to_binary(position), reward, terminal
