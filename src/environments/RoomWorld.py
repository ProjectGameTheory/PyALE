from MultiEnvironment import MultiEnvironment
import numpy as np
import random


class RoomWorld(MultiEnvironment):

    def __init__(self, grid, population_size=0, ids=None, begins=None, goal=[0, 0], *args, **kwargs):
        self.grid = grid
        self.size = np.shape(grid)
        # register the names of all flags
        self.flags = []
        for row in self.grid:
            for pos in row:
                if pos[1]:
                    self.flags.append(pos[1])
        super(RoomWorld, self).__init__(*args, **kwargs)
        # Number of agents in this environment
        self.population_size = population_size
        # Begin positions of agents
        self.current_states = dict(zip(ids, begins)) if len(
            ids) == len(begins) else {}
        # Goal position of all agents
        self.goal = goal
        if goal is None:
            self.goal = self.random_position()
        # 4 possible moves
        self.directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        # Flags to collect
        self.flags_taken = dict()
        for agent_id in ids:
            self.flags_taken[agent_id] = []

    def random_position(self):
        return np.array([random.randint(0, self.size[0]), random.randint(0, self.size[1])])

    def get_cell(self, position):
        return self.grid[position[0]][position[1]]

    def vertical_wall(self, position):
        walls = self.get_cell(position)[0]
        return int(walls) & 1

    def horizontal_wall(self, position):
        walls = self.get_cell(position)[0]
        return int(walls) & 2

    def move(self, position, direction):
        # Direction 0=up, 1=right, 2=down, 3=left
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

    # state of an agent:
    # 1: his position encoded in binary
    # 2: others position encoded in binary
    # 3: flags left in room (for each flag: 0 for taken, 1 for available)
    # 4: flags the agent has (for each flag: 1 for taken, 0 otherwise)
    # agent needs to know both the flags he has and the flags left on board
    # as others could also possess flags
    # (but it doesn't matter to the agent which other agent has a specific flag
    # , it's still unavailable)
    def to_binary(self, id):
        # max number of bits needed to encode position
        max_length = len(bin(self.size[0]*self.size[1] - 1)[2:])
        bin_repr = []
        for key, value in self.current_states.items():
            value = value[0] + self.size[0] * value[1]
            position = [int(x) for x in bin(value)[2:]]
            # nb of bits needs to be the same for every position
            while len(position) < max_length:
                position.insert(0, 0)
            # put the agent's position first
            if key == id:
                position.extend(bin_repr)
                bin_repr = position
            else:
                bin_repr.extend(position)
        available_flags = [1 for flag in self.flags]
        taken_flags = [0 for flag in self.flags]
        for key, taken in self.flags_taken.items():
            for flag in taken:
                flag_idx = self.flags.index(flag)
                available_flags[flag_idx] = 0
                if key == id:
                    taken_flags[flag_idx] = 1
        bin_repr.extend(available_flags)
        bin_repr.extend(taken_flags)
        return bin_repr

    def start(self, id):
        return self.to_binary(id)

    def step(self, id, action):
        position = self.move(self.current_states[id], action)
        if self.at_flag(position):
            if not self.flag_taken(position):
                # take_flag modifies self.flags_taken
                self.take_flag(id, position)
        self.current_states[id] = position
        reward = self.reward(id, position)
        terminal = self.terminal(position)
        return self.to_binary(id), reward, terminal
