from MultiEnvironment import MultiEnvironment
import numpy as np
import random


class RoomWorld(MultiEnvironment):
    def __init__(self, grid, population_size=0, ids=None, begins=None, goal=[0, 0], *args, **kwargs):
        super(RoomWorld, self).__init__(*args, **kwargs)
        self.grid = grid
        self.size = np.shape(grid)
        # register the names of all flags
        self.flags = []
        for row in self.grid:
            for pos in row:
                if pos[1]:
                    self.flags.append(pos[1])
        # Number of agents in this environment
        self.population_size = population_size
        # Begin positions of agents
        self.begins = dict(zip(ids, begins)) if len(ids) == len(begins) else {}
        self.current_states = dict(self.begins)
        # Goal position of all agents
        self.goal = goal
        if goal is None:
            self.goal = self.random_position()
        # 4 possible moves
        self.directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        self.flag_setup()

    def flag_setup(self):
        # Flags to collect
        self.flags_taken = dict()
        for agent_id in self.current_states.keys():
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

    def move(self, id, position, direction):
        # Direction 0=up, 1=right, 2=down, 3=left
        new_position = position + self.directions[direction]
        collision = False
        for agent_id in self.current_states.keys():
            if agent_id != id and np.array_equal(self.current_states[agent_id], new_position) and not self.terminal(new_position):
                collision = True
                break

        if collision:
            return position
        elif direction == 0 and (new_position[0] >= self.size[0] or self.horizontal_wall(new_position)):
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
    # 1: 0 for each cell in the grid,
    #    1 for the agent's cell number
    # 2: 0 for each non-taken flag,
    #    1 for each flag the agent possesses
    def to_binary(self, id):
        grid_size = self.size[0]*self.size[1]
        bin_repr = np.zeros(grid_size+len(self.flags))
        pos = self.current_states[id]
        bin_repr[pos[0]+self.size[0]*pos[1]] = 1
        for flag in self.flags_taken[id]:
            flag_idx = self.flags.index(flag)
            bin_repr[grid_size+flag_idx] = 1
        return bin_repr

    def start_setup(self):
        self.current_states = dict(self.begins)
        self.flag_setup()

    def start(self, id):
        return self.to_binary(id)

    def step(self, id, action):
        position = self.move(id, self.current_states[id], action)
        if self.at_flag(position):
            if not self.flag_taken(position):
                # take_flag modifies self.flags_taken
                self.take_flag(id, position)
        self.current_states[id] = position
        reward = self.reward(id, position)
        terminal = self.terminal(position)
        return self.to_binary(id), reward, terminal
