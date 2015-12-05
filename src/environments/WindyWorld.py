from GridWorld import GridWorld
import numpy as np

class WindyWorld(GridWorld):
    def __init__(self, wind=None, **kwargs):
        super(WindyWorld, self).__init__(**kwargs)
        self.wind = wind
        if wind is None:
            self.wind = np.zeros(self.size[0])
        assert len(self.wind) == self.size[0], 'Specify wind-strength for each x-position'
        # 8 possible moves
        self.directions = np.array([[0,1], [1,1], [1,0], [1,-1], [0,-1], [-1,-1], [-1, 0], [-1, 1]])

    def wind_strength(self, position):
        wind = self.wind[position[0]]
        if wind > 0:
            wind = np.random.choice([wind-1, wind, wind+1])
        return wind

    def move(self, position, direction):
        moved = super(WindyWorld, self).move(position, direction)
        moved[1] += self.wind_strength(moved)
        return self.in_grid(moved)