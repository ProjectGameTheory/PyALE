from Projector import Projector
from util.CTiles import tiles
import numpy as np


class TileCoding(Projector):

    def __init__(self, num_tilings=[10],
                 num_tiles=np.array([[10, 10]]),
                 limits=np.array([[0, 1], [0, 1]]),
                 memory_size=32768, id=0):
        if np.isscalar(num_tiles):
            num_tiles = num_tiles * \
                np.ones((len(num_tilings), limits.shape[0]))
        if not (type(num_tiles) is np.ndarray):
            num_tiles = np.array(num_tiles)
        assert limits.shape[0] == num_tiles.shape[1], 'Dimension mismatch'
        assert len(num_tilings) == num_tiles.shape[0], 'Undefined tilings'
        Projector.__init__(self, False, limits)
        self.memory_size = memory_size
        self.num_tilings = num_tilings
        self.total_tilings = np.sum(num_tilings)
        self.num_tiles = 1. / num_tiles
        self.id = id

    def num_features(self):
        return self.memory_size

    def supports_sparse(self):
        return True

    def phi_idx(self, state):
        n_state = self.normalize_state(state)
        ind = []
        for idx, nt in enumerate(self.num_tilings):
            ind.extend(tiles.tiles(nt, self.memory_size,
                                   (n_state / self.num_tiles[idx, :]).tolist(), [self.id + idx]))
        return np.array(ind)

    def phi(self, state):
        phi_vector = np.zeros(self.memory_size)
        inds = self.phi_idx(state)
        phi_vector[inds] = 1.
        return phi_vector
