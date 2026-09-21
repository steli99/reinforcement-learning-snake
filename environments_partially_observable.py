"""The same Snake dynamics with an egocentric square observation crop."""
import numpy as np
from environments_fully_observable import BaseEnvironment as FullEnvironment


class BaseEnvironment(FullEnvironment):
    def __init__(self, n_boards, board_size, mask_size, seed=None):
        if int(mask_size) != mask_size or mask_size < 0:
            raise ValueError('mask_size must be a nonnegative integer radius')
        self.mask_size = int(mask_size)
        super().__init__(n_boards, board_size, seed=seed)

    def to_state(self):
        radius = self.mask_size
        padded = np.pad(self.boards, ((0, 0), (radius, radius), (radius, radius)),
                        mode='constant', constant_values=self.WALL)
        heads = np.argwhere(self.boards == self.HEAD)
        patches = np.stack([padded[b, row:row+2*radius+1, col:col+2*radius+1]
                            for b, row, col in heads])
        return np.eye(5, dtype=np.float32)[patches][..., 1:]


class OriginalSnakeEnvironment(BaseEnvironment):
    """mask_size=2 returns a 5x5 crop centered on the head."""
