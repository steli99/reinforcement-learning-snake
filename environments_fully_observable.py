"""Batched Snake with the coursework rewards and nonterminal collision rules.

The simulator is NumPy-only. `move` retains the reward-column interface;
`step` additionally exposes win/reset flags needed for Bellman bootstrapping.
"""
import numpy as np


class BaseEnvironment:
    HEAD, BODY, FRUIT, EMPTY, WALL = 4, 3, 2, 1, 0
    UP, RIGHT, DOWN, LEFT, NONE = 0, 1, 2, 3, 4
    OFFSETS = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])

    def __init__(self, n_boards, board_size, seed=None):
        if int(n_boards) != n_boards or n_boards < 1:
            raise ValueError('n_boards must be a positive integer')
        if int(board_size) != board_size or board_size < 4:
            raise ValueError('board_size must be an integer >= 4')
        self.WIN_REWARD, self.FRUIT_REWARD, self.STEP_REWARD = 1., .5, 0.
        self.ATE_HIMSELF_REWARD, self.HIT_WALL_REWARD = -.2, -.1
        self.board_size, self.n_boards = int(board_size), int(n_boards)
        self.rng = np.random.default_rng(seed)
        self.boards = np.empty((self.n_boards, self.board_size, self.board_size), dtype=np.int8)
        self.bodies = [[] for _ in range(self.n_boards)]
        self.reset()

    def get_board(self):
        board = np.full((self.board_size, self.board_size), self.EMPTY, dtype=np.int8)
        board[[0, -1], :] = self.WALL
        board[:, [0, -1]] = self.WALL
        interior = np.argwhere(board == self.EMPTY)
        head = interior[self.rng.integers(len(interior))]
        board[tuple(head)] = self.HEAD
        return board

    def _place_fruit(self, index):
        free = np.argwhere(self.boards[index] == self.EMPTY)
        cell = free[self.rng.integers(len(free))]
        self.boards[index, cell[0], cell[1]] = self.FRUIT

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        for b in range(self.n_boards):
            self.boards[b] = self.get_board()
            self.bodies[b] = []
            self._place_fruit(b)
        self.last_dones = np.zeros(self.n_boards, dtype=bool)
        return self.to_state()

    def check_actions(self, new_heads):
        new_heads = np.asarray(new_heads, dtype=int)
        b, row, col = new_heads.T
        outside = (row < 0) | (col < 0) | (row >= self.board_size) | (col >= self.board_size)
        wall = outside.copy()
        valid = ~outside
        wall[valid] |= self.boards[b[valid], row[valid], col[valid]] == self.WALL
        return b[wall]

    def step(self, actions):
        # Accept both (N,) and (N,1), including eager TensorFlow tensors.
        actions = np.asarray(actions)
        if actions.shape not in [(self.n_boards,), (self.n_boards, 1)]:
            raise ValueError('Expected one action per board, shape (N,) or (N,1)')
        if not np.all(np.isfinite(actions)) or not np.all(actions == actions.astype(int)):
            raise ValueError('Actions must be integers')
        actions = actions.astype(int).reshape(-1)
        if np.any((actions < 0) | (actions > 3)):
            raise ValueError('Action must be UP=0, RIGHT=1, DOWN=2, or LEFT=3')
        heads = np.argwhere(self.boards == self.HEAD)
        if len(heads) != self.n_boards or not np.array_equal(heads[:, 0], np.arange(self.n_boards)):
            raise RuntimeError('Each board must contain exactly one head')
        rewards = np.full(self.n_boards, self.STEP_REWARD, dtype=np.float32)
        dones = np.zeros(self.n_boards, dtype=bool)
        fruit, wall, bite = (np.zeros(self.n_boards, dtype=bool) for _ in range(3))
        for b, (_, row, col) in enumerate(heads):
            old_head = (int(row), int(col))
            new_head = tuple(np.asarray(old_head) + self.OFFSETS[actions[b]])
            target = self.boards[b][new_head]
            if target == self.WALL:
                # A blocked action must not duplicate the head in the body list.
                wall[b] = True
                rewards[b] = self.HIT_WALL_REWARD
                continue
            fruit[b] = target == self.FRUIT
            body = list(self.bodies[b])
            if target == self.BODY:
                # Preserve the supplied assignment's truncation rule, not death.
                bite[b] = True
                hit_index = body.index(new_head)
                body = body[:hit_index]
                rewards[b] = self.ATE_HIMSELF_REWARD
            body.insert(0, old_head)
            if not fruit[b]:
                body.pop()
            board = self.boards[b]
            board[(board == self.HEAD) | (board == self.BODY)] = self.EMPTY
            if body:
                cells = np.asarray(body)
                board[cells[:, 0], cells[:, 1]] = self.BODY
            board[new_head] = self.HEAD
            self.bodies[b] = body
            if fruit[b]:
                rewards[b] = self.FRUIT_REWARD
                if not np.any(board == self.EMPTY):
                    # Reward the win, then start the next independent episode.
                    rewards[b] = self.WIN_REWARD
                    dones[b] = True
                    self.boards[b] = self.get_board()
                    self.bodies[b] = []
                self._place_fruit(b)
        self.last_dones = dones
        info = {'fruit': fruit, 'wall': wall, 'self_bite': bite, 'win': dones.copy(),
                'length': np.array([len(body) + 1 for body in self.bodies])}
        return self.to_state(), rewards, dones, info

    def move(self, actions):
        _, rewards, _, _ = self.step(actions)
        return rewards[:, None]

    def to_state(self):
        # Four channels: empty, fruit, body, head. Wall is the all-zero vector.
        return np.eye(5, dtype=np.float32)[self.boards][..., 1:]


class OriginalSnakeEnvironment(BaseEnvironment):
    """Full-board observation; internal ordered body remains in the simulator."""
