import numpy as np


class SquareRootSumSquare(object):
    """Combines coils via square-root-sum-squares, assuming first dim is coil dim.

    Args:
        dat_op (boolean, default=True): Whether to apply to 'dat' array.
        target_op (boolean, default=False): Whether to apply to 'target' array.
    """

    def __init__(self, dat_op=True, target_op=True):
        self.dat_op = dat_op
        self.target_op = target_op

    def __call__(self, sample):
        target, dat = sample['target'], sample['dat']

        if self.dat_op:
            dat = np.sqrt(np.sum(np.absolute(dat)**2, axis=0))

            sample['dat'] = dat

        if self.target_op:
            target = np.sqrt(np.sum(np.absolute(target)**2, axis=0))

            sample['target'] = target

        return sample

    def __repr__(self):
        out = '\n' + self.__class__.__name__ + '\n'
        out += '------------------------------------------------------------\n'
        out += 'dat_op: {}\n'.format(self.dat_op)
        out += 'target_op: {}\n'.format(self.target_op)

        return out
