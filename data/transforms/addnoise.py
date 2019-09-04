import numpy as np


class AddNoise(object):
    """Adds complex Gaussian noise to data.

    Args:
        dat_op (boolean, default=True): Whether to apply to 'dat' array.
        target_op (boolean, default=False): Whether to apply to 'target' array.
        sigma (double, default=1): Standard deviation of the noise.
    """

    def __init__(self, dat_op=True, target_op=False, sigma=1):
        self.dat_op = dat_op
        self.target_op = target_op
        self.sigma = sigma

    def __call__(self, sample):
        target, dat = sample['target'], sample['dat']

        if self.dat_op:
            dat = dat + self.sigma * (
                np.random.normal(size=dat.shape) +
                1j * np.random.normal(size=dat.shape)
            )

            sample['dat'] = dat

        if self.target_op:
            target = target + self.sigma * (
                np.random.normal(size=target.shape) +
                1j * np.random.normal(size=target.shape)
            )

            sample['target'] = target

        return sample

    def __repr__(self):
        out = '\n' + self.__class__.__name__ + '\n'
        out += '------------------------------------------------------------\n'
        out += 'dat_op: {}\n'.format(self.dat_op)
        out += 'target_op: {}\n'.format(self.target_op)

        return out
