import numpy as np


class Normalize(object):
    """Divides by max value in 'dat'.

    Args:
        dat_op (boolean, default=True): Whether to apply to 'dat' array.
        target_op (boolean, default=False): Whether to apply to 'target' array.
        thresh_val (double, default=1e-10): Minimum division value for
            divide-by-0 protection.
    """

    def __init__(self, dat_op=True, target_op=True, thresh_val=1e-10):
        self.dat_op = dat_op
        self.target_op = target_op
        self.thresh_val = thresh_val

    def __call__(self, sample):
        target, dat = sample['target'], sample['dat']

        norm_val = np.maximum(np.max(dat), self.thresh_val)

        if self.dat_op:
            dat = dat / norm_val

            sample['dat'] = dat

        if self.target_op:
            target = target / norm_val

            sample['target'] = target

        return sample

    def __repr__(self):
        out = '\n' + self.__class__.__name__ + '\n'
        out += '------------------------------------------------------------\n'
        out += 'dat_op: {}\n'.format(self.dat_op)
        out += 'target_op: {}\n'.format(self.target_op)
        out += 'thresh_val: {}\n'.format(self.thresh_val)

        return out
