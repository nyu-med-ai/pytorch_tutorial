import numpy as np
import skimage


class Resize(object):
    """Resizes an image. Only accepts magnitude inputs.

    Args:
        dat_op (boolean, default=True): Whether to resize 'dat' array.
        target_op (boolean, default=False): Whether to resize 'target' array.
        output_shape (string, default=(256, 128)): Output shape of the images.
    """

    def __init__(self, dat_op=True, target_op=False, output_shape=(256, 128)):
        self.dat_op = dat_op
        self.target_op = target_op
        self.output_shape = output_shape

    def __call__(self, sample):
        target, dat = sample['target'], sample['dat']

        if self.dat_op:
            dat = skimage.transform.resize(np.real(dat), self.output_shape)

            sample['dat'] = dat

        if self.target_op:
            target = skimage.transform.resize(
                np.real(target), self.output_shape)

            sample['target'] = target

        return sample

    def __repr__(self):
        out = '\n' + self.__class__.__name__ + '\n'
        out += '------------------------------------------------------------\n'
        out += 'dat_op: {}\n'.format(self.dat_op)
        out += 'target_op: {}\n'.format(self.target_op)
        out += 'output_shape: {}\n'.format(self.output_shape)

        return out
