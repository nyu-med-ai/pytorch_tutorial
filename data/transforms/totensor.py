import numpy as np
import torch


class ToTensor(object):
    """Convert a set of complex ndarrays to a tensor.

    Args:
        dat_complex (boolean, default=True): Whether 'dat' in sample is a
            complex image. If true, then tensor conversion puts real and
            imaginary components into separate channels.
        target_complex (boolean, default=True): Whether 'target' in sample is a
            complex image. If true, then tensor conversion puts real and
            imaginary components into separate channels.
        dtype (torch.dtype, default=torch.float): The PyTorch data type to cast
            the tensors to.
    """

    def __init__(self, dat_complex=False, target_complex=False, dtype=torch.float):
        self.dat_complex = dat_complex
        self.target_complex = target_complex
        self.dtype = dtype

    def __call__(self, sample):
        target, dat = sample['target'], sample['dat']

        # add a channel dimension to magnitude inputs
        if self.dat_complex:
            dat = np.concatenate((np.real(dat), np.imag(dat)))
        else:
            dat = np.expand_dims(dat, 0)

        if self.target_complex:
            target = np.concatenate((np.real(target), np.imag(target)))
        else:
            target = np.expand_dims(target, 0)

        return {
            'target': torch.tensor(target, dtype=self.dtype),
            'dat': torch.tensor(dat, dtype=self.dtype)
        }

    def __repr__(self):
        out = '\n' + self.__class__.__name__ + '\n'
        out += '------------------------------------------------------------\n'
        out += 'dat_complex: {}\n'.format(self.dat_complex)
        out += 'target_complex: {}\n'.format(self.target_complex)
        out += 'float_flag: {}\n'.format(self.float_flag)

        return out
