import numpy as np


class Ifft(object):
    """Applies the inverse FFT, assuming first dim is coil dim.

    Args:
        dat_op (boolean, default=True): Whether to iFFT 'dat' array.
        targ_op (boolean, default=False): Whether to iFFT 'target' array.
        norm (string, default=None): Normalization routine, use "ortho" for
            orthogonal FFTs.
    """

    def __init__(self, dat_op=True, target_op=False, norm=None):
        self.dat_op = dat_op
        self.target_op = target_op
        self.norm = norm

    def __call__(self, sample):
        target, dat = sample['target'], sample['dat']

        if self.dat_op:
            axes = list(range(1, dat.ndim))

            dat = np.fft.fftshift(
                np.fft.ifftn(
                    np.fft.ifftshift(
                        dat,
                        axes=axes
                    ),
                    axes=axes,
                    norm=self.norm
                ),
                axes=axes
            )

            sample['dat'] = dat

        if self.target_op:
            axes = list(range(1, target.ndim))

            target = np.fft.fftshift(
                np.fft.ifftn(
                    np.fft.ifftshift(
                        target,
                        axes=axes
                    ),
                    axes=axes,
                    norm=self.norm
                ),
                axes=axes
            )

            sample['target'] = target

        return sample

    def __repr__(self):
        out = '\n' + self.__class__.__name__ + '\n'
        out += '------------------------------------------------------------\n'
        out += 'dat_op: {}\n'.format(self.dat_op)
        out += 'target_op: {}\n'.format(self.target_op)
        out += 'norm: {}\n'.format(self.norm)

        return out
