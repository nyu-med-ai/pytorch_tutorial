class Compose(object):
    """Composes several transforms together.
        Args:
            transforms (list of ``Transform`` objects): list of transforms 
            to compose.
        Example:
            >>> transforms.Compose([
            >>>     transforms.MriNoise(),
            >>>     transforms.ToTensor(),
            >>> ])

        Returns:
            ob (PyTorch transform object): Can be used with PyTorch dataset
                with transform=ob option.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)

        return sample

    def __repr__(self):
        out = '\n' + self.__class__.__name__ + '\n'
        out += '------------------------------------------------------------\n'
        out += 'Transform List:\n'

        for t in self.transforms:
            out += t.__class__.__name__ + '\n'

        return out
