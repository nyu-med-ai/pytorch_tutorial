"""Knee data set.

This file contains a class for interacting with kneeGRASP data sets. It is intended
for use as a data loader interface for PyTorch. It treats each slice as a separate
sample. When the __getitem__ function is invoked via dataset[index], it retrieves
the slice at index from the dataset folder and returns its multicoil k-space.

Standard usage is as follows:
    Initialization:

        >>> dataset = KneeDataSet(data_dir, 'train')

    Query number of examples in dataset:

        >>> num_examples = len(dataset)

    Sampling index ind:

        >>> sample = dataset[ind]
"""

import itertools
import os
import time

import h5py
import numpy as np
from torch.utils.data import Dataset


class KneeDataSet(Dataset):
    """A data iterator for a knee data set.

    Args:
        root_dir (string): The directory with all the images.
        split (string): The split of the data set (e.g., 'train', 'val', 'test')
        transform (object, default: None): A transform object for manipulating
            the samples.
    """

    def __init__(self, root_dir, split, transform=None):
        self.directory = os.path.join(root_dir, split)
        self.transform = transform
        self.ndims = 3

        # looks in training directory, counts the files, and returns their paths as a list
        self.file_list = [f for f in os.listdir(
            self.directory) if os.path.isfile(os.path.join(self.directory, f))]
        self.num_files = len(self.file_list)

        # go through each file and count the slices in the file, put in a list
        # with tuples of (fileindex, sliceindex)
        self.slice_list = []
        for i, filename in enumerate(self.file_list):
            with h5py.File(os.path.join(self.directory, filename), 'r') as hf:
                self.slice_list = self.slice_list + list(
                    itertools.product(
                        range(i, i+1),
                        range(hf['kspace'].shape[0])
                    )
                )

    def __len__(self):
        return len(self.slice_list)

    def __getitem__(self, index):
        """Retrieve one knee data point.

        Returns:
            sample (dict): Dictionary with keys
                'dat': The corrupted data for input into neural network.
                'target': The "true" data the neural network is regressing towards.
        """
        file_index = self.slice_list[index][0]  # retrieve the file name
        slice_index = self.slice_list[index][1]  # retrieve the slice index

        # open the file and retrieve the data
        filename = os.path.join(self.directory, self.file_list[file_index])
        with h5py.File(filename, 'r') as hf:
            kdata = np.array(hf['kspace'][slice_index]).view(np.complex64)

        # start by copying same data into both 'dat' and 'target'
        sample = {
            'dat': kdata,
            'target': np.copy(kdata)
        }

        # use the transforms on the data
        # typically the transforms will corrupt 'dat' while leaving 'target' pristine
        if not self.transform == None:
            sample = self.transform(sample)

        return sample

    def __repr__(self):
        """Output for print(dataset) command."""
        out = '\n' + self.__class__.__name__ + '\n'
        out += '------------------------------------------------------------\n'
        out += 'directory: {}\n'.format(self.directory)
        out += 'total number of slices: {}\n'.format(len(self.slice_list))

        if not self.transform == None:
            out += '\n'
            out += 'Transform List:\n'

            for t in self.transform.transforms:
                out += t.__class__.__name__ + '\n'

        return out
