import os

import numpy as np

import matplotlib.pyplot as plt

import data.transforms as transforms
from data.kneedata import KneeDataSet


def main():
    print('starting denoising')

    transform_list = [
        transforms.AddNoise(sigma=1e-10),
        transforms.Ifft(target_op=True, norm='ortho'),
        transforms.SquareRootSumSquare(target_op=True),
        transforms.Normalize(target_op=True),
        transforms.ToTensor(dat_complex=False, target_complex=False)
    ]

    train_dataset = KneeDataSet(
        'pytorch_tutorial_data/',
        'train',
        transform=transforms.Compose(transform_list)
    )

    print('data set information:')
    print(train_dataset)

    print('dataset length: {}'.format(len(train_dataset)))

    sample = train_dataset[15]

    np_dat = np.squeeze(sample['dat'].numpy())
    np_target = np.squeeze(sample['target'].numpy())

    plt.figure(0)
    plt.gray()
    plt.imshow(np_dat)

    plt.figure(1)
    plt.gray()
    plt.imshow(np_target)

    plt.show()


if __name__ == '__main__':
    main()
