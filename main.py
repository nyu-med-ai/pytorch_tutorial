import os

from data.kneedata import KneeDataSet


def main():
    print('starting denoising')

    train_dataset = KneeDataSet(
        'pytorch_tutorial_data/',
        'train',
    )

    print('dataset length: {}'.format(len(train_dataset)))

    sample = train_dataset[0]
    print('first size: {}'.format(sample['target'].shape))


if __name__ == '__main__':
    main()
