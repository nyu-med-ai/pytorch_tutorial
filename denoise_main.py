import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import data.transforms as transforms
from data.kneedata import KneeDataSet
from models.residdenoisecnn import ResidDenoiseCnn


def worker_init_fn(worker_id):
    """Pytorch worker initialization function."""
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def main(display_visuals=False):
    print('starting denoising')

    noise_sigma = 2e-5
    batch_size = 16
    num_epochs = 20
    num_workers = 4
    device = torch.device('cuda')
    dtype = torch.float

    # -------------------------------------------------------------------------
    # NOISE SIMULATION SETUP
    transform_list = [
        transforms.AddNoise(sigma=noise_sigma),
        transforms.Ifft(target_op=True, norm='ortho'),
        transforms.SquareRootSumSquare(target_op=True),
        transforms.Normalize(target_op=True),
        transforms.ToTensor(dat_complex=False, target_complex=False)
    ]

    # -------------------------------------------------------------------------
    # DATALOADER SETUP
    train_dataset = KneeDataSet(
        'pytorch_tutorial_data/',
        'train',
        transform=transforms.Compose(transform_list)
    )
    print('data set information:')
    print(train_dataset)
    val_dataset = KneeDataSet(
        'pytorch_tutorial_data/',
        'val',
        transform=transforms.Compose(transform_list)
    )
    # convert to a PyTorch dataloader
    # this handles batching, random shuffling, parallelization
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn
    )

    # -------------------------------------------------------------------------
    # MODEL SETUP
    model = ResidDenoiseCnn(
        num_chans=64,
        num_layers=3,
        magnitude_input=True,
        magnitude_output=True
    )
    model = model.to(device)
    print('CNN model information:')
    print(model)

    # -------------------------------------------------------------------------
    # OPTIMIZER SETUP
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()

    # -------------------------------------------------------------------------
    # NETWORK TRAINING
    for epoch_index in range(num_epochs):
        print('epoch {} of {}'.format(epoch_index+1, num_epochs))

        # ---------------------------------------------------------------------
        # TRAINING LOOP
        losses = []
        for i, batch in enumerate(train_loader):
            target, dat = \
                batch['target'].to(device), batch['dat'].to(device)

            est = model(dat)
            loss = loss_fn(est, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            losses = losses[-50:]

            print('training loop progress: {:.0f}%'.format(
                100*(i+1)/len(train_loader)))

        print('trailing training loss: {}'.format(np.mean(losses)))

        # ---------------------------------------------------------------------
        # EVALUATION LOOP
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                target, dat = \
                    batch['target'].to(device), batch['dat'].to(device)

                est = model(dat)
                loss = loss_fn(est, target)

                val_losses.append(loss.item())

        print('validation loss: {}'.format(np.mean(val_losses)))


if __name__ == '__main__':
    main()
