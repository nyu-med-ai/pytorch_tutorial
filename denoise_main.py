import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import data.transforms as transforms
from data.kneedata import KneeDataSet
from models.denoisecnn import DenoiseCnn


def save_checkpoint(epoch, model, optimizer, val_loss, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'min_val_loss': val_loss,
    }

    torch.save(checkpoint, path)


def load_checkpoint(checkpoint_file, model, optimizer):
    if os.path.exists(checkpoint_file):
        print('loading existing run from {}'.format(checkpoint_file))

        checkpoint = torch.load(checkpoint_file)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        min_val_loss = checkpoint['min_val_loss']
        epoch = checkpoint['epoch']
    else:
        print('no model found in {}, running from epoch 0'.format(checkpoint_file))

        epoch = 0
        min_val_loss = np.inf

    return epoch, model, optimizer, min_val_loss


def main():
    print('starting denoising')

    noise_sigma = 4e-5  # sigma for the noise simulation
    batch_size = 8  # number of images to run for each minibach
    num_epochs = 200  # number of epochs to train
    validation_seed = 15  # rng seed for validation loop
    log_dir = 'logs/denoise/'  # log dir for models and tensorboard
    device = torch.device('cpu')  # model will run on this device
    dtype = torch.float  # dtype for data and model

    # set up tensorboard
    writer = SummaryWriter(log_dir=log_dir)

    # checkpoint file name
    checkpoint_file = os.path.join(log_dir + 'best_model.pt')

    # -------------------------------------------------------------------------
    # NOISE SIMULATION SETUP
    transform_list = [
        transforms.AddNoise(target_op=False, sigma=noise_sigma),
        transforms.Ifft(norm='ortho'),
        transforms.SquareRootSumSquare(),
        transforms.Normalize(),
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
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    display_dat = val_dataset[15]['dat'].unsqueeze(0).to(
        device=device, dtype=dtype)
    display_target = val_dataset[15]['target'].unsqueeze(0).to(
        device=device, dtype=dtype)
    display_vmax = np.max(np.squeeze(display_dat.cpu().numpy()))

    # -------------------------------------------------------------------------
    # MODEL SETUP
    model = DenoiseCnn(
        num_chans=64,
        num_layers=4,
        magnitude_input=True,
        magnitude_output=True
    )
    model = model.to(device)
    model = model.train()
    print('CNN model information:')
    print(model)

    # -------------------------------------------------------------------------
    # OPTIMIZER SETUP
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()

    # -------------------------------------------------------------------------
    # LOAD PREVIOUS STATE
    start_epoch, model, optimizer, min_val_loss = load_checkpoint(
        checkpoint_file, model, optimizer)
    current_seed = 20

    # -------------------------------------------------------------------------
    # NETWORK TRAINING
    for epoch_index in range(start_epoch, num_epochs):
        print('epoch {} of {}'.format(epoch_index+1, num_epochs))

        # ---------------------------------------------------------------------
        # TRAINING LOOP
        model = model.train()

        # rng seed for noise generation
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        torch.cuda.manual_seed(current_seed)

        # batch loop
        losses = []
        for batch in train_loader:
            target = batch['target'].to(device=device, dtype=dtype)
            dat = batch['dat'].to(device=device, dtype=dtype)

            est = model(dat)  # forward propagation
            loss = loss_fn(est, target)  # calculate the loss
            optimizer.zero_grad()  # clear out old gradients
            loss.backward()  # back propagation
            optimizer.step()  # update the CNN weights

            # keep last 10 minibatches to compute training loss
            losses.append(loss.item())
            losses = losses[-10:]

        print('trailing training loss: {}'.format(np.mean(losses)))

        # ---------------------------------------------------------------------
        # EVALUATION LOOP
        model = model.eval()

        # rng seed for noise generation
        current_seed = np.random.get_state()[1][0]
        torch.manual_seed(validation_seed)
        np.random.seed(validation_seed)
        torch.cuda.manual_seed(validation_seed)

        # batch loop
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                target = batch['target'].to(device=device, dtype=dtype)
                dat = batch['dat'].to(device=device, dtype=dtype)

                est = model(dat)
                loss = loss_fn(est, target)

                val_losses.append(loss.item())

        print('validation loss: {}'.format(np.mean(val_losses)))

        # ---------------------------------------------------------------------
        # VISUALIZATIONS AND CHECKPOINTS
        if np.mean(val_losses) < min_val_loss:
            save_checkpoint(
                epoch_index,
                model,
                optimizer,
                np.mean(val_losses),
                checkpoint_file
            )

        # write the losses
        writer.add_scalar('loss/train', np.mean(losses), epoch_index+1)
        writer.add_scalar('loss/validation',
                          np.mean(val_losses), epoch_index+1)

        # show an example image from the validation data
        model = model.eval()
        with torch.no_grad():
            display_est = model(display_dat)

        writer.add_image(
            'validation/dat',
            display_dat[0]/display_vmax,
            global_step=epoch_index+1
        )
        writer.add_image(
            'validation/cnn',
            display_est[0]/display_vmax,
            global_step=epoch_index+1
        )
        writer.add_image(
            'validation/target',
            display_target[0]/display_vmax,
            global_step=epoch_index+1
        )

    writer.close()


if __name__ == '__main__':
    main()
