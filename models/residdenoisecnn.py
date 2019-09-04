import torch.nn as nn


class ResidDenoiseCnn(nn.Module):
    """A residual denoising CNN model.

    Args:
        num_chans (int): Number of channels.
        num_layers (int): Number of layers.
        magnitude_input (boolean, default=True): Whether input is magnitude
            (True, 1-channel) or complex (False, 2-channel).
        magnitude_output (boolean, default=True): Whether output is magnitude
            (True, 1-channel) or complex (False, 2-channel).
    """

    def __init__(self, num_chans, num_layers, magnitude_input=True,
                 magnitude_output=True):
        super(ResidDenoiseCnn, self).__init__()

        # store the inputs as class attributes
        self.num_chans = num_chans
        self.num_layers = num_layers

        # allow for magnitude or complex inputs/outputs
        if magnitude_input:
            model_input_chans = 1
        else:
            model_input_chans = 2
        if magnitude_output:
            model_output_chans = 1
        else:
            model_output_chans = 2

        # create the layers
        layer_list = []
        in_ch = model_input_chans
        out_ch = num_chans

        if num_layers > 0:
            for i in range(num_layers):
                if i > 1:
                    in_ch = num_chans  # first layer
                if i == num_layers - 1:
                    out_ch = model_output_chans  # last layer

                layer_list.append(
                    nn.Conv2d(
                        in_channels=in_ch,  # num input channels
                        out_channels=out_ch,  # number of filters
                        kernel_size=3,  # size of the filter
                        padding=1  # padding for border of image
                    )
                )
                layer_list.append(
                    # applies a covariate shift for faster training
                    nn.BatchNorm2d(num_chans)
                )
                layer_list.append(
                    nn.ReLU()  # activation function
                )

            # combine it all into a PyTorch sequential module
            self.conv_sequence = nn.Sequential(*layer_list)
        else:
            print('input layers < 1!')

    def forward(self, x):
        return x + self.conv_sequence(x)
