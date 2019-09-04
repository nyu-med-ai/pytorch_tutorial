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
            self.model_input_chans = 1
        else:
            self.model_input_chans = 2
        if magnitude_output:
            self.model_output_chans = 1
        else:
            self.model_output_chans = 2

        # create the layers
        layer_list = []
        in_ch = self.model_input_chans
        out_ch = num_chans

        if num_layers > 0:
            for i in range(num_layers):
                if i > 0:
                    in_ch = num_chans  # first layer
                if i == num_layers - 1:
                    out_ch = self.model_output_chans  # last layer

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
                    nn.BatchNorm2d(out_ch)
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

    def __repr__(self):
        """Output for print(model) command."""
        out = '\n' + self.__class__.__name__ + '\n'
        out += '------------------------------------------------------------\n'
        out += 'model_input_chans: {}\n'.format(self.model_input_chans)
        out += 'model_output_chans: {}\n'.format(self.model_output_chans)
        out += 'num_layers: {}\n'.format(self.num_layers)
        out += 'num_chans: {}\n'.format(self.num_chans)

        num_parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        out += 'num_parameters: {}\n'.format(num_parameters)

        return out
