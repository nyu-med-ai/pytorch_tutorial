
# 90-Minute PyTorch Blitz

A repository with an example for training a basic denoising CNN in PyTorch.

Parts of this tutorial are based on the 60-Minute PyTorch Blitz by Soumith
Chintala, available at

```url
https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
```

## Goals

1) Introduce attendees to PyTorch.
2) Provide code for interacting with Matlab data.
3) Demonstrate coding and training of a basic, 4-layer CNN.

## Installation

First, you'll need to install Anaconda from here:

```url
https://www.anaconda.com/distribution/
```

Anaconda is available for Windows, Mac, and Linux.

Then, you'll need to download this repository. After you download this
repository, open the Anaconda prompt (on Linux or MacOS, this should just be
a terminal. On Windows it's a program called "Anaconda Prompt") and navigate
to the directory where you downloaded the repository. Then, type the following:

```sh
conda create --name pytorch_tutorial
conda activate pytorch_tutorial
bash anaconda_setup.bash
```

If you are on Windows you can copy-paste the contentds of anaconda_setup.bash
into your Anaconda prompt. You should get a bunch of installation messages.
After this is complete, you should verify your installation. To do this, first
type `python`. Then, once the Python interpreter is running, type

```python
import torch
print(torch.__version__)
```

If you see 1.2.0 or greater for the version, you should be able to run all
examples in this repository.

## Basic Example

### Data

Data is automatically loaded from a ```pytorch_tutorial_data/``` folder in the
root directory of the repository.

Data for this code is provided internally to NYU researchers. For external
researchers, executing the examples in this repository can be done with any set
of multicoil, Cartesian k-space data. One such set of data can be downloaded at
```https://fastmri.med.nyu.edu/``` (with some conversions for the dataloader
in this repository). The included dataloader expects the raw data to be stored
as contiguous, multicoil complex Matlab arrays. Behavior can be inferred by
inspecting ```data/kneedata.py``` and the corresponding transform modules.

### Running the example

Jupyter notebooks can be found in ```tensors_and_autograd_tutorial.ipynb``` and
```neural_networks_tutorial.ipynb``` in the base directory. These examples are
primarily meant to be didactic - to fully explore the code, we recommend
running the main function.

To run the main example, after installing Anaconda with the required packages
you should just have to run

```python
python denoise_main.py
```

After that the training will begin. outputs from the training are logged using
tensorboard into a new `logs/` directory in the folder, along with the best
model. If the training is interrupted, rerunning `denoise_main.py` will search
the `logs/` folder for the previous best model and continue training from
there.
