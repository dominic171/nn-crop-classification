# nn-crop-classification
Neural Networks for Crop Classification using Satellite Imagery

This repository contains the code to:

1. Process data downloaded from the EuroCrops dataset (www.eurocrops.tum.de)
2. Run a training process for different models with a specified configuration
3. Save the predictions together with the computed metrics.

NOTE: To achieve a meaningful performance, it is recommended you run the code on a system with an Nvidia GPU and CUDA set up.

## Usage

To run the preconfigured experiments (or once you have changed it to your setup), execute:

```bash
python3 ./main
```

## Setup

NOTE: This code was mostly tested on Python version 3.9.6

To setup the repository on your system:
1. Clone the repository
2. Create a new virtual environment (e.g. using venv)
3. Install the requirements (as listed in requirements.txt)

## Code Structure explained

The code is separated into the folders models, utils, data, and results & the files main.py, eurocrops.py, and trainer.py

models: This folder is intended for the neural network definitions (e.g. your torch.nn.Module class)

trainer.py: Based on the configuration passed in by the main file, it:
1. loads the data
2. sets up the PyTorch datasets and data loaders
3. instantiates the model
4. executes the PyTorch training loop and tests the model
5. saves the result to the results folder

main.py: This is the file you want to execute. It:
1. Contains the "run configuration" for each model, you want to train. That config is passed to the trainer and contains all the parameters that can change.
2. In here, you can easily add a new configuration to test an additional model or setup. Note that some more special models may require small changes in the trainer.py file.

eurocrops.py: This contains all the code to load and process the EuroCrops demo dataset. It provides the EuroCrops dataset class (PyTorch format).

utils: This folder contains additional definitions that are required by some model or the trainer

data: This folder is intended to contain your dataset (e.g. EuroCrops)

results: This folder is the target for saving model weights and the results of the experiment.

## Legal

### Copyright Notice

Copyright &copy; 2024 ETH Zurich, Dominic Hobi

Licensed under the MIT License - see LICENSE file for details.

### Third-party Software

This project includes third-party software.

For copyright and license notices - see NOTICE file.
