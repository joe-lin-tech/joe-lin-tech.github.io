---
layout: page
title: cifar image classification
description: using popular deep learning architectures
img: assets/img/cifar/preview.webp
importance: 2
related_publications: true
references: projects/cifar.bib
---

> Classify images from the CIFAR-10 dataset using a variety of modern architectures.

## Project Overview
This project implements a training and testing pipeline for an image classification task on the CIFAR-10 dataset. CIFAR-10 contains 60,000 32x32 RGB images distributed evenly across 10 image classes (6,000 images per class). The provided dataset splits consists of a train set with 50,000 images and a test set with 10,000 images. Here, the train set is further split into a train set with 45,000 images and a validation set with 5,000 images to allow for model evaluation throughout the training process. The models implemented in this repository includes a basic CNN, a resnet, and a vision transformer.

## Setup and Run
The repository contains both a python script and a Jupyter notebook. Each of their setup/run procedures are detailed below.

### Python Script
Clone the repository.
```shell
git clone git@github.com:joe-lin-tech/cifar.git
cd cifar
```

Create and activate a virtual environment. (Alternatively, use an existing environment of your choosing.)
```shell
python3 -m venv venv
source venv/bin/activate
```

Install required pip packages and dependencies.
```shell
python3 -m pip install -r requirements.txt
```

Login to a wandb account if you'd like to view train logs. (If not, make sure to toggle respective flag when running.)
```shell
wandb login
```

Your local environment should now be suitable to run the main script ```train.py```. You can either run it interactively or use the shell to specify run options.

#### Run Interactively
```shell
python3 train.py
```

#### Run in the Shell
```shell
python3 train.py -m previt -d cuda
```
The above command fine tunes a vision transformer pretrained on ImageNet with hyperparameters set to those used in this project. For reproducibility tests, specifying ```shell -m``` and ```shell -d``` like above will be sufficient. Additional specifiers detailed below.

```shell
python3 train.py -m resnet -e 50 -b 128 -l 0.1 -d cuda
```
As an example of a more customized run, the above command trains a resnet-based model on cuda for 50 epochs with batch size of 128 and initial learning rate of 0.1.

<table id="table" data-toggle="table" class="mb-3" style="width: 100%">
    <thead>
        <tr>
            <th data-field="specifier">Specifier</th>
            <th data-field="usage">Usage</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>-m</code>, <code>--model</code></td>
            <td>choose model architecture (<code>cnn</code>, <code>resnet</code>, <code>previt</code>, or <code>vit</code>)</td>
        </tr>
        <tr>
            <td><code>-e</code>, <code>--epoch</code></td>
            <td>number of epochs</td>
        </tr>
        <tr>
            <td><code>-b</code>, <code>--batch-size</code></td>
            <td>batch size</td>
        </tr>
        <tr>
            <td><code>-l</code>, <code>--learning-rate</code></td>
            <td>learning rate</td>
        </tr>
        <tr>
            <td><code>-d</code>, <code>--device</code></td>
            <td>device</td>
        </tr>
        <tr>
            <td><code>-c</code>, <code>--cross-validate</code></td>
            <td>flag for training with 5-fold cross-validation (default: False)</td>
        </tr>
        <tr>
            <td><code>-w</code>, <code>--wandb</code></td>
            <td>flag for wandb logging (default: False)</td>
        </tr>
        <tr>
            <td><code>-s</code>, <code>--save-folder</code></td>
            <td>path to desired model save folder (default: current working directory)</td>
        </tr>
        <tr>
            <td><code>-f</code>, <code>--ckpt-frequency</code></td>
            <td>how often to save model checkpoint, in number of epochs (default: 0, save final)</td>
        </tr>
    </tbody>
</table>

### Jupyter Notebook
Download the Jupyter notebook and run the first cell to import relevant packages. The following Python packages are used for this project and may need to be installed directly (if not installed in current environment) with ```!pip install <package name>```.

- **General Purpose:** For shuffling and seeding random processes, use ```random```. To read and write to local file system, use ```os```.
- **Data Manipulation:** Use ```numpy``` to represent and manipulate data.
- **Machine Learning:** Use ```torch``` and ```torchvision```, which are suitable for Computer Vision tasks. For logging the training loop, use ```wandb```.

Run the remaining cells to execute the training procedure of the latest notebook version (pretrained vision transformer).

## Model Architecture and Training

### Basic CNN Architecture
This implementation consists of 3 convolutional layers (conv + relu + max pool) and a fully connected network.

<table id="table" data-toggle="table" class="mb-3" style="width: 100%">
    <thead>
        <tr>
            <th data-field="layer">Layer</th>
            <th data-field="parameters">Parameters</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>nn.Conv2d</code></td>
            <td><code>in_channels</code> = 3, <code>out_channels</code> = 8, <code>kernel_size</code> = 5, <code>stride</code> = 1, <code>padding</code> = 2</td>
        </tr>
        <tr>
            <td><code>nn.MaxPool2d</code></td>
            <td><code>kernel_size</code> = 2, <code>stride</code> = 2</td>
        </tr>
        <tr>
            <td><code>nn.Conv2d</code></td>
            <td><code>in_channels</code> = 8, <code>out_channels</code> = 16, <code>kernel_size</code> = 5, <code>stride</code> = 1, <code>padding</code> = 2</td>
        </tr>
        <tr>
            <td><code>nn.MaxPool2d</code></td>
            <td><code>kernel_size</code> = 2, <code>stride</code> = 2</td>
        </tr>
        <tr>
            <td><code>nn.Conv2d</code></td>
            <td><code>in_channels</code> = 16, <code>out_channels</code> = 32, <code>kernel_size</code> = 5, <code>stride</code> = 1, <code>padding</code> = 2</td>
        </tr>
        <tr>
            <td><code>nn.MaxPool2d</code></td>
            <td><code>kernel_size</code> = 2, <code>stride</code> = 2</td>
        </tr>
        <tr>
            <td><code>nn.Linear</code></td>
            <td><code>in_channels</code> = 512, <code>out_channels</code> = 64</td>
        </tr>
        <tr>
            <td><code>nn.Linear</code></td>
            <td><code>in_channels</code> = 64, <code>out_channels</code> = 32</td>
        </tr>
        <tr>
            <td><code>nn.Linear</code></td>
            <td><code>in_channels</code> = 32, <code>out_channels</code> = 10</td>
        </tr>
    </tbody>
</table>

Using the hyperparameters below, the model is capable of achieving ~50% test accuracy on CIFAR-10.

<table id="table" data-toggle="table" class="mb-3" style="width: 20%">
    <thead>
        <tr>
            <th data-field="hyperparameter">Hyperparameter</th>
            <th data-field="value">Value</th>
        </tr>
    </thead>
    <tbody>
        <tr><td>EPOCHS</td><td>20</td></tr>
        <tr><td>BATCH_SIZE</td><td>128</td></tr>
        <tr><td>LEARNING_RATE</td><td>1e-4</td></tr>
    </tbody>
</table>

<table id="table" data-toggle="table" class="mb-3" style="width: 40%">
    <thead>
        <tr>
            <th data-field="optimizer">Optimizer</th>
            <th data-field="parameters">Parameters</th>
        </tr>
    </thead>
    <tbody>
        <tr><td>Adam</td><td><code>weight_decay</code> = 0.01</td></tr>
    </tbody>
</table>

Below is the wandb log of training the basic CNN model:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/cifar/cnn.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fig 1. wandb logs from basic cnn training.
</div>

### ResNet Architecture
This implementation utilizes residual connections to improve learning and allow us to build a deeper neural network, all whilst maintaining gradient flow. The original ResNet paper was referred to for implementation and technical details {% cite he2015deepresiduallearningimage -f projects/cifar %}.

Using the hyperparameters below, the model is capable of achieving ~91% test accuracy on CIFAR-10.

<table id="table" data-toggle="table" class="mb-3" style="width: 20%">
    <thead>
        <tr>
            <th data-field="hyperparameter">Hyperparameter</th>
            <th data-field="value">Value</th>
        </tr>
    </thead>
    <tbody>
        <tr><td>EPOCHS</td><td>50</td></tr>
        <tr><td>BATCH_SIZE</td><td>128</td></tr>
        <tr><td>LEARNING_RATE</td><td>0.1</td></tr>
    </tbody>
</table>

<table id="table" data-toggle="table" class="mb-3" style="width: 70%">
    <thead>
        <tr>
            <th data-field="optimizer">Optimizer</th>
            <th data-field="parameters">Parameters</th>
        </tr>
    </thead>
    <tbody>
        <tr><td>SGD</td><td><code>momentum</code> = 0.9, <code>weight_decay</code> = 5e-4, <code>nesterov</code> = True</td></tr>
    </tbody>
</table>

<table id="table" data-toggle="table" class="mb-3" style="width: 80%">
    <thead>
        <tr>
            <th data-field="scheduler">Scheduler</th>
            <th data-field="parameters">Parameters</th>
        </tr>
    </thead>
    <tbody>
        <tr><td>ReduceLROnPlateau</td><td><code>mode</code> = max, <code>factor</code> = 0.1, <code>patience</code> = 3, <code>threshold</code> = 1e-3</td></tr>
    </tbody>
</table>

Below is the wandb log of training the ResNet model:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/cifar/resnet.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fig 1. wandb logs from resnet training.
</div>

### Vision Transformer
The final implementation harnesses the expressive capabilities of transformers, especially with its utilization of self-attention {% cite dosovitskiy2021imageworth16x16words -f projects/cifar %}. Note that instead of patchifying the image and linear projecting, a convolutional layer is applied to obtain patch embeddings. This modification helps "increase optimization stability and also improves peak performance" as described in {% cite xiao2021earlyconvolutionshelptransformers -f projects/cifar %}.

This project consists of both (1) fine-tuning a vision transformer pretrained on ImageNet and (2) training a vision transformer from scratch.

Using the hyperparameters below, the pretrained vision transformer can be fine tuned to achieve ~97.6% test accuracy (cross-validated) on CIFAR-10.

<table id="table" data-toggle="table" class="mb-3" style="width: 20%">
    <thead>
        <tr>
            <th data-field="hyperparameter">Hyperparameter</th>
            <th data-field="value">Value</th>
        </tr>
    </thead>
    <tbody>
        <tr><td>EPOCHS</td><td>10</td></tr>
        <tr><td>BATCH_SIZE</td><td>32</td></tr>
        <tr><td>LEARNING_RATE</td><td>1e-4</td></tr>
    </tbody>
</table>

<table id="table" data-toggle="table" class="mb-3" style="width: 60%">
    <thead>
        <tr>
            <th data-field="optimizer">Optimizer</th>
            <th data-field="parameters">Parameters</th>
        </tr>
    </thead>
    <tbody>
        <tr><td>Adam</td><td><code>momentum</code> = 0.9, <code>weight_decay</code> = 1e-7</td></tr>
    </tbody>
</table>

<table id="table" data-toggle="table" class="mb-3" style="width: 30%">
    <thead>
        <tr>
            <th data-field="scheduler">Scheduler</th>
            <th data-field="parameters">Parameters</th>
        </tr>
    </thead>
    <tbody>
        <tr><td>CosineAnnealingLR</td><td><code>T_max</code> = 10</td></tr>
    </tbody>
</table>

The same hyperparameters are used to train a vision transformer from scratch except the learning rate is reduced to 1e-5, a different learning rate scheduler was used, and longer training time (details to be added soon).

Below is the wandb log of losses and learning rate for both of these training sessions (fine tune and from scratch):

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/cifar/vit.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fig 1. wandb logs from vision transformer training.
</div>