# GeoWGAN-GP
Code for training a WGAN-GP to reconstruct two-dimensional backscatter electron microscopy images of geological rock samples. More details can be found in **Quantifying complex microstructures of earth material: Reconstructing higher-order spatial correlations using deep generative adversarial networks**.

Original data and data for reproducing figures in the publication can be found in Yoda repository of Utrecht University available on:
% here the link to Yoda.

## Data
Large segmented backscattered electron (BSE) images of both meta-igneous and serpentinte samples can be found in the Original_images folder. These images are used to create training images of size 128 by 128 pixels (default image size).
Give the path to one of these images upon running training code (train.py).

## How to run the code?
* To run the code for training WGAN-GP type: train.py  -ip " path to the large segmented image in the Original_images folder".
* Other paramter is '-project_name' which is Meta-igneous by default but you can define your own name. it will create a folder where your training images will be saved.
* If you would like to use larger images for training, you can change
* Default image size for reconstruction is 128 by 128 pixels. if you want to train WGAN-GP on larger images you should pass argumnt '-isize' it is set to 128 by default.
Networks' architectures will automatically be updated to create images of this size. Note that isize should a multliple 16.
