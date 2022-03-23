# GeoWGAN-GP
Code for training a WGAN-GP to reconstruct two-dimensional backscatter electron microscopy images of geological rock samples. More details can be found in **Quantifying complex microstructures of earth material: Reconstructing higher-order spatial correlations using deep generative adversarial networks**.

Authors: [Hamed Amiri](https://www.researchgate.net/profile/Hamed-Amiri-10), [Ivan Vasconcelos](https://www.uu.nl/medewerkers/IPiresdeVasconcelos), [Yang Jiao](https://isearch.asu.edu/profile/1970397), [Pei-En Chen](https://www.researchgate.net/profile/Pei-En-Chen-2), [Oliver Plümper](https://www.uu.nl/staff/OPlumper)

Original data and data for reproducing figures in the publication can be found in Yoda repository of Utrecht University available on:
% here the link to Yoda.

## Data
Large segmented backscattered electron (BSE) images of both meta-igneous and serpentinte samples can be found in the Original_images folder. These images are used to create training images of size 128 by 128 pixels (default image size).
Give the path to one of these images upon running training code (train.py).

## How to run the code?
* To run the code for training WGAN-GP type: train.py  -ip " path to the large segmented image in the Original_images folder".
* Other paramter is '-project_name' which is Meta-igneous by default but you can define your own name. it will create a folder where your training images will be saved.
* Default image size for reconstruction is 128 by 128 pixels. if you want to train WGAN-GP on larger images you should pass argumnt '-isize' it is set to 128 by default.
Networks' architectures will be utomatically updated to create images of this size. Note that isize should be a multliple 16.

## Remarks
* MSE between two point correlation of real and reconstructed images are used as a accuracy metric and also for saving the models. By default, when mse is less than 5e-7, model is saved in a new folder called "checkpoints" in which there will be a subfolder to save the best model with smallest mse.
* Two-point correlation and autoscaled correlations plots calculated from real and reconstructed images are also save in output folder.
* Both mse and two-point correlation functions are calculated on 128 images (=batch size) and average values are considered.

## Requirements
Following libraries and versions have been used in this repository.
* Python 3.8.12
* PyTorch 1.10.12
* Numpy 1.19.5
* Numba 0.54.0
* Pandas 1.3.5
* Scikit-image 0.18.3
* Tifffile 2021.11.2
