####
# **MRI_GAN Reproduction**
Edited: Jan 29th, 2021
***
## Introduction
This repository contains the reproduction code for MM-GAN: 
[Missing MRI Pulse Sequence Synthesis using Multi-Modal Generative Adversarial Network](https://ieeexplore.ieee.org/document/8859286).
The scripts for preprocessing and training are modified with exact same method of the original
but focus only on BRATS2018 HGG and LGG dataset.

The original paper:
```
@ARTICLE{Sharma20,
  author={A. {Sharma} and G. {Hamarneh}},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Missing MRI Pulse Sequence Synthesis Using Multi-Modal Generative Adversarial Network}, 
  year={2020},
  volume={39},
  number={4},
  pages={1170-1183},}
```

As well as the original repository:
```
@misc{Sharma20Code,
  author = {A. {Sharma}},
  title = {MM-GAN: Missing MRI Pulse Sequence Synthesis using Multi-Modal Generative Adversarial Network},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/trane293/mm-gan}}
}
```

## How to Run
An Anaconda environment would be provided latter, before that, the expected modules and versions
are listed below:
```
    python 3.8 --- (all python 3.x versions should work);
    cuda 11.0 (with cudnn installed) & torch 1.7.1+cu110;
    (and a Nvidia GPU with memory >= 6GB);
    torchvision 0.7.0+cpu;
    h5py        2.10.0;
    SimpleITK   2.0.0;
    numpy       1.17.3;
    matplotlib  3.2.2;
```
### Setting File paths
The default top-level path of the dateset is set in the code directory, and the output dir is
also set under this top level path, which could both be found in `settings.py`
```
    TOP_LEVEL_PATH = os.getcwd() + r'/BRATS2018'
    DATASET_PATH = os.path.join(TOP_LEVEL_PATH, 'out')
```
Under the dataset dir `/BRATS2018` there are expected to be two sub folders: `/Training` and `/Talidation`, in
addition, both HGG and LGG folders should lie under `/Training`. 
For convenience, a sample directory is provided in this repository, and the directory structure should be the same
with it yet with a much larger number of patients for training, testing and validating.
### Preprocessing
When the patients are in position, execute `preprocess.py`, which will first generate three .h5 files for original
data unpacked from the .gz files. Later, these original images will be cropped and restored in another three .h5 files, 
and their mean and variance values will be computed and saved at the same time. Notice that at preprocess part, it doesn't
matter how many patients are listed, the hdf5 files will always be generated with the exact number of them, however, it 
still needs to be set how many of them will be used for training and testing.
### Training
The dataset to train can be selected by changing the parameter of the `train` function between 'HGG' and 'LGG', notice that
the case of the letter matters, so it would be best to keep using upper cases.
```
    if __name__ == '__main__':
    train(DATASET_PATH, 'HGG', TRAIN_PATIENTS_HGG, TEST_PATIENTS_HGG)
    train(DATASET_PATH, 'LGG', TRAIN_PATIENTS_LGG, TEST_PATIENTS_LGG)
```
