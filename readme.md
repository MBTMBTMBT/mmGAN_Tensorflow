####
# **MRI_GAN Reproduction**
Edited: Sep 27th, 2021
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

##How to Run
This is a demo of how to run the model, including preprocessing,
training and testing.
In the demo we will use BRATS2018 dataset, the dataset contains 
groups of patients of HGG and LGG. Patients in HGG group will be 
separated into 5 sub-groups (folds) by random, one group will be used
for validation purpose and the other four will be used for training.

###Environment
The anaconda environment for this project is prepared, and can be
imported with:
```
conda env create -f mm_gan.yaml
```
However, it is noticeable that tensorflow might have version 
requirements on CUDA and CUDNN. This environment works with CUDA 11.4 
and CUDNN 11.4.

###Preprocessing
Folder of each patient in the dataset contains 5 nii files, each can be
read as a (240, 240, 155) tensor, that is, 155 slices of (240, 240)
images. There are the four channels of the MRI
sequence: T1, T2, T1ce and Flair, as well as one names segmask, which 
is used as labels for tumor segmentation. During preprocessing, each
image will be padded into (256, 256) in order to fit MM-GAN. For 
reproduction of the original paper, the image should not be padded,
but resized directly, however, in order to fit the requirement of 
DeepMedic network at down stream, the brain image should be limited 
within (240, 240) range. It turns out that there is no significant 
performance loss according to MSE, PSNR and SSIM, for padding images
compared with resizing them (the measured performance in fact seems 
slightly better than the claimed ones, which might due to larger dark
area in the image).
It is standardized by letting each pixel of each channel of each 
patient divided by the mean of this channel; and when computing this 
mean, only the pixels in a smaller range of the image, which closely
surrounds the brain count 
((29:223, 41:196), which was used in Sharma 2020 as the outline to 
magnify the brain image). This will decrease the effect of great 
amount of zeros that lowers the mean.
An additional step is to cut some slices with pure
dark (zeros). It is noticeable that the further training and testing 
with DeepMedic model will be performed with this preprocessed dataset,
so if cutting off any slices from each channel, the same operation 
should also be done with the segmask.

As you can see in run.py, preprocessing includes a few parts: 
1. **divides the datasets into 5 groups by random;**
2. **then run the mentioned data preprocessing process;**
3. **for convenience, write the
nii files into Tensorflow TF-record files** (using for both TF and 
PyTorch version), using TF Dataset API
will not cause any change for PyTorch version model, 
the pytorch model is still the same as the original;
but the training speed is improved, as PyTorch DataLoader might
have compatibility issue for running with multiple processes on some 
Windows machines. 
4. **move segmasks into the new folder, and cut off some slices** 
(this is to support Testing in DeepMedic)

To do preprocessing, please run command with following format:
```
python run.py --preprocess cfg_demo/preprocess_cfg_demo.cfg
```
where in the configuration file, it contains some parameter used 
during preprocessing.

###Training
To train the model, with either PyTorch or Tensorflow2 models, 
TF-record files must be provided, which is used for DataSet API
in tensorflow2 to load the data with high efficiency. If preprocessing
methods are run properly, five TF-record files will be generated 
automatically at the configured output directory, each containing a 
group of patients. The saved parameters of the U-Net and adversarial 
networks will be written and read automatically. Only he number of 
total epochs is needed to be given, even if the training process 
is broken at half, the programme will pick up the checkpoint and 
continue the rest epochs. Therefore, it is very important to remain
two generated txt log files unchanged, since the programme relies on
them for counting epochs. Please follow the demo config files to 
create your own configurations, and do not change the parameter names.
The shape of the input image is set to (256, 256), any smaller shape
might cause errors for convolution.

During training, each epoch is separated into several sub-epochs, 
and a small sample of validation group will be used to run a rough 
sub-validation; and during a certain number of epochs (10 epochs), 
there will be a doom validation, that all the slices of validation
group will be tested to give a strong result of the current model.
The predicted output of training process will not be stored, this is
different from the test process.

Additionally, Torch version training process allows running with CPU,
yet due to dimension issue during convolution, the TF version can only
run on GPU.

To run the training process, please run command with following 
format, for Torch model:
```
python run.py --torchtrain cfg_demo/train_cfg_demo.cfg
```
for TF model:
```
python run.py --tftrain cfg_demo/train_cfg_demo.cfg
```
For both models, the configure file can be the same. In fact, 
they use the same function name and parameters in the python module,
so that they can be called in same way if necessary. The Torch 
version generally runs twice as fast as the TF version that runs in
eager training mode.

To view the training process with tensorboard, use:
```
tensorboard --logdir=./comb0_demo/tensorboard
```
where this path should point at output directory of the session.

###Testing
During testing process, patients stored in the given TF-record file
will be extracted and re-organised according to different scenarios
(e.g., 1100 means that T1ce and Flair are lost, but T1 and T2 remain),
the synthesised images will be stored within several nii files, and
the evaluation results will be kept.

To tun test process, please run command with following 
format, for Torch model:
```
python run.py --torchtest cfg_demo/test_cfg_demo.cfg
```
for TF model:
```
python run.py --tftest cfg_demo/test_cfg_demo.cfg
```
Similar to training, two test functions can share the same config 
file.
