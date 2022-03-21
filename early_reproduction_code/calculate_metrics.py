import h5py as h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import platform
from math import log10
import csv
import shutil
import matplotlib

from helpers import psnr_torch
from settings import *

if 'cdr' in platform.node() or 'ts' in platform.node():
    import matplotlib.pyplot

    matplotlib.pyplot.switch_backend('agg')
# from skimage.measure import compare_ssim
from skimage import measure
from matplotlib import pylab
import matplotlib.pyplot as plt
import os
import numpy.ma as ma
import pickle as pickle
import numpy as np
from skimage.transform import resize
import itertools
import torch.nn as nn
import copy
import pytorch_msssim as pyt_ssim

# get logger
logger = LOGGER


def calculate_metrics(G, patient_list,
                      save_path, all_scenarios,
                      epoch, curr_scenario_range=None,
                      batch_size_to_test=2,
                      impute_type=None,
                      convert_normalization=False,
                      save_stats=False,
                      mean_var_file=None,
                      use_pytorch_ssim=False, seq_type='T1', cuda=True):
    """
    For ISLES2015
    all_scenarios: scenarios
    curr_scenario_range: None
    batch_to_test: 2
    """

    if isinstance(patient_list, list):
        patients = patient_list
    else:
        patients = [patient_list]
    # put the generator in EVAL mode.
    mse = nn.MSELoss()
    if cuda:
        mse.cuda()

    # save_im_path = os.path.join(save_path, 'all_slices', 'epoch_{}'.format(epoch))
    save_im_path = os.path.join(save_path, 'all_slices', 'slice' + str(SAVE_IMG_SLICE))
    if not os.path.isdir(save_im_path):
        os.makedirs(save_im_path)
    logger.info("save path: %s" % save_im_path)

    device = torch.device("cuda:0" if cuda else "cpu")
    logger.debug("device: ", end='')
    logger.debug(device)

    # contains metrics for EACH slice from EACH OF THE SCENARIO. Basically everything. This is what we need for
    # ISLES2015
    running_mse = {}
    running_psnr = {}
    running_ssim = {}

    # move all the patients into one tensor
    patient_shape = patients[0].shape
    patients_mat = np.empty((len(patients), patient_shape[0], patient_shape[1], ))

    for (pat_ind, patient) in enumerate(patients):
        # pat_name = patient['name'].decode('UTF-8')
        logger.info('Testing Patient: {}'.format(pat_ind))
        patient_image = patient['image']

        patient_copy = patient['image'].clone()

        patient_numpy = patient_copy.detach().cpu().numpy()

        scenarios = all_scenarios
        all_minus_1_g = None
        all_minus_x_test_r = None
        if cuda:
            all_minus_1_g = torch.ones((batch_size_to_test, 1, SPATIAL_SIZE_FOR_TRAINING[0],
                                                 SPATIAL_SIZE_FOR_TRAINING[1])).cuda() * -1.0
            all_minus_x_test_r = torch.ones((batch_size_to_test, SPATIAL_SIZE_FOR_TRAINING[0],
                                                 SPATIAL_SIZE_FOR_TRAINING[1])).cuda() * -1.0
        else:
            all_minus_1_g = torch.ones((batch_size_to_test, 1, SPATIAL_SIZE_FOR_TRAINING[0],
                                                 SPATIAL_SIZE_FOR_TRAINING[1])) * -1.0
            all_minus_x_test_r = torch.ones((batch_size_to_test, SPATIAL_SIZE_FOR_TRAINING[0],
                                                 SPATIAL_SIZE_FOR_TRAINING[1])) * -1.0

        sh = patient_numpy.shape

        batch_size = batch_size_to_test

        # this will store output for ALL patients

        if curr_scenario_range is not None:
            scenarios = scenarios[curr_scenario_range[0]:curr_scenario_range[1]]

        logger.info('Testing on scenarios: {}'.format(scenarios))
        for curr_scenario in scenarios:

            curr_scenario_str = ''.join([str(x) for x in curr_scenario])

            running_mse[curr_scenario_str] = []
            running_psnr[curr_scenario_str] = []
            running_ssim[curr_scenario_str] = []

            logger.info('Testing on scenario: {}'.format(curr_scenario))

            # get the batch indices
            batch_indices = range(0, sh[0], batch_size)
            # print(batch_indices)

            # for each batch
            for _num, batch_idx in enumerate(batch_indices):
                x_test_r = None
                x_test_z = None
                if cuda:
                    x_test_r = patient_image[batch_idx:batch_idx + batch_size, ...].cuda()
                    x_test_z = x_test_r.clone().cuda().type(torch.cuda.FloatTensor)
                else:
                    x_test_r = patient_image[batch_idx:batch_idx + batch_size, ...]
                    x_test_z = x_test_r.clone().type(torch.FloatTensor)

                x_test_r = x_test_r.to(device)
                x_test_z = x_test_z.to(device)

                if impute_type == 'noise':
                    impute_tensor = torch.randn((batch_size,
                                                 SPATIAL_SIZE_FOR_TRAINING[0],
                                                 SPATIAL_SIZE_FOR_TRAINING[1]), device=device)

                elif impute_type == 'average':
                    avail_indx = [i for i, x in enumerate(curr_scenario) if x == 1]
                    impute_tensor = torch.mean(x_test_r[:, avail_indx, ...], dim=1)
                elif impute_type == 'zeros':
                    impute_tensor = torch.zeros((batch_size,
                                                 SPATIAL_SIZE_FOR_TRAINING[0],
                                                 SPATIAL_SIZE_FOR_TRAINING[1]), device=device)
                else:
                    impute_tensor = torch.ones((sh[-2], sh[-1]), device=device) * -1.0
                    # print('Imputing with -1')

                # print('Imputing with {}'.format(impute_type))
                for idx_, k in enumerate(curr_scenario):
                    if k == 0:
                        x_test_z[:, idx_, ...] = impute_tensor

                G_result = G(x_test_z)

                # save pic
                if _num == SAVE_IMG_SLICE:
                    for channel in range(4):
                        pic_path = "test_patient_%d_channel_%d_slice%d__epoch_%d.png" % (pat_ind, channel, _num, epoch)
                        scenario_str = str(curr_scenario)
                        pic_dir = os.path.join(save_im_path, scenario_str)
                        if not os.path.isdir(pic_dir):
                            os.mkdir(pic_dir)
                        pic_path = os.path.join(save_im_path, scenario_str, pic_path)
                        matplotlib.pyplot.imsave(pic_path, (G_result.cpu().detach().numpy())[0, channel], cmap='gray')

                # save all images
                # consider BRATS2018 only
                for idx_curr_label, j in enumerate(curr_scenario):
                    if j == 0:
                        running_mse[curr_scenario_str].append(
                            mse(G_result[:, idx_curr_label] /
                                (torch.max(G_result[:, idx_curr_label]) + 0.0001),
                                x_test_r[:, idx_curr_label] /
                                (torch.max(x_test_r[:, idx_curr_label]) + 0.0001)).item())

                        running_ssim[curr_scenario_str].append(pyt_ssim.ssim(
                            G_result[:, idx_curr_label].unsqueeze(1) /
                            (torch.max(G_result[:, idx_curr_label]) + 0.0001),
                            x_test_r[:, idx_curr_label].unsqueeze(1) /
                            (torch.max(x_test_r[:, idx_curr_label]) + 0.0001),
                            val_range=1).item())

                        running_psnr[curr_scenario_str].append(
                            psnr_torch(G_result[:, idx_curr_label],
                                       x_test_r[:, idx_curr_label], cuda=cuda).item())

    num_dict = {}
    all_mean_mse = []
    all_mean_psnr = []
    all_mean_ssim = []

    for (mse_key, mse_list, psnr_key, psnr_list, ssim_key, ssim_list) in zip(running_mse.keys(), running_mse.values(),
                                                                             running_psnr.keys(), running_psnr.values(),
                                                                             running_ssim.keys(),
                                                                             running_ssim.values()):
        assert mse_key == ssim_key == psnr_key
        num_dict[mse_key] = {
            'mse': np.mean(mse_list),
            'psnr': np.mean(psnr_list),
            'ssim': np.mean(ssim_list)
        }

        all_mean_mse += mse_list
        all_mean_psnr += psnr_list
        all_mean_ssim += ssim_list

    num_dict['mean'] = {
        'mse': np.mean(all_mean_mse),
        'psnr': np.mean(all_mean_psnr),
        'ssim': np.mean(all_mean_ssim)
    }
    if save_stats:
        stat_folder = os.path.join(save_path, "stats/".format())
        if not os.path.isdir(stat_folder):
            os.makedirs(stat_folder)
        logger.info('Saving running statistics to folder: {}'.format(stat_folder))
        # save mse, psnr and ssim
        pickle.dump(running_mse, open(os.path.join(stat_folder, 'mse.p'), 'wb'))
        pickle.dump(running_psnr, open(os.path.join(stat_folder, 'psnr.p'), 'wb'))
        pickle.dump(running_ssim, open(os.path.join(stat_folder, 'ssim.p'), 'wb'))

        return num_dict, running_mse, running_psnr, running_ssim

    return num_dict
