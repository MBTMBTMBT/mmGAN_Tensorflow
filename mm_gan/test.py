import pickle
import sys
import logging
import os.path
import json
import SimpleITK as sitk
import tqdm
import itertools
import numpy as np
import tensorflow as tf
import mm_gan.io_helpers as io_helpers
import time
import multiprocessing
from mm_gan.constants_and_tools import make_dir
from tensorflow.keras.losses import MeanSquaredError
from mm_gan.models import get_generator_unet, get_discriminator
from mm_gan.tf_train import impute_reals_into_fake
import os
import re
from skimage.metrics import structural_similarity as sk_ssim

# import argparse

gpus = tf.config.experimental.list_physical_devices('GPU')
'''tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])'''
# tf.config.experimental.set_memory_growth(gpus[0], True)


def tf_test(parameter_path: str, session_name: str, output_dir: str,
            tfrecords: list, channels: list, img_shape: tuple) -> dict:
    generator = get_generator_unet(input_shape=(
        4, img_shape[0], img_shape[1]), out_channels=4)
    discriminator = get_discriminator(input_shape=(
        4, img_shape[0], img_shape[1]), out_channels=4)
    checkpoint = tf.train.Checkpoint(generator=generator, discriminator=discriminator)
    checkpoint.restore(parameter_path)

    # create sc scenarios from 0000 to 1111, yet 0000 and 1111 should not be used
    scenarios = list(map(list, itertools.product([0, 1], repeat=4)))
    scenarios.remove([0, 0, 0, 0])
    scenarios.remove([1, 1, 1, 1])

    # sort the scenarios according to decreasing difficulty. Easy scenarios last, and difficult ones first.
    scenarios.sort(key=lambda x: x.count(1))

    test_dataset = io_helpers.get_dataset(tfrecords=tfrecords, compression_type='ZLIB', batch_size=1,
                                          drop_remainder=True,
                                          shuffle=False, buffer_size=0)
    patients = 0
    slices = 0
    pat = r"\d+"
    for each_file in tfrecords:
        num_in_names = re.findall(pat, os.path.split(each_file)[-1])
        patients += int(num_in_names[1])
        slices += int(num_in_names[2])

    output_dir = os.path.join(output_dir, session_name)
    make_dir(output_dir)
    make_dir(os.path.join(output_dir, "log"))

    formatter = logging.Formatter(
        '%(asctime)s: %(message)s', "%y%b%d-%H:%M:%S")
    test_logger = logging.getLogger(name=session_name + "_val")
    test_logger.setLevel(logging.INFO)
    log_path_val = os.path.join(os.path.join(
        output_dir, "log"), session_name + "_val" + ".txt")
    file_handler = logging.FileHandler(log_path_val, mode='a')
    file_handler.setFormatter(formatter)
    test_logger.addHandler(file_handler)

    mse = MeanSquaredError()

    assert slices % patients == 0
    slices_per_patient = slices // patients
    pbar = tqdm.tqdm(total=patients, desc="Testing on patients:")
    test_iter = test_dataset.__iter__()

    for patient_idx in range(patients):
        metrics = []
        slices = []
        for i in range(len(scenarios)):
            metrics.append({'mse': [], 'psnr': [], 'ssim': []})
            slices.append([])

        for idx in range(slices_per_patient):
            each_batch = test_iter.__next__()
            each_patient = each_batch[0].numpy().item().decode()
            t1 = each_batch[1]
            t2 = each_batch[2]
            t1ce = each_batch[3]
            flair = each_batch[4]
            batch = tf.concat([t1, t2, t1ce, flair], axis=1).numpy()

            for sc_idx, each_scenario in enumerate(scenarios):
                batch_z = batch.copy()

                for s in range(batch.shape[0]):
                    for c, each_channel in enumerate(each_scenario):
                        if each_channel == 0:
                            batch_z[s, c] = np.zeros(
                                shape=(batch.shape[2], batch.shape[3]), dtype=np.float32)

                # with device(dev):
                fake_img = generator(batch_z, training=False)
                fake_img = impute_reals_into_fake(
                    batch_z, fake_img, [each_scenario])
                slices[sc_idx].append(fake_img)
                # scenario_metrics = {'mse': [], 'psnr': [], 'ssim': []}
                for c, each_channel in enumerate(each_scenario):
                    if each_channel == 0:
                        fake_channel_arr = fake_img[:, c, ...]
                        real_channel_arr = batch[:, c, ...]
                        # max_val = max(fake_channel_arr.numpy().max(), real_channel_arr.numpy().max())
                        max_val = 1.0
                        metrics[sc_idx]['mse'].append(
                            mse(fake_channel_arr, real_channel_arr).numpy().item())
                        metrics[sc_idx]['psnr'].append(
                            tf.image.psnr(fake_channel_arr, real_channel_arr, max_val).numpy().item())
                        metrics[sc_idx]['ssim'].append(
                            tf.image.ssim(fake_channel_arr, real_channel_arr, max_val, filter_size=1).numpy().item())

        epoch_rst_dict = {}
        test_logger.info("testing patient %s" % each_patient)
        for sc_idx, scenario in enumerate(scenarios):
            scenario_str = ''
            for each in scenario:
                scenario_str += str(each)
            epoch_rst_dict[scenario_str] = {
                "mse": np.array(metrics[sc_idx]['mse']).mean(),
                "psnr": np.array(metrics[sc_idx]['psnr']).mean(),
                "ssim": np.array(metrics[sc_idx]['ssim']).mean(),
            }
            test_logger.info("%s: mse: %f; psnr: %f, ssim: %f" % (scenario_str, epoch_rst_dict[scenario_str]["mse"],
                                                                  epoch_rst_dict[scenario_str]["psnr"],
                                                                  epoch_rst_dict[scenario_str]["ssim"]))
            s_slices = slices[sc_idx]
            s_slices = np.squeeze(np.array(s_slices)).transpose(1, 0, 2, 3)
            for i in range(4):
                c_slices = s_slices[i]
                channel = channels[i]
                zero_voxel = c_slices[0, 0, 0]
                c_slices[c_slices < zero_voxel] = zero_voxel
                c_slices = c_slices[:, 8:248, 8:248]
                img = sitk.GetImageFromArray(c_slices)
                file_path = os.path.join(output_dir, each_patient)
                file_path = os.path.join(file_path, scenario_str)
                make_dir(file_path)
                file_path = os.path.join(file_path, "%s.nii.gz" % channel)
                if each_patient == "Brats18_CBICA_AZD_1":
                    pass
                make_dir(file_path)
                time.sleep(0.1)
                sitk.WriteImage(img, file_path)

        pbar.update()  # update the bar

        # json
        js_obj = json.dumps(epoch_rst_dict)
        js_path = os.path.join(output_dir, 'test-%s.json' % each_patient)
        js_file = open(js_path, 'w')
        js_file.write(js_obj)
        js_file.close()

    pbar.close()
    return epoch_rst_dict


def torch_test(parameter_path: str, session_name: str, output_dir: str,
               tfrecords: list, channels: list, img_shape: tuple) -> dict:
    import torch
    from mm_gan import pytorch_msssim as pyt_ssim
    from mm_gan.torch_train import mse, psnr_torch
    import mm_gan.torch_models as torch_models
    import mm_gan.torch_train as torch_train

    def mse_torch(slice_a: torch.tensor, slice_b: torch.tensor) -> float:
        return (torch.square(torch.subtract(slice_a, slice_b))).mean()

    # check if GPU is available
    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda:0" if cuda else "cpu")

    # define tensor type
    tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])

    output_dir = os.path.join(output_dir, session_name)
    make_dir(output_dir)
    make_dir(os.path.join(output_dir, "log"))

    formatter = logging.Formatter(
        '%(asctime)s: %(message)s', "%y%b%d-%H:%M:%S")
    test_logger = logging.getLogger(name=session_name + "_val")
    test_logger.setLevel(logging.INFO)
    log_path_val = os.path.join(os.path.join(
        output_dir, "log"), session_name + "_val" + ".txt")
    file_handler = logging.FileHandler(log_path_val, mode='a')
    file_handler.setFormatter(formatter)
    test_logger.addHandler(file_handler)

    # get networks
    generator = torch_models.GeneratorUNet(
        in_channels=4, out_channels=4, with_relu=True, with_tanh=False)

    if cuda:
        generator = torch.nn.DataParallel(generator.cuda())
    else:
        generator = torch.nn.DataParallel(generator)

    if os.path.isfile(parameter_path):
        test_logger.info("Loading checkpoint '{}'".format(parameter_path))
        checkpoint = torch.load(
            parameter_path, pickle_module=pickle, map_location=device)
        generator.load_state_dict(checkpoint['state_dict'])
    else:
        test_logger.critical(
            'Checkpoint {} does not exist.'.format(parameter_path))

    # create sc scenarios from 0000 to 1111, yet 0000 and 1111 should not be used
    scenarios = list(map(list, itertools.product([0, 1], repeat=4)))
    scenarios.remove([0, 0, 0, 0])
    scenarios.remove([1, 1, 1, 1])

    # sort the scenarios according to decreasing difficulty. Easy scenarios last, and difficult ones first.
    scenarios.sort(key=lambda x: x.count(1))

    test_dataset = io_helpers.get_dataset(tfrecords=tfrecords, compression_type='ZLIB', batch_size=1,
                                          drop_remainder=True,
                                          shuffle=False, buffer_size=0)
    patients = 0
    slices = 0
    pat = r"\d+"
    for each_file in tfrecords:
        num_in_names = re.findall(pat, os.path.split(each_file)[-1])
        patients += int(num_in_names[1])
        slices += int(num_in_names[2])

    assert slices % patients == 0
    slices_per_patient = slices // patients
    pbar = tqdm.tqdm(total=patients, desc="Testing on patients:")
    test_iter = test_dataset.__iter__()

    for patient_idx in range(patients):
        metrics = []
        slices = []
        for i in range(len(scenarios)):
            metrics.append({'mse': [], 'psnr': [], 'ssim': []})
            slices.append([])

        for idx in range(slices_per_patient):
            each_batch = test_iter.__next__()
            each_patient = each_batch[0].numpy().item().decode()
            t1 = each_batch[1]
            t2 = each_batch[2]
            t1ce = each_batch[3]
            flair = each_batch[4]
            batch = tf.concat([t1, t2, t1ce, flair], axis=1).numpy()

            if cuda:
                batch = torch.from_numpy(batch).cuda().type(tensor)
            else:
                batch = torch.from_numpy(batch).type(tensor)

            for sc_idx, each_scenario in enumerate(scenarios):
                batch_z = batch.clone()

                for s in range(batch.shape[0]):
                    for c, each_channel in enumerate(each_scenario):
                        if each_channel == 0:
                            if cuda:
                                batch_z[s, c] = torch.zeros(
                                    batch.shape[2], batch.shape[3]).cuda().type(tensor)
                            else:
                                batch_z[s, c] = torch.zeros(
                                    batch.shape[2], batch.shape[3]).type(tensor)

                # with device(dev):
                fake_img = generator(batch_z).detach()
                fake_img = torch_train.impute_reals_into_fake(
                    batch_z, fake_img, [each_scenario])
                if cuda:
                    fake_img_np = fake_img.cpu().numpy()
                else:
                    fake_img_np = fake_img.numpy()
                slices[sc_idx].append(fake_img_np)
                # scenario_metrics = {'mse': [], 'psnr': [], 'ssim': []}
                for c, each_channel in enumerate(each_scenario):
                    if each_channel == 0:
                        fake_channel_arr = torch.unsqueeze(
                            fake_img[:, c, ...], 2)
                        real_channel_arr = torch.unsqueeze(batch[:, c, ...], 2)
                        # max_val = max(fake_channel_arr.numpy().max(), real_channel_arr.numpy().max())
                        # max_val = 1.0
                        mse_rst = mse_torch(fake_channel_arr / (torch.max(fake_channel_arr) + 0.0001),
                                            real_channel_arr / (torch.max(real_channel_arr) + 0.0001))
                        # psnr_rst = psnr(fake_channel_arr, real_channel_arr, max_val)
                        psnr_rst = psnr_torch(
                            fake_channel_arr, real_channel_arr)
                        #ssim_rst = pyt_ssim.ssim(
                        #    fake_channel_arr / (torch.max(fake_channel_arr) + 0.0001),
                        #    real_channel_arr / (torch.max(real_channel_arr) + 0.0001),
                        #    val_range=1,
                        #)

                        # important: can only take one slice a time!
                        if cuda:
                            ssim_rst = sk_ssim(
                                np.array(torch.squeeze(fake_channel_arr / (torch.max(fake_channel_arr) + 0.0001)).cpu()),
                                np.array(torch.squeeze(real_channel_arr / (torch.max(real_channel_arr) + 0.0001)).cpu()),
                                win_size=11,
                            )
                        else:
                            ssim_rst = sk_ssim(
                                np.array(torch.squeeze(fake_channel_arr / (torch.max(fake_channel_arr) + 0.0001))),
                                np.array(torch.squeeze(real_channel_arr / (torch.max(real_channel_arr) + 0.0001))),
                                win_size=11,
                            )
                        if cuda:
                            mse_rst = mse_rst.cpu().numpy()
                            psnr_rst = psnr_rst.cpu().numpy()
                            # ssim_rst = ssim_rst.cpu().numpy()
                        else:
                            mse_rst = mse_rst.numpy()
                            psnr_rst = psnr_rst.numpy()
                            # ssim_rst = ssim_rst.numpy()
                        metrics[sc_idx]['mse'].append(mse_rst.item())
                        metrics[sc_idx]['psnr'].append(psnr_rst.item())
                        metrics[sc_idx]['ssim'].append(ssim_rst.item())

        epoch_rst_dict = {}
        test_logger.info("testing patient %s" % each_patient)
        for sc_idx, scenario in enumerate(scenarios):
            scenario_str = ''
            for each in scenario:
                scenario_str += str(each)
            epoch_rst_dict[scenario_str] = {
                "mse": np.array(metrics[sc_idx]['mse']).mean(),
                "psnr": np.array(metrics[sc_idx]['psnr']).mean(),
                "ssim": np.array(metrics[sc_idx]['ssim']).mean(),
            }
            test_logger.info("%s: mse: %f; psnr: %f, ssim: %f" % (scenario_str, epoch_rst_dict[scenario_str]["mse"],
                                                                  epoch_rst_dict[scenario_str]["psnr"],
                                                                  epoch_rst_dict[scenario_str]["ssim"]))
            s_slices = slices[sc_idx]
            s_slices = np.squeeze(np.array(s_slices)).transpose(1, 0, 2, 3)
            for i in range(4):
                c_slices = s_slices[i]
                channel = channels[i]
                zero_voxel = c_slices[0, 0, 0]
                c_slices[c_slices < zero_voxel] = zero_voxel
                c_slices = c_slices[:, 8:248, 8:248]
                img = sitk.GetImageFromArray(c_slices)
                file_path = os.path.join(output_dir, each_patient)
                file_path = os.path.join(file_path, scenario_str)
                make_dir(file_path)
                file_path = os.path.join(file_path, "%s.nii.gz" % channel)
                #if each_patient == "Brats18_CBICA_AZD_1":
                #    pass
                make_dir(file_path)
                time.sleep(0.1)
                sitk.WriteImage(img, file_path)

        pbar.update()  # update the bar

        # json
        js_obj = json.dumps(epoch_rst_dict)
        js_path = os.path.join(output_dir, 'test-%s.json' % each_patient)
        js_file = open(js_path, 'w')
        js_file.write(js_obj)
        js_file.close()

    pbar.close()
    return epoch_rst_dict


def slice_to_slice_comparison(nii_path_a: str, nii_path_b: str) -> tuple:
    with tf.device('/cpu:0'):
        mse = tf.losses.MeanSquaredError()
        arr_a = io_helpers.load_nii(nii_path=nii_path_a)
        arr_b = io_helpers.load_nii(nii_path=nii_path_b)
        assert arr_a.shape[0] == arr_b.shape[0]
        slices = arr_a.shape[0]
        max_val = ((np.max(arr_a) - np.min(arr_a)) +
                   (np.max(arr_b) - np.min(arr_b))) / 2
        max_val = 1
        # print("max val: %f" % max_val)
        sum_mse, sum_psnr, sum_ssim = 0, 0, 0
        for i in range(slices):
            a = np.expand_dims(arr_a[i], 2)
            b = np.expand_dims(arr_b[i], 2)
            mse_rst = mse(a, b)
            psnr_rst = tf.image.psnr(a, b, max_val=max_val)
            ssim_rst = tf.image.ssim(a, b, max_val=max_val)
            print("[%d] mse: %f psnr: %f ssim: %f" %
                  (i, mse_rst, psnr_rst, ssim_rst))
            sum_mse += mse_rst
            sum_psnr += psnr_rst
            sum_ssim += ssim_rst
        mean_mse = sum_mse / slices
        mean_psnr = sum_psnr / slices
        mean_ssim = sum_ssim / slices
        print("[mean] mse: %f psnr: %f ssim: %f" %
              (mean_mse, mean_psnr, mean_ssim))
        return mean_mse, mean_psnr, mean_ssim


def fast_slice_to_slice_comparison(nii_path_a: str, nii_path_b: str, workers=4) -> tuple:
    arr_a = io_helpers.load_nii(nii_path=nii_path_a)
    arr_b = io_helpers.load_nii(nii_path=nii_path_b)
    assert arr_a.shape[0] == arr_b.shape[0]
    slices = arr_a.shape[0]
    max_val = ((np.max(arr_a) - np.min(arr_a)) +
               (np.max(arr_b) - np.min(arr_b))) / 2
    max_val = 1
    processes = []
    slice_per_worker = slices // workers
    rsts = multiprocessing.Manager().list()
    sum_mse, sum_psnr, sum_ssim = 0, 0, 0
    for i in range(workers):
        begin = i * slice_per_worker
        end = begin + slice_per_worker
        if i == 7:
            end = slices
        slice_range = (begin, end)
        process = multiprocessing.Process(target=run_comparison,
                                          args=(slice_range, arr_a, arr_b, max_val, rsts))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
    for rst in rsts:
        sum_mse += rst['mse']
        sum_psnr += rst['psnr']
        sum_ssim += rst['ssim']
    mean_mse = sum_mse / slices
    mean_psnr = sum_psnr / slices
    mean_ssim = sum_ssim / slices
    '''print("[mean] mse: %f psnr: %f ssim: %f" %
            (mean_mse, mean_psnr, mean_ssim))'''
    return mean_mse, mean_psnr, mean_ssim


def run_comparison(slice_range: tuple, arr_a: np.ndarray, arr_b: np.ndarray, max_val: float, rsts: list):
    with tf.device('/cpu:0'):
        rst = {}
        sum_mse, sum_psnr, sum_ssim = 0, 0, 0
        # mse = tf.losses.MeanSquaredError()
        for i in range(slice_range[0], slice_range[1]):
            a = np.expand_dims(arr_a[i], 2)
            b = np.expand_dims(arr_b[i], 2)
            mse_rst = mse(a, b)
            psnr_rst = tf.image.psnr(a, b, max_val=max_val)
            ssim_rst = tf.image.ssim(a, b, max_val=max_val)
            '''print("[%d] mse: %f psnr: %f ssim: %f" %
                  (i, mse_rst, psnr_rst, ssim_rst))'''
            sum_mse += mse_rst
            sum_psnr += psnr_rst
            sum_ssim += ssim_rst
        rst['mse'] = sum_mse
        rst['psnr'] = sum_psnr
        rst['ssim'] = sum_ssim
        rsts.append(rst)


def evaluate_nii_to_nii(names_txt: str, test_path: str, scenarios: list, channels: list, out_txt_dir: str,
                        out_txt_name: str):
    make_dir(out_txt_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(
        sys.stdout)  # stderr output to console
    log_path = os.path.join(out_txt_dir, out_txt_name)
    file_handler = logging.FileHandler(log_path, mode='a')
    formatter = logging.Formatter('%(message)s')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.info("start evaluation")
    patients = []
    txt_file = open(names_txt, 'r')
    next_patient = txt_file.readline()
    while next_patient:
        next_patient = next_patient.replace("\n", "")
        patients.append(os.path.split(next_patient)[-1])
        next_patient = txt_file.readline()
    txt_file.close()
    mses = {}
    psnrs = {}
    ssims = {}
    for scenario in scenarios:
        mses[scenario] = []
        psnrs[scenario] = []
        ssims[scenario] = []
    for idx, each_patient in enumerate(patients):
        patient_dir = os.path.join(test_path, each_patient)
        logger.info("=========================")
        logger.info("[%d/%d] patient: %s" %
                    (idx, len(patients) - 1, patient_dir))
        logger.info("scen\tmse\tpsnr\tssim")
        ground_truth = [
            os.path.join(os.path.join(patient_dir, '1000'),
                         '%s.nii.gz' % channels[0]),
            os.path.join(os.path.join(patient_dir, '0100'),
                         '%s.nii.gz' % channels[1]),
            os.path.join(os.path.join(patient_dir, '0010'),
                         '%s.nii.gz' % channels[2]),
            os.path.join(os.path.join(patient_dir, '0001'),
                         '%s.nii.gz' % channels[3]),
        ]
        patient_mses = {}
        patient_psnrs = {}
        patient_ssims = {}
        for scenario in scenarios:
            scenario_dir = os.path.join(patient_dir, scenario)
            scen_mses = []
            scen_psnrs = []
            scen_ssims = []
            for idx, ch in enumerate(scenario):
                if ch == '0':
                    gt_dir = ground_truth[idx]
                    syn_dir = os.path.join(
                        scenario_dir, '%s.nii.gz' % channels[idx])
                    mean_mse, mean_psnr, mean_ssim = fast_slice_to_slice_comparison(
                        gt_dir, syn_dir)
                    scen_mses.append(mean_mse)
                    scen_psnrs.append(mean_psnr)
                    scen_ssims.append(mean_ssim)
            patient_mses[scenario] = np.mean(np.array(scen_mses))
            patient_psnrs[scenario] = np.mean(np.array(scen_psnrs))
            patient_ssims[scenario] = np.mean(np.array(scen_ssims))
            logger.info("%s\t%.4f\t%.4f\t%.4f\t" % (
                scenario, patient_mses[scenario], patient_psnrs[scenario], patient_ssims[scenario]))
            mses[scenario].append(patient_mses[scenario])
            psnrs[scenario].append(patient_psnrs[scenario])
            ssims[scenario].append(patient_ssims[scenario])
    mse_means = {}
    psnr_means = {}
    ssim_means = {}
    mse_stds = {}
    psnr_stds = {}
    ssim_stds = {}
    logger.info("=========================")
    logger.info("conclusion:")
    logger.info("mean")
    logger.info("scen\tmse\tpsnr\tssim")
    for scenario in scenarios:
        mses_arr = np.array(mses[scenario])
        psnrs_arr = np.array(psnrs[scenario])
        ssims_arr = np.array(ssims[scenario])
        mse_means[scenario] = np.mean(mses_arr)
        psnr_means[scenario] = np.mean(psnrs_arr)
        ssim_means[scenario] = np.mean(ssims_arr)
        mse_stds[scenario] = np.std(mses_arr)
        psnr_stds[scenario] = np.std(psnrs_arr)
        ssim_stds[scenario] = np.std(ssims_arr)
        logger.info("%s\t%.4f\t%.4f\t%.4f\t" % (
            scenario, mse_means[scenario], psnr_means[scenario], ssim_means[scenario]))
    logger.info("std")
    logger.info("scen\tmse\tpsnr\tssim")
    for scenario in scenarios:
        logger.info("%s\t%.4f\t%.4f\t%.4f\t" % (
            scenario, mse_stds[scenario], psnr_stds[scenario], ssim_stds[scenario]))
    return mse_means, psnr_means, ssim_means, mse_stds, psnr_stds, ssim_stds


def mse(slice_a: np.ndarray, slice_b: np.ndarray) -> float:
    return (np.square(np.subtract(slice_a, slice_b))).mean()


'''
def psnr(slice_a: np.ndarray, slice_b: np.ndarray, max_val: float) -> float:
    return 10 * np.log(max_val ** 2 / mse(slice_a=slice_a, slice_b=slice_b))
'''


def read_jsons(json_dir: str, scenarios: list):
    files = os.listdir(json_dir)
    mse_rsts = {}
    psnr_rsts = {}
    ssim_rsts = {}
    scenarios.append("mean")
    for scenario in scenarios:
        mse_rsts[scenario] = []
        psnr_rsts[scenario] = []
        ssim_rsts[scenario] = []
    for file in files:
        file = os.path.join(json_dir, file)
        if os.path.isfile(file) and ".json" in file:
            file_obj = open(file, encoding="utf-8")
            patient_dict = json.load(file_obj)
            patient_mses = []
            patient_psnrs = []
            patient_ssims = []
            for scenario in scenarios:
                if scenario == 'mean':
                    mse_rsts[scenario].append(np.array(patient_mses).mean())
                    psnr_rsts[scenario].append(np.array(patient_psnrs).mean())
                    ssim_rsts[scenario].append(np.array(patient_ssims).mean())
                    break
                mse_rsts[scenario].append(patient_dict[scenario]['mse'])
                psnr_rsts[scenario].append(patient_dict[scenario]['psnr'])
                ssim_rsts[scenario].append(patient_dict[scenario]['ssim'])
                patient_mses.append(patient_dict[scenario]['mse'])
                patient_psnrs.append(patient_dict[scenario]['psnr'])
                patient_ssims.append(patient_dict[scenario]['ssim'])
    print("results:\n")
    print("sc\t\tmse\t\tpsnr\t\tssim")
    for scenario in scenarios:
        print("%s-mean\t%f\t%f\t%f\t" % (
            scenario,
            np.array(mse_rsts[scenario]).mean(),
            np.array(psnr_rsts[scenario]).mean(),
            np.array(ssim_rsts[scenario]).mean(),
        ))
        print("%s-std\t%f\t%f\t%f\t" % (
            scenario,
            np.array(mse_rsts[scenario]).std(),
            np.array(psnr_rsts[scenario]).std(),
            np.array(ssim_rsts[scenario]).std(),
        ))
    for scenario in scenarios:
        print("%s-mean\t%f\t%f\t%f\t" % (
            scenario,
            np.array(mse_rsts[scenario]).mean(),
            np.array(psnr_rsts[scenario]).mean(),
            np.array(ssim_rsts[scenario]).mean(),
        ))
    for scenario in scenarios:
        print("%s-std\t%f\t%f\t%f\t" % (
            scenario,
            np.array(mse_rsts[scenario]).std(),
            np.array(psnr_rsts[scenario]).std(),
            np.array(ssim_rsts[scenario]).std(),
        ))


if __name__ == '__main__':
    tfrecords = [
        r"E:\my_files\programmes\python\BRATS2018_normalized\group0_standardized_42patients_5712slices.tfrecord",
        r"E:\my_files\programmes\python\BRATS2018_normalized\group1_standardized_42patients_5712slices.tfrecord",
        r"E:\my_files\programmes\python\BRATS2018_normalized\group2_standardized_42patients_5712slices.tfrecord",
        r"E:\my_files\programmes\python\BRATS2018_normalized\group3_standardized_42patients_5712slices.tfrecord",
        r"E:\my_files\programmes\python\BRATS2018_normalized\group4_standardized_42patients_5712slices.tfrecord",
    ]
    '''test(parameter_path=r"E:\my_files\programmes\python\mri_gan_output\comb0\saved_checkpoints\ckpt-19",
         session_name="test-epoch19", output_dir=r"E:\my_files\programmes\python\mri_gan_output\comb0",
         tfrecords=[tfrecords[4]], channels=['t1', 't2', 't1ce', 'flair'], img_shape=(256, 256))
    '''

    '''torch_test(parameter_path=r"E:\my_files\programmes\python\mri_gan_output\comb0_torch\saved_checkpoints\generator_param_59.pkl",
               session_name="test-epoch59", output_dir=r"E:\my_files\programmes\python\mri_gan_output\comb0_torch",
               tfrecords=[tfrecords[4]], channels=['t1', 't2', 't1ce', 'flair'], img_shape=(256, 256))'''

    read_jsons(json_dir=r'E:\my_files\programmes\python\mri_gan_output\comb0\test-epoch59',
               scenarios=['0001', '0010', '0100', '1000',
                          '0011', '0101', '0110', '1001', '1010', '1100',
                          '0111', '1011', '1101', '1110', ], )

    '''slice_to_slice_comparison(
        nii_path_a=r'E:\my_files\programmes\python\mri_gan_output\comb0\test-epoch19\Brats18_TCIA01_186_1\1110\flair.nii.gz',
        nii_path_b=r'E:\my_files\programmes\python\mri_gan_output\comb0\test-epoch19\Brats18_TCIA01_186_1\1101\flair.nii.gz'
    )'''

    '''evaluate_nii_to_nii(
        names_txt=r"E:\my_files\programmes\python\BRATS2018_normalized\group4_standardized.txt",
        test_path=r"E:\my_files\programmes\python\mri_gan_output\comb0\test-epoch19",
        scenarios=['0001', '0010', '0100', '1000',
                   '0011', '0101', '0110', '1001', '1010', '1100',
                   '0111', '1011', '1101', '1110',],
        channels=CHANNELS,
        out_txt_dir=r'E:\my_files\programmes\python\mri_gan_output\comb0',
        out_txt_name='evaluation.txt'
    )'''
