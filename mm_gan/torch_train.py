import itertools
import logging
import os
import numpy as np
import time
import datetime
import json
import sys

import matplotlib.pyplot as plt
import tqdm
import re
# import argparse
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import torch
import mm_gan.torch_models as torch_models
import pickle
from mm_gan import pytorch_msssim as pyt_ssim
from mm_gan.constants_and_tools import make_dir
from mm_gan.io_helpers import get_dataset
from pathlib import Path
from tensorflow import summary, concat
from skimage.metrics import structural_similarity as sk_ssim


gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])

# check if GPU is available
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")

# define tensor type
tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def select_scenarios(batch_size: int, scenario_range: list) -> list:
    rand_vals = []
    for i in range(batch_size):
        rand_val = np.random.randint(
            low=scenario_range[0], high=scenario_range[-1])
        rand_vals.append(rand_val)
    return rand_vals


def get_label_scenarios(scenarios: list, label_vals: list, full_random=True) -> list:
    label_scenarios = []
    for each in label_vals:
        label_scenario = None
        if full_random:
            label_scenario = scenarios[int(each)]
        else:
            label_scenario = scenarios[int(label_vals[0])]
        label_scenarios.append(label_scenario)
    return label_scenarios


def create_impute_tensors_and_update_label_list(label_scenarios: list, x_z: np.ndarray,
                                                label_list: np.ndarray, img_shape: tuple):
    for slice_num, each_label_scenario in enumerate(label_scenarios):
        impute_tensor = np.zeros((img_shape[0], img_shape[1]))
        for channel_num, each_channel_label in enumerate(each_label_scenario):
            if each_channel_label == 0:
                x_z[slice_num, channel_num, ...] = impute_tensor
                label_list[slice_num, channel_num, ...] = 0
            else:
                label_list[slice_num, channel_num, ...] = 1
    return x_z, label_list


def impute_reals_into_fake(x_z, fake_x, label_scenarios):
    # indices = []
    # updates = []
    for slice_num, label_scenario in enumerate(label_scenarios):
        for idx, k in enumerate(label_scenario):
            if k == 1:  # THIS IS A REAL AVAILABLE SEQUENCE
                fake_x[slice_num, idx, ...] = x_z[slice_num, idx, ...]
                # fake_x[slice_num, idx, ...].assign(x_z[slice_num, idx, ...])
                # indices.append([slice_num, idx])
                # updates.append(x_z[slice_num, idx, ...])
    # fake_x = tensor_scatter_nd_update(fake_x, indices, updates)
    return fake_x


def mse(slice_a: torch.Tensor, slice_b: torch.Tensor) -> float:
    return (torch.square(torch.subtract(slice_a, slice_b))).mean()


def psnr(slice_a: torch.Tensor, slice_b: torch.Tensor, max_val: float):
    return 10 * torch.log10(max_val**2 / mse(slice_a, slice_b))


def psnr_torch(pred, gt, cuda=True):
    # normalize images between [0, 1]
    epsilon = 0.00001
    epsilon2 = torch.from_numpy(np.array(0.01, dtype=np.float32)).cuda()\
        if cuda else torch.from_numpy(np.array(0.01, dtype=np.float32))
    # always use ground truth
    gt_n = gt / (gt.max() + epsilon)
    pred_n = pred / (pred.max() + epsilon)

    PIXEL_MAX = 1.0

    mse = torch.mean((gt_n - pred_n) ** 2)
    if mse.item() == 0.0:
        psnr = 20 * torch.log10(PIXEL_MAX / epsilon2)
    else:
        psnr = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

    return psnr


def load_checkpoint(model, optimizer, filename, pickle_module, device, logger):
    if os.path.isfile(filename):
        logger.info("Loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(
            filename, pickle_module=pickle_module, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("Loaded checkpoint '{}' (epoch {})"
                    .format(filename, checkpoint['epoch']))
    else:
        logger.critical('Checkpoint {} does not exist.'.format(filename))
    return model, optimizer


def train(session_name: str, output_dir: str, tfrecords_train: list, tfrecords_val: list,
          batch_size_train: int, img_shape: tuple, full_random: bool,
          learning_rate: float, beta_1: float, beta_2: float, lambda_param: float, epochs: int, sub_epochs: int,
          epochs_per_dom_epoch: int, implicit_conditioning=True, curriculum_learning=True, focus_on_worst=True, debug=False):
    # make dir for output dir
    output_dir = os.path.join(output_dir, session_name)
    make_dir(output_dir)

    log_dir = os.path.join(output_dir, 'logs')
    make_dir(log_dir)

    json_dir = os.path.join(output_dir, 'jsons')
    make_dir(json_dir)

    # get loggers
    train_logger = logging.getLogger(name=session_name + "_train")
    train_logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(
        sys.stdout)  # stderr output to console
    log_path_train = os.path.join(log_dir, session_name + "_train" + ".txt")
    file_handler = logging.FileHandler(log_path_train, mode='a')
    formatter = logging.Formatter(
        '%(asctime)s: %(message)s', "%y%b%d-%H:%M:%S")
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    train_logger.addHandler(stream_handler)
    train_logger.addHandler(file_handler)

    val_logger = logging.getLogger(name=session_name + "_val")
    val_logger.setLevel(logging.INFO)
    log_path_val = os.path.join(log_dir, session_name + "_val" + ".txt")
    file_handler = logging.FileHandler(log_path_val, mode='a')
    file_handler.setFormatter(formatter)
    val_logger.addHandler(file_handler)

    # create tensorboard loggers
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    tb_train_dir = os.path.join(tensorboard_dir, "train")
    tb_val_dir = os.path.join(tensorboard_dir, "val")
    make_dir(tb_train_dir)
    make_dir(tb_val_dir)
    train_summary_writer = summary.create_file_writer(tb_train_dir)
    val_summary_writer = summary.create_file_writer(tb_val_dir)

    pat = r"\d+"
    train_patients = 0
    train_slices = 0
    for each_file in tfrecords_train:
        num_in_names = re.findall(pat, os.path.split(each_file)[-1])
        train_patients += int(num_in_names[1])
        train_slices += int(num_in_names[2])
    val_patients = 0
    val_slices = 0
    for each_file in tfrecords_val:
        num_in_names = re.findall(pat, os.path.split(each_file)[-1])
        val_patients += int(num_in_names[1])
        val_slices += int(num_in_names[2])

    # loss functions
    criterion_gan = torch.nn.MSELoss()  # BCELoss()
    criterion_pixelwise = torch.nn.L1Loss()
    mse_fake_vs_real = torch.nn.MSELoss()

    # patch size is 16 * 16 for 256 * 256 input
    patch_size = (4, 16, 16)

    # get networks
    generator = torch_models.GeneratorUNet(
        in_channels=4, out_channels=4, with_relu=True, with_tanh=False)
    discriminator = torch_models.Discriminator(
        in_channels=4, dataset='BRATS2018')

    # get optimizers
    optimizer_g = torch.optim.Adam(
        generator.parameters(), lr=learning_rate, betas=(beta_1, beta_2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate,
                                   betas=(beta_1, beta_2))

    # if cuda is available, send everything to GPU
    if cuda:
        generator = torch.nn.DataParallel(generator.cuda())
        discriminator = torch.nn.DataParallel(discriminator.cuda())
        criterion_gan.cuda()
        criterion_pixelwise.cuda()
        mse_fake_vs_real.cuda()
    else:
        generator = torch.nn.DataParallel(generator)
        discriminator = torch.nn.DataParallel(discriminator)

    # use checkpoints
    checkpoint_dir = os.path.join(output_dir, "saved_checkpoints")
    make_dir(checkpoint_dir)

    # get current epoch
    start_epoch = 0
    if Path(log_path_train).is_file():
        log = open(log_path_train, 'r')
        next_line = log.readline()
        while next_line:
            if "===EPOCH=FINISH===" in next_line:
                start_epoch += 1
            next_line = log.readline()

    # init networks and optimizers
    # Initialize weights
    if start_epoch != 0:
        # Load pretrained models
        train_logger.info('Loading previous checkpoint!')
        generator, optimizer_g \
            = load_checkpoint(generator, optimizer_g,
                              os.path.join(checkpoint_dir, "{}_param_{}.pkl".format(
                                  'generator', start_epoch - 1)),
                              pickle_module=pickle, device=device, logger=train_logger)
        discriminator, optimizer_d \
            = load_checkpoint(discriminator, optimizer_d,
                              os.path.join(checkpoint_dir, "{}_param_{}.pkl".format(
                                  'discriminator', start_epoch - 1)),
                              pickle_module=pickle, device=device, logger=train_logger)
    else:
        generator.apply(torch_models.weights_init_normal)
        discriminator.apply(torch_models.weights_init_normal)

    # Get the device we're working on.
    train_logger.debug("cuda & device")
    train_logger.debug(cuda)
    train_logger.debug(device)

    # create sc scenarios from 0000 to 1111, yet 0000 and 1111 should not be used
    scenarios = list(map(list, itertools.product([0, 1], repeat=4)))
    scenarios.remove([0, 0, 0, 0])
    scenarios.remove([1, 1, 1, 1])

    # sort the scenarios according to decreasing difficulty. Easy scenarios last, and difficult ones first.
    scenarios.sort(key=lambda x: x.count(1))
    bad_scenario = -1

    # start training
    train_logger.info("===== start training =====")
    dom_validations_left = (epochs - start_epoch) // epochs_per_dom_epoch
    if (epochs - start_epoch) % epochs_per_dom_epoch != 0:
        dom_validations_left += 1  # keep one validation always for the last epoch
    dom_val_time = 3600

    for epoch_idx in range(start_epoch, epochs):
        train_logger.info("starting epoch %d..." % epoch_idx)
        # we've already loaded some patients
        patients_num = train_patients
        batches_per_epoch = train_slices // batch_size_train
        batches_per_sub_epoch = batches_per_epoch // sub_epochs

        # prepare val patients
        val_patient_num = val_patients
        val_patient_num_per_sub_epoch = val_patient_num // sub_epochs
        val_patient_left = val_patient_num

        # create data fetchers, shuffle them every epoch
        dataset_train = get_dataset(tfrecords=tfrecords_train, compression_type='ZLIB', batch_size=batch_size_train, drop_remainder=True,
                                    shuffle=True, buffer_size=3000)
        dataset_val = get_dataset(tfrecords=tfrecords_val, compression_type='ZLIB', batch_size=1, drop_remainder=True,
                                  shuffle=True, buffer_size=2000)

        train_iter = dataset_train.__iter__()
        val_iter = dataset_val.__iter__()

        # run sub-epochs
        for sub_epoch_idx in range(sub_epochs):
            train_logger.info("starting sub-epoch %d..." % sub_epoch_idx)

            # record sub-epoch time
            pbar = tqdm.tqdm(total=batches_per_sub_epoch)
            sub_epoch_start_time = time.time()
            pbar.set_description("sub-epoch %d training" % sub_epoch_idx)

            # sub-epoch summary
            batch_summaries = {
                'g_train_loss': [],
                'd_train_loss': [],
                'pixel_loss': [],
                'mse_loss': []
            }
            sub_epoch_summary = {}

            for batch_idx in range(batches_per_sub_epoch):
                # curriculum learning:
                rand_val = 0
                if curriculum_learning:
                    if epoch_idx < 10:
                        curr_scenario_range = [10, 14]
                    elif 10 <= epoch_idx < 20:
                        curr_scenario_range = [4, 14]
                    elif 20 <= epoch_idx < 30:
                        curr_scenario_range = [4, 14]
                    else:
                        curr_scenario_range = [0, 14]

                    rand_vals = select_scenarios(
                        batch_size_train, curr_scenario_range)
                else:  # not going to take curriculum learning?
                    rand_vals = select_scenarios(batch_size_train, [0, 14])

                # if its the first 20% batches of this sub epoch, we focus on the worst scenario
                if focus_on_worst and bad_scenario != -1:
                    if batch_idx <= batches_per_epoch * 0.1 or batches_per_epoch * 0.5 <= batch_idx <= batches_per_epoch * 0.6:
                        for i in range(len(rand_vals)):
                            rand_vals[i] = bad_scenario

                # get scenarios for each slice in the batch
                label_scenarios = get_label_scenarios(
                    scenarios, rand_vals, full_random=full_random)

                # get batch
                # zero is the name
                each_batch = train_iter.__next__()
                t1 = each_batch[1]
                t2 = each_batch[2]
                t1ce = each_batch[3]
                flair = each_batch[4]
                batch = concat([t1, t2, t1ce, flair], axis=1).numpy()
                batch_z = batch.copy()

                # create label list
                label_list_f = np.ones(
                    (batch_size_train, patch_size[0], patch_size[1], patch_size[2]))

                batch_z, label_list_f = create_impute_tensors_and_update_label_list(label_scenarios, batch_z,
                                                                                    label_list_f, img_shape)

                # get real batch output
                label_list_r = np.ones(
                    (batch_size_train, patch_size[0], patch_size[1], patch_size[2]))

                # train on batch
                batch_summary = train_on_batch(batch=batch, batch_z=batch_z, label_scenarios=label_scenarios,
                                               implicit_conditioning=implicit_conditioning, generator=generator,
                                               discriminator=discriminator, criterion_gan=criterion_gan,
                                               criterion_pixelwise=criterion_pixelwise,
                                               mse_fake_vs_real=mse_fake_vs_real, label_list_real=label_list_r,
                                               label_list_fake=label_list_f, lambda_param=lambda_param,
                                               optimizer_g=optimizer_g, optimizer_d=optimizer_d, debug=debug)

                for each_key in batch_summary.keys():
                    batch_summary[each_key] = batch_summary[each_key].item()
                    batch_summaries[each_key].append(
                        batch_summary[each_key])  # add them to summaries

                pbar.update()  # update pbar

            pbar.close()  # must close this
            # compute mean losses
            for each_key in batch_summaries.keys():
                sub_epoch_summary[each_key] = np.mean(
                    np.array(batch_summaries[each_key])).item()
            train_logger.info("g_train_loss: %f; d_train_loss: %f; pixel_loss: %f; mse_loss: %f"
                              % (sub_epoch_summary['g_train_loss'], sub_epoch_summary['d_train_loss'],
                                 sub_epoch_summary['pixel_loss'], sub_epoch_summary['mse_loss']))

            # save dict_sum to json
            js_obj = json.dumps(batch_summaries)
            js_path = os.path.join(
                json_dir, 'training-%d-%d.json' % (epoch_idx, sub_epoch_idx))
            js_file = open(js_path, 'w')
            js_file.write(js_obj)
            js_file.close()

            # save epoch results to summary
            step = epoch_idx * sub_epochs + sub_epoch_idx
            with train_summary_writer.as_default():
                for each_key in batch_summaries.keys():
                    summary.scalar(
                        each_key, sub_epoch_summary[each_key], step=step)

            # sub epoch validation
            # train_logger.info("start validation for this sub-epoch...")
            val_summary, bad_scenario = validate(val_iter=val_iter, val_logger=val_logger, generator=generator,
                                                 scenarios=scenarios, json_dir=json_dir, epoch=epoch_idx, sub_epoch=sub_epoch_idx,
                                                 val_batchs=int(val_slices // sub_epochs * 0.15), debug=debug)

            # save val results to summary
            with val_summary_writer.as_default():
                for each_scenario in val_summary.keys():
                    summary.scalar('%s-mse' % each_scenario,
                                   val_summary[each_scenario]['mse'], step=step)
                    summary.scalar('%s-psnr' % each_scenario,
                                   val_summary[each_scenario]['psnr'], step=step)
                    summary.scalar('%s-ssim' % each_scenario,
                                   val_summary[each_scenario]['ssim'], step=step)
            val_patient_left -= val_patient_num_per_sub_epoch

            # print total time and average time of batches
            sub_epoch_end_time = time.time()
            sub_epoch_total_time = int(
                sub_epoch_end_time - sub_epoch_start_time)
            sub_epoch_total_time_hms = str(
                datetime.timedelta(seconds=sub_epoch_total_time))
            total_sub_epochs_left = sub_epochs * \
                (epochs - epoch_idx - 1) + sub_epochs - sub_epoch_idx - 1
            total_time_left = int(
                total_sub_epochs_left * sub_epoch_total_time + dom_val_time * dom_validations_left)
            total_time_left_hms = str(
                datetime.timedelta(seconds=total_time_left))

            # output to log
            train_logger.info("epoch %d sub-epoch %d; time total: %s total_time_left: %s"
                              % (epoch_idx, sub_epoch_idx, sub_epoch_total_time_hms, total_time_left_hms))

        # save checkpoint
        gen_state_checkpoint = {
            'epoch': epoch_idx,
            'state_dict': generator.state_dict(),
            'optimizer': optimizer_g.state_dict(),
        }
        des_state_checkpoint = {
            'epoch': epoch_idx,
            'state_dict': discriminator.state_dict(),
            'optimizer': optimizer_d.state_dict(),
        }
        torch.save(gen_state_checkpoint,
                   os.path.join(checkpoint_dir, "{}_param_{}.pkl".format('generator', epoch_idx)), pickle_module=pickle)
        torch.save(des_state_checkpoint,
                   os.path.join(checkpoint_dir, "{}_param_{}.pkl".format('discriminator', epoch_idx)), pickle_module=pickle)

        train_logger.info("===EPOCH=FINISH===")

        # dom-validation
        if (epoch_idx + 1) % epochs_per_dom_epoch == 0 or (epoch_idx + 1) == epochs:
            train_logger.info("start dom validation...")
            val_start_time = time.time()
            val_iter = dataset_val.__iter__()  # get a new iterator
            val_summary, bad_scenario = validate(val_iter=val_iter, val_logger=val_logger, generator=generator,
                                                 scenarios=scenarios, json_dir=json_dir, epoch=epoch_idx, sub_epoch=-1,
                                                 val_batchs=val_slices, debug=debug)
            val_end_time = time.time()
            dom_val_time = int(val_end_time - val_start_time)
            dom_validations_left -= 1

            # save val results to summary
            with val_summary_writer.as_default():
                for each_scenario in val_summary.keys():
                    summary.scalar('%s-mse-dom' % each_scenario,
                                   val_summary[each_scenario]['mse'], step=epoch_idx)
                    summary.scalar('%s-psnr-dom' % each_scenario,
                                   val_summary[each_scenario]['psnr'], step=epoch_idx)
                    summary.scalar('%s-ssim-dom' % each_scenario,
                                   val_summary[each_scenario]['ssim'], step=epoch_idx)


# @function  # tf.function
def train_on_batch(batch: np.ndarray, batch_z: np.ndarray, label_scenarios: list, implicit_conditioning: bool,
                   generator, discriminator, criterion_gan, criterion_pixelwise, mse_fake_vs_real,
                   label_list_fake: np.ndarray, label_list_real: np.ndarray, lambda_param: float,
                   optimizer_g, optimizer_d, debug: bool) -> dict:

    with torch.autograd.set_detect_anomaly(True):
        # set train states
        generator.train()
        discriminator.eval()

        # train generator
        generator.zero_grad()
        optimizer_g.zero_grad()

        if cuda:
            batch = torch.from_numpy(batch).cuda().type(tensor)
            batch_z = torch.from_numpy(batch_z).cuda().type(tensor)
            label_list_real = torch.from_numpy(
                label_list_real).cuda().type(tensor)
            label_list_fake = torch.from_numpy(
                label_list_fake).cuda().type(tensor)
        else:
            batch = torch.from_numpy(batch).type(tensor)
            batch_z = torch.from_numpy(batch_z).type(tensor)
            label_list_real = torch.from_numpy(label_list_real).type(tensor)
            label_list_fake = torch.from_numpy(label_list_fake).type(tensor)

        # get fake img
        fake_x = generator(batch_z)

        # implicit conditioning
        if implicit_conditioning:
            fake_x = impute_reals_into_fake(batch_z, fake_x, label_scenarios)

        # predict the fake & real
        pred_fake = discriminator(fake_x, batch)

        loss_gan = criterion_gan(
            pred_fake, label_list_real)  # generator should make discriminator think it is real

        if implicit_conditioning:
            loss_pixel = 0
            synth_loss = 0
            count = 0
            for num_slice, label_scenario in enumerate(label_scenarios):
                for idx_curr_label, i in enumerate(label_scenario):
                    if i == 0:
                        loss_pixel = loss_pixel + criterion_pixelwise(fake_x[num_slice, idx_curr_label, ...],
                                                                      batch[num_slice, idx_curr_label, ...])

                        synth_loss = synth_loss + mse_fake_vs_real(fake_x[num_slice, idx_curr_label, ...],
                                                                   batch[num_slice, idx_curr_label, ...])
                        count += 1
            loss_pixel = loss_pixel / count
            synth_loss = loss_pixel / count
        else:  # no IC, calculate loss for all output w.r.t all GT.
            loss_pixel = criterion_pixelwise(fake_x, batch)
            # loss_pixel = mae(fake_x, batch)
            synth_loss = mse_fake_vs_real(fake_x, batch)

        # total generator loss
        g_train_total_loss = (1 - lambda_param) * \
            loss_gan + lambda_param * loss_pixel

        # apply gradients
        g_train_total_loss.backward()
        optimizer_g.step()

        # train discriminator
        discriminator.train()
        discriminator.zero_grad()
        optimizer_d.zero_grad()

        pred_real = discriminator(batch, batch)

        if debug:
            pred_real_debug = pred_real.detach().cpu().numpy()
            pred_fake_debug = pred_fake.detach().cpu().numpy()
            fake_x_debug = fake_x.detach().cpu().numpy()
            channels = ['t1', 't2', 't1ce', 'flair']
            for idx, label in enumerate(label_scenarios):
                label_str = ''
                for each in label:
                    label_str += str(each)
                for i in range(4):
                    plt.figure()
                    plt.subplot(1, 3, 1)
                    plt.imshow(pred_real_debug[idx][i])
                    label_title = label_str + channels[i] + "pred_real"
                    plt.title(label_title)
                    plt.subplot(1, 3, 2)
                    plt.imshow(pred_fake_debug[idx][i])
                    label_title = label_str + channels[i] + "pred_fake"
                    plt.title(label_title)
                    plt.subplot(1, 3, 3)
                    plt.imshow(fake_x_debug[idx][i])
                    label_title = label_str + channels[i] + "fake_x"
                    plt.title(label_title)
                plt.show()

        # discriminator should think the real img real
        loss_real = criterion_gan(pred_real, label_list_real)

        # get fake img
        fake_x = generator(batch_z)

        # implicit conditioning
        if implicit_conditioning:
            fake_x = impute_reals_into_fake(batch_z, fake_x, label_scenarios)

        # predict the fake & real
        pred_fake = discriminator(fake_x.detach(), batch)

        # discriminator should think the fake img fake
        loss_fake = criterion_gan(pred_fake, label_list_fake)

        # total discriminator loss
        d_train_total_loss = 0.5 * (loss_real + loss_fake)

        # apply gradients
        d_train_total_loss.backward()
        optimizer_d.step()

    dict_summary = {
        'g_train_loss': g_train_total_loss,
        'd_train_loss': d_train_total_loss,
        'pixel_loss': loss_pixel,
        'mse_loss': synth_loss
    }

    return dict_summary


def validate(val_iter: iter, val_logger: logging.Logger, generator, scenarios: list,
             json_dir: str, epoch: int, sub_epoch: int, val_batchs: int, debug: bool) -> tuple:
    """
    sub & dom validation
    :param sub_epoch: -1 means dom-validation
    """
    if sub_epoch != -1:
        val_logger.info("epoch %d sub-epoch %d, validating" %
                        (epoch, sub_epoch))
    else:
        val_logger.info("epoch %d, dom-validating" % epoch)
    # print("validation-->| ", end='', flush=True)

    # set eval state
    generator.eval()

    metrics = []
    for i in range(len(scenarios)):
        metrics.append({'mse': [], 'psnr': [], 'ssim': []})

    # mse = MeanSquaredError()

    # we've already loaded some patients
    # patient_num = len(validation_fetcher.patient_dirs) + validation_fetcher.patients_in_buffer
    batch_num = val_batchs

    pbar = tqdm.tqdm(total=batch_num)
    # print("sub-epoch %d >| " % sub_epoch_idx, end='', flush=True)
    if sub_epoch != -1:
        pbar.set_description("sub-epoch %d validation" % sub_epoch)
    else:
        pbar.set_description("dom-validation for epoch %d" % epoch)

    for idx in range(batch_num):
        each_batch = val_iter.__next__()
        t1 = each_batch[1]
        t2 = each_batch[2]
        t1ce = each_batch[3]
        flair = each_batch[4]
        batch = concat([t1, t2, t1ce, flair], axis=1).numpy()
        sc_idx = 0

        if cuda:
            batch = torch.from_numpy(batch).cuda().type(tensor)
        else:
            batch = torch.from_numpy(batch).type(tensor)

        for each_scenario in scenarios:
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

            if debug:
                scenario_str = ''
                for each in each_scenario:
                    scenario_str += str(each)
                plt.figure()
                plt.subplot(1, 4, 1)
                if each_scenario[0] == 0:
                    plt.imshow(fake_img[0, 0, ...])
                else:
                    plt.imshow(batch_z[0, 0, ...])
                plt.title("%s - t1" % scenario_str)
                plt.subplot(1, 4, 2)
                if each_scenario[1] == 0:
                    plt.imshow(fake_img[0, 1, ...])
                else:
                    plt.imshow(batch_z[0, 1, ...])
                plt.title("%s - t2" % scenario_str)
                plt.subplot(1, 4, 3)
                if each_scenario[2] == 0:
                    plt.imshow(fake_img[0, 2, ...])
                else:
                    plt.imshow(batch_z[0, 2, ...])
                plt.title("%s - t1ce" % scenario_str)
                plt.subplot(1, 4, 4)
                if each_scenario[3] == 0:
                    plt.imshow(fake_img[0, 3, ...])
                else:
                    plt.imshow(batch_z[0, 3, ...])
                plt.title("%s - flair" % scenario_str)
                plt.show()

            ssim_filter_size = 11
            '''if batch_z.shape[0] < 11:
                ssim_filter_size = batch_z.shape[0]'''

            for c, each_channel in enumerate(each_scenario):
                if each_channel == 0:
                    fake_channel_arr = torch.unsqueeze(fake_img[:, c, ...], 2)
                    real_channel_arr = torch.unsqueeze(batch[:, c, ...], 2)
                    # max_val = max(fake_channel_arr.numpy().max(), real_channel_arr.numpy().max())
                    # max_val = 1.0
                    mse_rst = mse(fake_channel_arr / (torch.max(fake_channel_arr) + 0.0001),
                                  real_channel_arr / (torch.max(real_channel_arr) + 0.0001))
                    # psnr_rst = psnr(fake_channel_arr, real_channel_arr, max_val)
                    psnr_rst = psnr_torch(fake_channel_arr, real_channel_arr)
                    # ssim_rst = pyt_ssim.ssim(fake_channel_arr / (torch.max(fake_channel_arr) + 0.0001),
                    #                         real_channel_arr / (torch.max(real_channel_arr) + 0.0001), val_range=1)

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
            sc_idx += 1
        pbar.update()  # update the bar

    epoch_rst_dict = {}
    print("\n", end="", flush=True)
    bad_scenario = 0
    bad_scenario_str = ''
    for each in scenarios[0]:
        bad_scenario_str += str(each)
    bad_mse = np.array(metrics[0]['mse']).mean()
    val_logger.info("sc\tmse\t\tpsnr\t\tssim")
    print("sc\tmse\t\tpsnr\t\tssim")
    for sc_idx, scenario in enumerate(scenarios):
        scenario_str = ''
        for each in scenario:
            scenario_str += str(each)
        mse_mat = np.array(metrics[sc_idx]['mse'])
        psnr_mat = np.array(metrics[sc_idx]['psnr'])
        ssim_mat = np.array(metrics[sc_idx]['ssim'])
        psnr_mat[np.isposinf(psnr_mat)] = 30
        epoch_rst_dict[scenario_str] = {
            "mse": mse_mat.mean(),
            "psnr": psnr_mat.mean(),
            "ssim": ssim_mat.mean(),
        }
        val_logger.info("%s\t%f\t%f\t%f" % (scenario_str, epoch_rst_dict[scenario_str]["mse"],
                                                  epoch_rst_dict[scenario_str]["psnr"],
                                                  epoch_rst_dict[scenario_str]["ssim"]))
        print("%s\t%f\t%f\t%f" % (scenario_str, epoch_rst_dict[scenario_str]["mse"],
                                        epoch_rst_dict[scenario_str]["psnr"],
                                        epoch_rst_dict[scenario_str]["ssim"]))
        if np.array(metrics[sc_idx]['mse']).mean() > bad_mse:
            bad_mse = np.array(metrics[sc_idx]['mse']).mean()
            bad_scenario = sc_idx
            bad_scenario_str = scenario_str
    print("bad scenario: [%d] - %s" % (bad_scenario, bad_scenario_str))

    # json
    js_obj = json.dumps(epoch_rst_dict)
    if sub_epoch != -1:
        js_path = os.path.join(
            json_dir, 'validation-%d-%d.json' % (epoch, sub_epoch))
    else:
        js_path = os.path.join(json_dir, 'dom-validation-%d.json' % epoch)
    js_file = open(js_path, 'w')
    js_file.write(js_obj)
    js_file.close()
    pbar.close()

    return epoch_rst_dict, bad_scenario


if __name__ == '__main__':
    tfrecords = [
        r"E:\my_files\programmes\python\BRATS2018_normalized\group0_standardized_42patients_5712slices.tfrecord",
        r"E:\my_files\programmes\python\BRATS2018_normalized\group1_standardized_42patients_5712slices.tfrecord",
        r"E:\my_files\programmes\python\BRATS2018_normalized\group2_standardized_42patients_5712slices.tfrecord",
        r"E:\my_files\programmes\python\BRATS2018_normalized\group3_standardized_42patients_5712slices.tfrecord",
        r"E:\my_files\programmes\python\BRATS2018_normalized\group4_standardized_42patients_5712slices.tfrecord",
    ]
    train(session_name="comb0_torch", output_dir=r"E:\my_files\programmes\python\mri_gan_output",
          tfrecords_train=tfrecords[0:4], tfrecords_val=[tfrecords[4]],
          batch_size_train=8, full_random=False,
          img_shape=(256, 256), learning_rate=0.0002, beta_1=0.5, beta_2=0.999, lambda_param=0.9, epochs=60,
          epochs_per_dom_epoch=10, sub_epochs=10, implicit_conditioning=True, curriculum_learning=True,
          focus_on_worst=False, debug=False)
