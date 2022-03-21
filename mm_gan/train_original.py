import itertools
import logging
import os
import numpy as np
import time
import datetime
import json
import sys
import data_fetcher
import matplotlib.pyplot as plt
import tqdm
# import argparse
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from mm_gan.constants_and_tools import make_dir
from mm_gan.data_fetcher import DataFetcher
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from mm_gan.models import get_generator_unet, get_discriminator
from tensorflow.train import Checkpoint, CheckpointManager
from pathlib import Path
from tensorflow import GradientTape, float32, constant, cast, tensor_scatter_nd_update, summary
# , function, py_function
from tensorflow.image import ssim, psnr


# ap = argparse.ArgumentParser()
# ap.add_argument("")


def select_scenarios(batch_size: int, scenario_range: list) -> list:
    rand_vals = []
    for i in range(batch_size):
        rand_val = np.random.randint(low=scenario_range[0], high=scenario_range[-1])
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
    indices = []
    updates = []
    for slice_num, label_scenario in enumerate(label_scenarios):
        for idx, k in enumerate(label_scenario):
            if k == 1:  # THIS IS A REAL AVAILABLE SEQUENCE
                # fake_x[slice_num, idx, ...] = x_z[slice_num, idx, ...]
                # fake_x[slice_num, idx, ...].assign(x_z[slice_num, idx, ...])
                indices.append([slice_num, idx])
                updates.append(x_z[slice_num, idx, ...])
    fake_x = tensor_scatter_nd_update(fake_x, indices, updates)
    return fake_x


def train(session_name: str, output_dir: str,
          group_txt_names_train: list, group_txt_names_val: list, group_txt_dir_train: str, group_txt_dir_val: str,
          channels: list, batch_size_train: int, num_patients_in_buffer_train: int, batch_size_val: int,
          num_patients_in_buffer_val: int, nii_shape: tuple, slice_cut_off: tuple, img_shape: tuple,
          learning_rate: float, beta_1: float, beta_2: float, lambda_param: float, epochs: int, sub_epochs: int,
          epochs_per_dom_epoch: int, implicit_conditioning=True, curriculum_learning=True, debug=False):
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
    stream_handler = logging.StreamHandler(sys.stdout)  # stderr output to console
    log_path_train = os.path.join(log_dir, session_name + "_train" + ".txt")
    file_handler = logging.FileHandler(log_path_train, mode='a')
    formatter = logging.Formatter('%(asctime)s: %(message)s', "%y%b%d-%H:%M:%S")
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

    # create data fetchers
    reset_start_time = time.time()
    training_fetcher = DataFetcher(group_txt_dir=group_txt_dir_train,
                                   group_txt_names=group_txt_names_train,
                                   channels=channels, batch_size=batch_size_train,
                                   patients_in_buffer=num_patients_in_buffer_train, nii_shape=nii_shape, shuffle=True,
                                   slice_cut_off=slice_cut_off, img_shape=img_shape, name="training fetcher")
    validation_fetcher = DataFetcher(group_txt_dir=group_txt_dir_val,
                                     group_txt_names=group_txt_names_val,
                                     channels=channels, batch_size=batch_size_val,
                                     patients_in_buffer=num_patients_in_buffer_val, nii_shape=nii_shape, shuffle=True,
                                     slice_cut_off=slice_cut_off, img_shape=img_shape, name="validation fetcher")
    reset_end_time = time.time()
    reset_time = int(reset_end_time - reset_start_time)

    # get losses
    criterion_gan = MeanSquaredError()
    criterion_pixelwise = MeanAbsoluteError()
    mse_fake_vs_real = MeanSquaredError()

    # patch size is 16 * 16 for 256 * 256 input
    patch_size = (4, 16, 16)

    # get networks
    generator = get_generator_unet(input_shape=(4, img_shape[0], img_shape[1]), out_channels=4)
    discriminator = get_discriminator(input_shape=(4, img_shape[0], img_shape[1]), out_channels=4)

    # get optimizers
    optimizer_g = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
    optimizer_d = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)

    # use checkpoints
    checkpoint_dir = os.path.join(output_dir, "saved_checkpoints")
    make_dir(checkpoint_dir)
    # checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
    checkpoint = Checkpoint(generator_optimizer=optimizer_g, discriminator_optimizer=optimizer_d,
                            generator=generator, discriminator=discriminator)
    checkpoint_manager = CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=epochs)

    # get current epoch
    start_epoch = 0
    if Path(log_path_train).is_file():
        log = open(log_path_train, 'r')
        next_line = log.readline()
        while next_line:
            if "===EPOCH=FINISH===" in next_line:
                start_epoch += 1
            next_line = log.readline()

    if start_epoch != 0:
        checkpoint_manager.restore_or_initialize()
        train_logger.info("checkpoint restored")

    # create sc scenarios from 0000 to 1111, yet 0000 and 1111 should not be used
    scenarios = list(map(list, itertools.product([0, 1], repeat=4)))
    scenarios.remove([0, 0, 0, 0])
    scenarios.remove([1, 1, 1, 1])

    # sort the scenarios according to decreasing difficulty. Easy scenarios last, and difficult ones first.
    scenarios.sort(key=lambda x: x.count(1))

    # start training
    train_logger.info("===== start training =====")
    dom_validations_left = (epochs - start_epoch) // epochs_per_dom_epoch
    if (epochs - start_epoch) % epochs_per_dom_epoch != 0:
        dom_validations_left += 1  # keep one validation always for the last epoch
    dom_val_time = 8 * 60

    for epoch_idx in range(start_epoch, epochs):
        train_logger.info("starting epoch %d..." % epoch_idx)
        # we've already loaded some patients
        patients_num = len(training_fetcher.patient_dirs) + num_patients_in_buffer_train
        batches_per_epoch = patients_num * training_fetcher.load_img_shape[0] // batch_size_train
        batches_left = batches_per_epoch
        batches_per_sub_epoch = batches_per_epoch // sub_epochs

        # prepare val patients
        val_patient_num = len(validation_fetcher.patient_dirs) + validation_fetcher.patients_in_buffer
        val_patient_num_per_sub_epoch = val_patient_num // sub_epochs
        val_patient_left = val_patient_num

        # run sub-epochs
        for sub_epoch_idx in range(sub_epochs):
            train_logger.info("starting sub-epoch %d..." % sub_epoch_idx)

            # if this is the last sub-epoch, run all the batches left
            '''if sub_epoch_idx == sub_epochs - 1:
                batches_per_sub_epoch = batches_left
                val_patient_num_per_sub_epoch = val_patient_left'''

            # record sub-epoch time
            pbar = tqdm.tqdm(total=batches_per_sub_epoch)
            sub_epoch_start_time = time.time()
            # print("sub-epoch %d >| " % sub_epoch_idx, end='', flush=True)
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

                # print bar
                # bar = '='
                # if batch_idx % (batches_per_sub_epoch // 40) == 0:
                #     print(bar, end='', flush=True)

                batches_left -= 1

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

                    rand_vals = select_scenarios(batch_size_train, curr_scenario_range)
                else:  # not going to take curriculum learning?
                    rand_vals = select_scenarios(batch_size_train, [0, 14])

                # get scenarios for each slice in the batch
                label_scenarios = get_label_scenarios(scenarios, rand_vals)

                # get batch
                batch = training_fetcher.get_next_batch()
                if batch is None:
                    break
                batch_z = batch.copy()

                # create label list
                label_list_f = np.ones((batch_size_train, patch_size[0], patch_size[1], patch_size[2]))

                batch_z, label_list_f = create_impute_tensors_and_update_label_list(label_scenarios, batch_z,
                                                                                    label_list_f, img_shape)

                # get real batch output
                label_list_r = np.ones((batch_size_train, patch_size[0], patch_size[1], patch_size[2]))

                # train on batch
                batch_summary = train_on_batch(batch=batch, batch_z=batch_z, label_scenarios=label_scenarios,
                                               implicit_conditioning=implicit_conditioning, generator=generator,
                                               discriminator=discriminator, criterion_gan=criterion_gan,
                                               criterion_pixelwise=criterion_pixelwise,
                                               mse_fake_vs_real=mse_fake_vs_real, label_list_real=label_list_r,
                                               label_list_fake=label_list_f, lambda_param=lambda_param,
                                               optimizer_g=optimizer_g, optimizer_d=optimizer_d, debug=debug)

                # turn the tensors back to numpy values
                for each_key in batch_summary.keys():
                    batch_summary[each_key] = batch_summary[each_key].numpy().item()
                    batch_summaries[each_key].append(batch_summary[each_key])  # add them to summaries

                pbar.update()  # update pbar

            pbar.close()  # must close this
            # compute mean losses
            for each_key in batch_summaries.keys():
                sub_epoch_summary[each_key] = np.mean(np.array(batch_summaries[each_key])).item()
            train_logger.info("g_train_loss: %f; d_train_loss: %f; pixel_loss: %f; mse_loss: %f"
                              % (sub_epoch_summary['g_train_loss'], sub_epoch_summary['d_train_loss'],
                                 sub_epoch_summary['pixel_loss'], sub_epoch_summary['mse_loss']))

            # save dict_sum to json
            js_obj = json.dumps(batch_summaries)
            js_path = os.path.join(json_dir, 'training-%d-%d.json' % (epoch_idx, sub_epoch_idx))
            js_file = open(js_path, 'w')
            js_file.write(js_obj)
            js_file.close()

            # save epoch results to summary
            step = epoch_idx * sub_epochs + sub_epoch_idx
            with train_summary_writer.as_default():
                for each_key in batch_summaries.keys():
                    summary.scalar(each_key, sub_epoch_summary[each_key], step=step)

            # sub epoch validation
            # train_logger.info("start validation for this sub-epoch...")
            validation_fetcher.shuffle = True
            val_summary = validate(validation_fetcher=validation_fetcher, val_logger=val_logger, generator=generator,
                                   scenarios=scenarios, json_dir=json_dir, epoch=epoch_idx, sub_epoch=sub_epoch_idx,
                                   patient_num=val_patient_num_per_sub_epoch, debug=debug)

            # save val results to summary
            with val_summary_writer.as_default():
                for each_scenario in val_summary.keys():
                    summary.scalar('%s-mse' % each_scenario, val_summary[each_scenario]['mse'], step=step)
                    summary.scalar('%s-psnr' % each_scenario, val_summary[each_scenario]['psnr'], step=step)
                    summary.scalar('%s-ssim' % each_scenario, val_summary[each_scenario]['ssim'], step=step)
            val_patient_left -= val_patient_num_per_sub_epoch

            # print total time and average time of batches
            sub_epoch_end_time = time.time()
            sub_epoch_total_time = int(sub_epoch_end_time - sub_epoch_start_time)
            sub_epoch_total_time_hms = str(datetime.timedelta(seconds=sub_epoch_total_time))
            total_sub_epochs_left = sub_epochs * (epochs - epoch_idx - 1) + sub_epochs - sub_epoch_idx - 1
            total_time_left = int(total_sub_epochs_left * sub_epoch_total_time
                                  + reset_time * (epochs - epoch_idx)) + dom_val_time * dom_validations_left
            total_time_left_hms = str(datetime.timedelta(seconds=total_time_left))

            # output to log
            train_logger.info("epoch %d sub-epoch %d; time total: %s total_time_left: %s"
                              % (epoch_idx, sub_epoch_idx, sub_epoch_total_time_hms, total_time_left_hms))

        checkpoint_manager.save(checkpoint_number=epoch_idx)
        train_logger.info("===EPOCH=FINISH===")

        # dom-validation
        if (epoch_idx + 1) % epochs_per_dom_epoch == 0 or (epoch_idx + 1) == epochs:
            train_logger.info("start dom validation...")
            val_start_time = time.time()
            validation_fetcher.shuffle = False
            validation_fetcher.reset()
            val_summary = validate(validation_fetcher=validation_fetcher, val_logger=val_logger, generator=generator,
                                   scenarios=scenarios, json_dir=json_dir, epoch=epoch_idx, sub_epoch=-1,
                                   patient_num=val_patient_num, debug=debug)
            val_end_time = time.time()
            dom_val_time = int(val_end_time - val_start_time)
            dom_validations_left -= 1

            # save val results to summary
            with val_summary_writer.as_default():
                for each_scenario in val_summary.keys():
                    summary.scalar('%s-mse-dom' % each_scenario, val_summary[each_scenario]['mse'], step=epoch_idx)
                    summary.scalar('%s-psnr-dom' % each_scenario, val_summary[each_scenario]['psnr'], step=epoch_idx)
                    summary.scalar('%s-ssim-dom' % each_scenario, val_summary[each_scenario]['ssim'], step=epoch_idx)

        if epoch_idx != epochs - 1:
            reset_start_time = time.time()
            training_fetcher.reset()  # reset this
            validation_fetcher.reset()
            reset_end_time = time.time()
            reset_time = int(reset_end_time - reset_start_time)
            if epoch_idx == epochs - 2:
                reset_time = 0
        else:
            if training_fetcher.loader_process is not None:
                training_fetcher.loader_process.close()
            if validation_fetcher.loader_process is not None:
                validation_fetcher.loader_process.close()


# @function  # tf.function
def train_on_batch(batch: np.ndarray, batch_z: np.ndarray, label_scenarios: list, implicit_conditioning: bool,
                   generator, discriminator, criterion_gan, criterion_pixelwise, mse_fake_vs_real,
                   label_list_fake: np.ndarray, label_list_real: np.ndarray, lambda_param: float,
                   optimizer_g, optimizer_d, debug: bool) -> dict:
    # init tape
    with GradientTape() as g_tape, GradientTape() as d_tape:

        # get fake img
        fake_x = generator(batch_z, training=True)

        # implicit conditioning
        if implicit_conditioning:
            # fake_x = py_function(func=impute_reals_into_fake, inp=(batch_z, fake_x, label_scenarios), Tout=float32)
            fake_x = impute_reals_into_fake(batch_z, fake_x, label_scenarios)

        # predict the fake & real
        pred_fake = discriminator([fake_x, batch], training=True)
        pred_real = discriminator([batch, batch], training=True)

        # loss from discriminator
        label_list_real = cast(label_list_real, dtype=float32)
        label_list_fake = cast(label_list_fake, dtype=float32)
        '''if debug:
            channels = ['t1', 't2', 't1ce', 'flair']
            for idx, label in enumerate(label_scenarios):
                label_str = ''
                for each in label:
                    label_str += str(each)
                for i in range(4):
                    plt.figure()
                    plt.imshow(fake_x[idx][i])
                    label_title = label_str + channels[i] + "fake_x"
                    plt.title(label_title)
                    plt.figure()
                    plt.imshow(pred_real[idx][i])
                    label_title = label_str + channels[i] + "pred_real"
                    plt.title(label_title)
                    plt.figure()
                    plt.imshow(pred_fake[idx][i])
                    label_title = label_str + channels[i] + "pred_fake"
                    plt.title(label_title)
            plt.show()'''
        loss_gan = criterion_gan(pred_fake, label_list_real)  # generator should make discriminator think it is real
        loss_real = criterion_gan(pred_real, label_list_real)  # discriminator should think the real img real
        loss_fake = criterion_gan(pred_fake, label_list_fake)  # discriminator should think the fake img fake

        if implicit_conditioning:
            loss_pixel = 0
            synth_loss = 0
            count = 0
            for num_slice, label_scenario in enumerate(label_scenarios):
                for idx_curr_label, i in enumerate(label_scenario):
                    if i == 0:
                        loss_pixel += criterion_pixelwise(fake_x[num_slice, idx_curr_label, ...],
                                                          batch[num_slice, idx_curr_label, ...])

                        synth_loss += mse_fake_vs_real(fake_x[num_slice, idx_curr_label, ...],
                                                       batch[num_slice, idx_curr_label, ...])
                        count += 1
            loss_pixel /= count
            synth_loss /= count
        else:  # no IC, calculate loss for all output w.r.t all GT.
            loss_pixel = criterion_pixelwise(fake_x, batch)
            synth_loss = mse_fake_vs_real(fake_x, batch)

        # total generator loss
        lambda_param = cast(lambda_param, dtype=float32)
        g_train_total_loss = (constant(1, dtype=float32) - lambda_param) * loss_gan + lambda_param * loss_pixel

        # total discriminator loss
        d_train_total_loss = constant(0.5, dtype=float32) * (loss_fake + loss_real)

    # apply gradients
    gradients_of_generator = g_tape.gradient(g_train_total_loss, generator.trainable_variables)
    gradients_of_discriminator = d_tape.gradient(d_train_total_loss, discriminator.trainable_variables)
    optimizer_g.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    optimizer_d.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # 'Tensor' object has no attribute 'numpy' in eager training
    '''dict_summary = {
        'g_train_loss': g_train_total_loss.numpy().item(),
        'd_train_loss': d_train_total_loss.numpy().item(),
        'pixel_loss': loss_pixel.numpy().item(),
        'mse_loss': synth_loss.numpy().item()
    }'''
    dict_summary = {
        'g_train_loss': g_train_total_loss,
        'd_train_loss': d_train_total_loss,
        'pixel_loss': loss_pixel,
        'mse_loss': synth_loss
    }

    return dict_summary


def validate(validation_fetcher: data_fetcher.DataFetcher, val_logger: logging.Logger, generator, scenarios: list,
             json_dir: str, epoch: int, sub_epoch: int, patient_num: int, debug: bool) -> dict:
    """
    sub & dom validation
    :param sub_epoch: -1 means dom-validation
    """
    if sub_epoch != -1:
        val_logger.info("epoch %d sub-epoch %d, validating" % (epoch, sub_epoch))
    else:
        val_logger.info("epoch %d, dom-validating" % epoch)
    # print("validation-->| ", end='', flush=True)

    metrics = []
    for i in range(len(scenarios)):
        metrics.append({'mse': [], 'psnr': [], 'ssim': []})

    mse = MeanSquaredError()

    # we've already loaded some patients
    # patient_num = len(validation_fetcher.patient_dirs) + validation_fetcher.patients_in_buffer
    batch_num = patient_num * validation_fetcher.load_img_shape[0] // validation_fetcher.batch_size

    pbar = tqdm.tqdm(total=batch_num)
    # print("sub-epoch %d >| " % sub_epoch_idx, end='', flush=True)
    if sub_epoch != -1:
        pbar.set_description("sub-epoch %d validation" % sub_epoch)
    else:
        pbar.set_description("dom-validation for epoch %d" % epoch)

    for idx in range(batch_num):
        batch = validation_fetcher.get_next_batch()  # get next batch
        sc_idx = 0
        for each_scenario in scenarios:
            batch_z = batch.copy()

            for s in range(batch.shape[0]):
                for c, each_channel in enumerate(each_scenario):
                    if each_channel == 0:
                        batch_z[s, c] = np.zeros(shape=(batch.shape[2], batch.shape[3]), dtype=np.float32)

            # with device(dev):
            fake_img = generator(batch_z)

            if debug:
                scenario_str = ''
                for each in each_scenario:
                    scenario_str += str(each)
                plt.figure()
                if each_scenario[0] == 0:
                    plt.imshow(fake_img[0, 0, ...])
                else:
                    plt.imshow(batch_z[0, 0, ...])
                plt.title("%s - t1" % scenario_str)
                plt.figure()
                if each_scenario[1] == 0:
                    plt.imshow(fake_img[0, 1, ...])
                else:
                    plt.imshow(batch_z[0, 1, ...])
                plt.title("%s - t2" % scenario_str)
                plt.figure()
                if each_scenario[2] == 0:
                    plt.imshow(fake_img[0, 2, ...])
                else:
                    plt.imshow(batch_z[0, 2, ...])
                plt.title("%s - t1ce" % scenario_str)
                plt.figure()
                if each_scenario[3] == 0:
                    plt.imshow(fake_img[0, 3, ...])
                else:
                    plt.imshow(batch_z[0, 3, ...])
                plt.title("%s - flair" % scenario_str)
                plt.show()

            # scenario_metrics = {'mse': [], 'psnr': [], 'ssim': []}
            for c, each_channel in enumerate(each_scenario):
                if each_channel == 0:
                    fake_channel_arr = fake_img[:, c, ...]
                    real_channel_arr = batch[:, c, ...]
                    # max_val = max(fake_channel_arr.numpy().max(), real_channel_arr.numpy().max())
                    max_val = 1.0
                    metrics[sc_idx]['mse'].append(mse(fake_channel_arr, real_channel_arr).numpy().item())
                    metrics[sc_idx]['psnr'].append(psnr(fake_channel_arr, real_channel_arr, max_val).numpy().item())
                    metrics[sc_idx]['ssim'].append(ssim(fake_channel_arr, real_channel_arr, max_val).numpy().item())
            sc_idx += 1

        pbar.update()  # update the bar

    epoch_rst_dict = {}
    sc_idx = 0
    print("\n", end="", flush=True)
    for scenario in scenarios:
        scenario_str = ''
        for each in scenario:
            scenario_str += str(each)
        epoch_rst_dict[scenario_str] = {
            "mse": np.array(metrics[sc_idx]['mse']).mean(),
            "psnr": np.array(metrics[sc_idx]['psnr']).mean(),
            "ssim": np.array(metrics[sc_idx]['ssim']).mean()
        }
        val_logger.info("%s: mse: %f; psnr: %f; ssim: %f" % (scenario_str, epoch_rst_dict[scenario_str]["mse"],
                                                             epoch_rst_dict[scenario_str]["psnr"],
                                                             epoch_rst_dict[scenario_str]["ssim"]))
        print("%s: mse: %f; psnr: %f; ssim: %f" % (scenario_str, epoch_rst_dict[scenario_str]["mse"],
                                                   epoch_rst_dict[scenario_str]["psnr"],
                                                   epoch_rst_dict[scenario_str]["ssim"]))
        sc_idx += 1

    # json
    js_obj = json.dumps(epoch_rst_dict)
    if sub_epoch != -1:
        js_path = os.path.join(json_dir, 'validation-%d-%d.json' % (epoch, sub_epoch))
    else:
        js_path = os.path.join(json_dir, 'dom-validation-%d.json' % epoch)
    js_file = open(js_path, 'w')
    js_file.write(js_obj)
    js_file.close()
    pbar.close()

    return epoch_rst_dict


if __name__ == '__main__':
    train(session_name="test_02", output_dir=r"E:\my_files\programmes\python\mri_gan_output",
          group_txt_names_train=['group0_standardized.txt', 'group1_standardized.txt',
                                 'group2_standardized.txt', 'group3_standardized.txt'],
          group_txt_names_val=['group4_standardized.txt'],
          group_txt_dir_train=r"E:\my_files\programmes\python\BRATS2018_normalized",
          group_txt_dir_val=r"E:\my_files\programmes\python\BRATS2018_normalized",
          channels=['t1', 't2', 't1ce', 'flair'], batch_size_train=8, num_patients_in_buffer_train=8,
          batch_size_val=32, num_patients_in_buffer_val=8, nii_shape=(155, 240, 240), slice_cut_off=(12, 12),
          img_shape=(256, 256), learning_rate=0.0002, beta_1=0.5, beta_2=0.999, lambda_param=0.9, epochs=60,
          epochs_per_dom_epoch=10, sub_epochs=10, implicit_conditioning=True, curriculum_learning=False, debug=True)
