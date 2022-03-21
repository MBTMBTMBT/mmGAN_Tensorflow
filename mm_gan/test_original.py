import logging
import os.path
import json
import SimpleITK as sitk
import tqdm
import mm_gan.data_fetcher
import itertools
import numpy as np
# import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from constants_and_tools import make_dir
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.image import psnr
from mm_gan.models import get_generator_unet
from tensorflow.train import Checkpoint


def test(parameter_path: str, session_name: str, output_dir: str,
         group_txt_dir_test: str, group_txt_names_test: list, channels: list,
         nii_shape: tuple, img_shape: tuple) -> dict:

    generator = get_generator_unet(input_shape=(
        4, img_shape[0], img_shape[1]), out_channels=4)
    checkpoint = Checkpoint(generator)
    checkpoint.restore(parameter_path)

    # create sc scenarios from 0000 to 1111, yet 0000 and 1111 should not be used
    scenarios = list(map(list, itertools.product([0, 1], repeat=4)))
    scenarios.remove([0, 0, 0, 0])
    scenarios.remove([1, 1, 1, 1])

    # sort the scenarios according to decreasing difficulty. Easy scenarios last, and difficult ones first.
    scenarios.sort(key=lambda x: x.count(1))

    txt_paths = []
    group_names = []
    patient_dirs = []
    patient_names = []

    for each_txt_name in group_txt_names_test:
        txt_paths.append(os.path.join(group_txt_dir_test, each_txt_name))
        group_names.append(each_txt_name.split('.')[0])

    for idx, each_txt_path in enumerate(txt_paths):
        # self.logger.debug("loading patients' path from %s" % self.group_names[idx])
        file = open(each_txt_path, 'r')
        next_path = file.readline()
        while next_path:
            next_path = next_path.replace('\n', '')
            patient_dirs.append(next_path)
            patient_names.append(os.path.split(next_path)[-1])
            next_path = file.readline()
        file.close()

    test_fetcher = data_fetcher.DataFetcher(group_txt_dir=group_txt_dir_test, group_txt_names=group_txt_names_test,
                                            channels=channels, batch_size=1,
                                            patients_in_buffer=1, nii_shape=nii_shape,
                                            shuffle=False, img_shape=img_shape,
                                            name="validation fetcher")

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

    # batch_num = len(test_fetcher.patient_dirs) * test_fetcher.load_img_shape[0] // test_fetcher.batch_size

    for each_patient in patient_names:
        test_logger.info("testing patient %s" % each_patient)
        metrics = []
        slices = []
        for i in range(len(scenarios)):
            metrics.append({'mse': [], 'psnr': [], 'ssim': []})
            slices.append([])
        pbar = tqdm.tqdm(total=nii_shape[0])
        pbar.set_description("testing patient %s" % each_patient)
        for idx in range(nii_shape[0]):
            batch = test_fetcher.get_next_batch()  # get next batch
            sc_idx = 0
            for s_idx, each_scenario in enumerate(scenarios):
                batch_z = batch.copy()

                for s in range(batch.shape[0]):
                    for c, each_channel in enumerate(each_scenario):
                        if each_channel == 0:
                            batch_z[s, c] = np.zeros(
                                shape=(batch.shape[2], batch.shape[3]), dtype=np.float32)

                # with device(dev):
                fake_img = generator(batch_z)
                slices[s_idx].append(fake_img)
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
                            psnr(fake_channel_arr, real_channel_arr, max_val).numpy().item())
                        # print(fake_channel_arr.shape, real_channel_arr.shape)
                        '''metrics[sc_idx]['ssim'].append(
                            ssim(fake_channel_arr[0], real_channel_arr[0], max_val).numpy().item())'''
                sc_idx += 1

            pbar.update()  # update the bar

        epoch_rst_dict = {}
        sc_idx = 0
        print("\n", end="", flush=True)
        for s_idx, scenario in enumerate(scenarios):
            scenario_str = ''
            for each in scenario:
                scenario_str += str(each)
            epoch_rst_dict[scenario_str] = {
                "mse": np.array(metrics[sc_idx]['mse']).mean(),
                "psnr": np.array(metrics[sc_idx]['psnr']).mean()
                # '''"ssim": np.array(metrics[sc_idx]['ssim']).mean()'''
            }
            test_logger.info("%s: mse: %f; psnr: %f" % (scenario_str, epoch_rst_dict[scenario_str]["mse"],
                                                                  epoch_rst_dict[scenario_str]["psnr"]))
            print("%s: mse: %f; psnr: %f" % (scenario_str, epoch_rst_dict[scenario_str]["mse"],
                                                       epoch_rst_dict[scenario_str]["psnr"]))
            sc_idx += 1
            s_slices = slices[s_idx]  
            s_slices = np.squeeze(np.array(s_slices)).transpose(1, 0, 2, 3)  # 155, 4, 256, 256 -> 4, 155, 256, 256
            print(s_slices.shape)
            for i in range(4):
                c_slices = s_slices[i]
                channel = channels[i]
                img = sitk.GetImageFromArray(c_slices)
                file_path = os.path.join(output_dir, each_patient)
                file_path = os.path.join(file_path, scenario_str)
                # file_path = os.path.join(file_path, channel)
                make_dir(file_path)
                file_path = os.path.join(file_path, "%s.nii.gz" % channel)
                sitk.WriteImage(img, file_path)

        # json
        js_obj = json.dumps(epoch_rst_dict)
        js_path = os.path.join(output_dir, 'test-%s.json' % each_patient)
        js_file = open(js_path, 'w')
        js_file.write(js_obj)
        js_file.close()
        pbar.close()

    return epoch_rst_dict


if __name__ == '__main__':
    test(parameter_path=r"E:\my_files\programmes\python\mri_gan_output\test_00\saved_checkpoints\ckpt-48",
         session_name="test", output_dir=r"E:\my_files\programmes\python\mri_gan_output\test_00",
         group_txt_dir_test=r"E:\my_files\programmes\python\BRATS2018_normalized",
         group_txt_names_test=["group4_standardized.txt"], channels=['t1', 't2', 't1ce', 'flair'],
         nii_shape=(155, 240, 240), img_shape=(256, 256))
