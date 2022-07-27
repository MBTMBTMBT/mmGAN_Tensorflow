import os
import random
import numpy as np
import SimpleITK as sitk
import multiprocessing
import math
import tensorflow as tf
import mm_gan.io_helpers as io_helpers
import tqdm
import shutil

from skimage.transform import resize
from mm_gan.constants_and_tools import TXT_NAMES, CHANNELS, make_dir, TXT_OUT_NAMES, SLICE_RANGE


def arrange_data_into_5_folds(parent_path: str, output_dir: str, shuffle=True):
    paths = get_patients_with_full_path(parent_path)
    if shuffle:
        random.shuffle(paths)
    patients_num = len(paths)
    patients_num_in_each_group = patients_num // 5
    patient_count = 0
    make_dir(output_dir)
    for i_group in range(5):
        file = open(os.path.join(output_dir, 'group%d.txt' % i_group), 'w')
        for j_patient in range(patients_num_in_each_group):
            file.write(paths[patient_count] + '\n')
            patient_count += 1
        if i_group == 4:
            file.writelines(paths[patient_count:-1])
        file.close()


# def group_preprocess(group_txt_path: str, output_dir: str, channels: list, mean: float, std: float):
def group_preprocess(group_txt_path: str, output_dir: str, channels: list):
    make_dir(output_dir)
    group_txt = open(group_txt_path, 'r')
    output_txt_name = group_txt_path.split('.')[0] + '_standardized.txt'
    output_group_txt = open(output_txt_name, 'w')
    next_patient = group_txt.readline()
    while next_patient:
        next_patient = next_patient.replace("\n", "")
        nii_list = os.listdir(next_patient)
        patient_name = os.path.split(next_patient)[-1]
        each_output_dir = os.path.join(output_dir, patient_name)
        output_group_txt.write(each_output_dir + '\n')
        for each_name in nii_list:
            if '.nii.gz' in each_name:
                for each_channel in channels:
                    if each_channel in each_name:
                        if each_channel == "t1" and "t1ce" in each_name:
                            each_channel = "t1ce"
                        print("processing: %s - %s" %
                              (each_name, each_channel))
                        '''
                        nii_to_nii_standardize(os.path.join(next_patient, each_name),
                                            os.path.join(
                                                each_output_dir, each_channel + '_standardized.nii.gz'),
                                            mean, std)
                        '''
                        nii_to_nii_standardize(os.path.join(next_patient, each_name),
                                               os.path.join(
                            each_output_dir, each_channel + '_standardized.nii.gz'),
                        )
                        break
        next_patient = group_txt.readline()
    group_txt.close()


def preprocess(group_text_dir: str, group_text_names: list, output_root_dir: str, channels=None):
    if channels is None:
        channels = CHANNELS

    ''' we don't need these now
    print("computing mean & std values...")
    mean = get_mean(group_text_dir, group_text_names, channels)
    print("mean: ", mean)
    std = get_std(group_text_dir, group_text_names, mean, channels)
    print("std: ", std)
    '''

    processes = []
    for idx, each_text_name in enumerate(group_text_names):
        each_dir = os.path.join(group_text_dir, each_text_name)
        output_dir = os.path.join(output_root_dir, 'fold%d' % idx)
        make_dir(output_dir)

        '''
        process = multiprocessing.Process(target=group_preprocess, args=(
            each_dir, output_dir, channels, mean, std))
        '''

        process = multiprocessing.Process(target=group_preprocess, args=(
            each_dir, output_dir, channels))
        process.start()
        processes.append(process)
    for each_process in processes:
        each_process.join()
        each_process.close()


def get_files_with_full_path(path: str, channel_name=None) -> list:
    names = os.listdir(path)
    path_list = [os.path.join(path, name) for name in names]
    for path in path_list:
        files = os.listdir(path)
        for each_file in files:
            if channel_name is not None and channel_name in each_file:
                print(os.path.join(path, each_file))


def get_patients_with_full_path(parent_path: str) -> list:
    paths = []
    names = os.listdir(parent_path)
    path_list = [os.path.join(parent_path, name) for name in names]
    for parent_path in path_list:
        # print(path)
        paths.append(parent_path)
    return paths


# def standardize_channel(img: np.ndarray, mean_value, std_value) -> np.ndarray:
def standardize_channel(img: np.ndarray) -> np.ndarray:
    # normalized = np.empty(img.shape, dtype='float32')
    # mean_value, std_value = np.mean(img), np.std(img)
    '''
    for s in range(img.shape[0]):
    normalized[s, ...] = (img[s, ...] - mean_value) / std_value
    '''
    '''cropped = img[:, 29:223, 41:196]
    normalized = cropped / np.mean(cropped).item()
    resized = np.empty(shape=(cropped.shape[0], 256, 256))
    for i in range(resized.shape[0]):
        resized[i] = skimage.transform.resize(normalized[i], output_shape=(256, 256), preserve_range=True)'''
    # print("mean: ", np.mean(normalized[s, ...]), " std: ", np.std(normalized[s, ...]))
    normalized = img / np.mean(img[:, 29:223, 41:196]).item()
    # normalized = img / np.mean(img).item()
    # normalized = resized
    # normalized = (img - np.mean(img[:, 29:223, 41:196])) / np.std(img[:, 29:223, 41:196])
    # normalized[np.isnan(normalized)] = 0.0  # remove all nan
    normalized[normalized < 0] = 0.0
    return normalized


def sum_all(group_txt_path: str, channels: list, shared_count, shared_sum, shared_lock):
    group_txt = open(group_txt_path, 'r')
    next_patient = group_txt.readline()
    while next_patient:
        next_patient = next_patient.replace("\n", "")
        nii_list = os.listdir(next_patient)
        for each_name in nii_list:
            if '.nii.gz' in each_name:
                for each_channel in channels:
                    if each_channel in each_name:
                        # print(each_name)
                        each_name = os.path.join(next_patient, each_name)
                        label_img = sitk.ReadImage(each_name)
                        arr_img = sitk.GetArrayFromImage(label_img)
                        shared_lock.acquire()
                        # print("mean: %s - %s" % (each_name, each_channel))
                        shared_sum.value += np.sum(arr_img)
                        shared_count.value += arr_img.shape[0] * \
                            arr_img.shape[1] * arr_img.shape[2]
                        shared_lock.release()
        next_patient = group_txt.readline()
    group_txt.close()


def sum_difference_square(group_txt_path: str, channels: list, mean: float, shared_count, shared_sum, shared_lock):
    group_txt = open(group_txt_path, 'r')
    next_patient = group_txt.readline()
    while next_patient:
        next_patient = next_patient.replace("\n", "")
        nii_list = os.listdir(next_patient)
        for each_name in nii_list:
            if '.nii.gz' in each_name:
                for each_channel in channels:
                    if each_channel in each_name:
                        each_name = os.path.join(next_patient, each_name)
                        label_img = sitk.ReadImage(each_name)
                        arr_img = sitk.GetArrayFromImage(label_img)
                        shared_lock.acquire()
                        # print("std: %s - %s" % (each_name, each_channel))
                        shared_sum.value += np.sum(np.square(arr_img - mean))
                        shared_count.value += arr_img.shape[0] * \
                            arr_img.shape[1] * arr_img.shape[2]
                        shared_lock.release()
        next_patient = group_txt.readline()
    group_txt.close()


def get_mean(group_text_dir: str, group_text_names: list, channels=None) -> float:
    shared_sum = multiprocessing.Manager().Value(float, 0.0)
    shared_count = multiprocessing.Manager().Value(int, 0)
    shared_lock = multiprocessing.Manager().Lock()
    processes = []
    if channels is None:
        channels = CHANNELS
    for idx, each_text_name in enumerate(group_text_names):
        each_dir = os.path.join(group_text_dir, each_text_name)
        process \
            = multiprocessing.Process(target=sum_all, args=(each_dir, channels, shared_count, shared_sum, shared_lock))
        process.start()
        # process.join()
        processes.append(process)
    for each_process in processes:
        each_process.join()
    mean_value = shared_sum.value / shared_count.value
    print("sum: ", shared_sum.value, " count: ", shared_count.value)
    return mean_value


def get_std(group_text_dir: str, group_text_names: list, mean: float, channels=None) -> float:
    shared_sum = multiprocessing.Manager().Value(float, 0.0)
    shared_count = multiprocessing.Manager().Value(int, 0)
    shared_lock = multiprocessing.Manager().Lock()
    processes = []
    if channels is None:
        channels = CHANNELS
    for idx, each_text_name in enumerate(group_text_names):
        each_dir = os.path.join(group_text_dir, each_text_name)
        process = multiprocessing.Process(target=sum_difference_square,
                                          args=(each_dir, channels, mean, shared_count, shared_sum, shared_lock))
        process.start()
        processes.append(process)
    for each_process in processes:
        each_process.join()
    std_value = math.sqrt(shared_sum.value / shared_count.value)
    print("sum: ", shared_sum.value, " count: ", shared_count.value)
    return std_value


# def nii_to_nii_standardize(img_path: str, output_path: str, mean: float, std: float):
def nii_to_nii_standardize(img_path: str, output_path: str):
    label_img = sitk.ReadImage(img_path)
    arr_img = sitk.GetArrayFromImage(label_img)
    # standardized_arr_img = standardize_channel(arr_img, mean, std)
    standardized_arr_img = standardize_channel(arr_img)
    standardized_img = sitk.GetImageFromArray(standardized_arr_img)
    make_dir(output_path)
    sitk.WriteImage(standardized_img, output_path)


def nii_to_tfrecord(txt_path: str, txt_names: list, output_dir: str, channels: list, operation: str, slice_range: tuple,
                    crop_cord=None):
    # assert operation in ['padding', 'crop', 'none']  # bad code
    if operation == 'crop':
        assert crop_cord is not None
    txt_paths = []
    for each_txt_name in txt_names:
        txt_paths.append(os.path.join(txt_path, each_txt_name))
    for each_txt_path, each_txt_name in zip(txt_paths, txt_names):
        patient_dirs = []
        file = open(each_txt_path, 'r')
        next_path = file.readline()
        while next_path:
            next_path = next_path.replace('\n', '')
            patient_dirs.append(next_path)
            next_path = file.readline()
        file.close()
        each_txt_name = each_txt_name.split(".")[0]
        writer_options = tf.io.TFRecordOptions(
            compression_type='ZLIB', compression_level=9)
        patients = len(patient_dirs)
        slices = patients * (slice_range[1] - slice_range[0])
        output_path = os.path.join(output_dir, "%s_%dpatients_%dslices.tfrecord" % (
            each_txt_name, patients, slices))
        make_dir(output_dir)
        writer = tf.io.TFRecordWriter(path=output_path, options=writer_options)
        for each_patient_dir in tqdm.tqdm(patient_dirs, desc="Writing to %s" % output_path):
            patient_name = os.path.split(each_patient_dir)[-1]
            # print("saving %s to tfrecord..." % patient_name)
            img_four_channel = None
            for channel_idx, each_channel in enumerate(channels):
                file_path = os.path.join(
                    each_patient_dir, "%s_standardized.nii.gz" % each_channel)
                img_arr = io_helpers.load_nii(file_path)
                one_c_shape = img_arr.shape
                if img_four_channel is None:
                    if operation == 'padding' or operation == 'crop':
                        assert one_c_shape[1] == 240 and one_c_shape[2] == 240
                        img_four_channel = np.ones(
                            shape=(one_c_shape[0], 4, 256, 256)) * img_arr[0, 0, 0]
                    else:
                        img_four_channel = np.empty(
                            shape=(one_c_shape[0], 4, one_c_shape[1], one_c_shape[2]))
                if operation == 'padding':
                    img_four_channel[:, channel_idx, 8:248, 8:248] = img_arr
                elif operation == 'crop':
                    # print(crop_cord)
                    for img_idx in range(one_c_shape[0]):
                        img_four_channel[img_idx, channel_idx, :, :] \
                            = resize(
                                img_arr[img_idx, crop_cord[0]:crop_cord[1], crop_cord[2]:crop_cord[3]],
                                output_shape=(256, 256),
                                preserve_range=True
                            )
                else:
                    img_four_channel[:, channel_idx, :, :] = img_arr

            for slice_idx in range(slice_range[0], slice_range[1]):
                features = tf.train.Features(
                    feature={
                        "name": tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(patient_name, encoding="utf-8")])),
                        "t1": tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[img_four_channel[slice_idx, 0, ...].astype(np.float32).tobytes()])),
                        "t2": tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[img_four_channel[slice_idx, 1, ...].astype(np.float32).tobytes()])),
                        "t1ce": tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[img_four_channel[slice_idx, 2, ...].astype(np.float32).tobytes()])),
                        "flair": tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[img_four_channel[slice_idx, 3, ...].astype(np.float32).tobytes()])),
                    }
                )

                example = tf.train.Example(features=features)
                serialized = example.SerializeToString()
                writer.write(serialized)


def move_segmasks_to_folds(raw_group_txt_dir: str, raw_group_txt_names: list, std_group_txt_dir: str, std_group_txt_names: list):
    in_txt_dirs = []
    for each in raw_group_txt_names:
        in_txt_dirs.append(os.path.join(raw_group_txt_dir, each))
    out_txt_dirs = []
    for each in std_group_txt_names:
        out_txt_dirs.append(os.path.join(std_group_txt_dir, each))
    in_dirs = []
    for each_file in in_txt_dirs:
        group_txt = open(each_file, 'r')
        next_patient = group_txt.readline()
        while next_patient:
            next_patient = next_patient.replace("\n", "")
            print(next_patient)
            in_dirs.append(next_patient)
            next_patient = group_txt.readline()
        group_txt.close()
    out_dirs = []
    for each_file in out_txt_dirs:
        group_txt = open(each_file, 'r')
        next_patient = group_txt.readline()
        while next_patient:
            next_patient = next_patient.replace("\n", "")
            print(next_patient)
            out_dirs.append(next_patient)
            next_patient = group_txt.readline()
        group_txt.close()
    for idx, each in enumerate(in_dirs):
        patient_name = os.path.split(each)[-1]
        in_seg_path = os.path.join(each, "%s_seg.nii.gz" % patient_name)
        out_seg_path = os.path.join(
            out_dirs[idx], "seg.nii.gz")
        shutil.copy(in_seg_path, out_seg_path)


def move_segmasks_to_folds_and_remove_some_slices(
    raw_group_txt_dir: str, raw_group_txt_names: list,
    std_group_txt_dir: str, std_group_txt_names: list,
    slices_range: tuple,
):
    in_txt_dirs = []
    for each in raw_group_txt_names:
        in_txt_dirs.append(os.path.join(raw_group_txt_dir, each))
    out_txt_dirs = []
    for each in std_group_txt_names:
        out_txt_dirs.append(os.path.join(std_group_txt_dir, each))
    in_dirs = []
    for each_file in in_txt_dirs:
        group_txt = open(each_file, 'r')
        next_patient = group_txt.readline()
        while next_patient:
            next_patient = next_patient.replace("\n", "")
            print(next_patient)
            in_dirs.append(next_patient)
            next_patient = group_txt.readline()
        group_txt.close()
    out_dirs = []
    for each_file in out_txt_dirs:
        group_txt = open(each_file, 'r')
        next_patient = group_txt.readline()
        while next_patient:
            next_patient = next_patient.replace("\n", "")
            print(next_patient)
            out_dirs.append(next_patient)
            next_patient = group_txt.readline()
        group_txt.close()
    for idx, each in enumerate(in_dirs):
        patient_name = os.path.split(each)[-1]
        in_seg_path = os.path.join(each, "%s_seg.nii.gz" % patient_name)
        out_seg_path = os.path.join(
            out_dirs[idx], "seg_preprocessed.nii.gz")
        # shutil.copy(in_seg_path, out_seg_path)
        in_arr = io_helpers.load_nii(in_seg_path)
        out_arr = in_arr[slices_range[0]: slices_range[1]]
        io_helpers.write_nii(out_arr, out_seg_path)


if __name__ == '__main__':
    pass
    # get_files_with_full_path(r"D:\python\BRATS2018\validation", "seg")
    # arrange_data_into_5_folds(r"D:\python\BRATS2018\training\HGG", shuffle=True)
    # standardize_channel(np.random.randn(155, 240, 240))

    '''arrange_data_into_5_folds(r"E:\my_files\programmes\python\BRATS2018\training\HGG",
                              r"E:\my_files\programmes\python\BRATS2018_normalized", shuffle=True)'''

    preprocess(r"E:\my_files\programmes\python\BRATS2018_normalized", TXT_NAMES,
               r"E:\my_files\programmes\python\BRATS2018_normalized", CHANNELS)
    nii_to_tfrecord(r"E:\my_files\programmes\python\BRATS2018_normalized", TXT_OUT_NAMES,
                    r"E:\my_files\programmes\python\BRATS2018_normalized", CHANNELS, operation='padding', slice_range=SLICE_RANGE)
    move_segmasks_to_folds(r"E:\my_files\programmes\python\BRATS2018_normalized",
                           ["group0.txt", "group1.txt", "group2.txt",
                               "group3.txt", "group4.txt"],
                           r"E:\my_files\programmes\python\BRATS2018_normalized",
                           ["group0_standardized.txt", "group1_standardized.txt", "group2_standardized.txt", "group3_standardized.txt", "group4_standardized.txt"])
    move_segmasks_to_folds_and_remove_some_slices(r"E:\my_files\programmes\python\BRATS2018_normalized",
                                                  ["group0.txt", "group1.txt", "group2.txt",
                                                   "group3.txt", "group4.txt"],
                                                  r"E:\my_files\programmes\python\BRATS2018_normalized",
                                                  ["group0_standardized.txt", "group1_standardized.txt", "group2_standardized.txt",
                                                   "group3_standardized.txt", "group4_standardized.txt"],
                                                  slices_range=SLICE_RANGE)
